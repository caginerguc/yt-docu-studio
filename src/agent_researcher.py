import json
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import List
import feedparser
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    HISTORY_FILE,
    HISTORY_RETENTION_DAYS,
    TARGET_LANGUAGE,
    get_active_config
)

# Loglama Ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Agent1_Researcher")


# ==========================================
# PYDANTIC ŞEMASI (GEMINI ÇIKTI GARANTİSİ)
# ==========================================
class SelectedTopicSchema(BaseModel):
    topic_id: str = Field(
        description="Konuyu tanımlayan benzersiz, küçük harfli ve tireli İngilizce slug (örn: 'eu-ai-act-chip-shortage-2026')."
    )
    original_headline: str = Field(
        description="Seçilen haberin RSS akışındaki orijinal başlığı."
    )
    source_url: str = Field(
        description="Seçilen haberin kaynak URL'si."
    )
    core_event_summary: str = Field(
        description="Telif hakkı (Leistungsschutzrecht) ihlali yaratmamak için haber metninden bağımsız, salt olgusal (kim, ne, nerede, neden) 3-4 cümlelik tarafsız özet."
    )
    editorial_angle: str = Field(
        description="Bu haber neden önemli? Belgeselde işlenecek derinlikli ekonomik, teknolojik veya jeopolitik analiz açısı."
    )
    viral_score: int = Field(
        description="Konunun izlenme ve tartışma yaratma potansiyeli (1-100 arası).",
        ge=1,
        le=100
    )


# ==========================================
# HAFIZA YÖNETİMİ (history.json)
# ==========================================
class TopicMemory:
    def __init__(self, filepath=HISTORY_FILE, retention_days=HISTORY_RETENTION_DAYS):
        self.filepath = filepath
        self.retention_days = retention_days
        self._ensure_file()

    def _ensure_file(self):
        if not self.filepath.exists():
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def load_and_prune(self) -> List[dict]:
        """Hafızayı yükler ve 30 günden eski kayıtları otomatik siler."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                records = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            records = []

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        valid_records = []

        for rec in records:
            try:
                rec_date = datetime.fromisoformat(rec.get("timestamp"))
                if rec_date >= cutoff_date:
                    valid_records.append(rec)
            except (ValueError, TypeError):
                continue

        if len(valid_records) != len(records):
            self._save_raw(valid_records)
            logger.info(f"Hafıza temizlendi: {len(records) - len(valid_records)} eski kayıt silindi.")

        return valid_records

    def save_topic(self, topic: SelectedTopicSchema, language: str):
        """Yeni seçilen konuyu hafızaya kaydeder."""
        records = self.load_and_prune()
        new_entry = {
            "topic_id": topic.topic_id,
            "original_headline": topic.original_headline,
            "core_event_summary": topic.core_event_summary,
            "source_url": topic.source_url,
            "language": language,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        records.append(new_entry)
        self._save_raw(records)
        logger.info(f"Konu hafızaya kaydedildi: {topic.topic_id}")

    def _save_raw(self, records: List[dict]):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


# ==========================================
# AJAN 1: HABER TARAMA VE SEÇİM MOTORU
# ==========================================
class NewsIntelligenceAgent:
    def __init__(self, language: str = TARGET_LANGUAGE):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY bulunamadı! Lütfen .env dosyasını kontrol edin.")
        
        self.language = language.upper()
        self.config = get_active_config(self.language)
        self.memory = TopicMemory()
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    @staticmethod
    def _clean_html(raw_html: str) -> str:
        """RSS özetlerindeki HTML etiketlerini temizler."""
        if not raw_html:
            return ""
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext.strip()

    def fetch_rss_candidates(self, max_per_feed: int = 12) -> List[dict]:
        """Tanımlı RSS kaynaklarından güncel haberleri çeker."""
        candidates = []
        seen_titles = set()
        past_records = self.memory.load_and_prune()
        past_urls = {r.get("source_url") for r in past_records}

        for feed_url in self.config["rss_feeds"]:
            logger.info(f"RSS Taranıyor: {feed_url}")
            try:
                feed = feedparser.parse(feed_url)
                count = 0
                for entry in feed.entries:
                    if count >= max_per_feed:
                        break
                    
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "").strip()
                    summary = self._clean_html(entry.get("summary", entry.get("description", "")))

                    if not title or title in seen_titles or link in past_urls:
                        continue

                    seen_titles.add(title)
                    candidates.append({
                        "title": title,
                        "url": link,
                        "snippet": summary[:350]
                    })
                    count += 1
            except Exception as e:
                logger.warning(f"RSS okuma hatası ({feed_url}): {e}")

        logger.info(f"Toplam {len(candidates)} benzersiz haber adayı toplandı.")
        return candidates

    def select_best_topic(self, auto_save_to_memory: bool = True) -> SelectedTopicSchema:
        """Aday haberleri Gemini 2.5 Flash ile analiz eder ve en iyi belgesel konusunu seçer."""
        candidates = self.fetch_rss_candidates()
        if not candidates:
            raise RuntimeError("RSS kaynaklarından hiçbir haber çekilemedi. İnternet bağlantısını kontrol edin.")

        past_records = self.memory.load_and_prune()
        past_topics_summary = [
            f"- [{r['topic_id']}]: {r['original_headline']}" for r in past_records
        ]
        past_context = "\n".join(past_topics_summary) if past_topics_summary else "Henüz işlenmiş konu yok."

        candidates_text = "\n".join(
            [f"{i+1}. BAŞLIK: {c['title']}\n   ÖZET: {c['snippet']}\n   URL: {c['url']}" for i, c in enumerate(candidates)]
        )

        system_instruction = f"""
        Sen Kıdemli bir Belgesel Yapımcısı ve Stratejik Haber Analistisin.
        Hedef Pazar ve Çıktı Dili: {self.config['name']} ({self.language}).

        GÖREVİN:
        Verilen güncel haber havuzundan, 3 dakikalık derinlikli bir "Ekonomi, Bilim veya Teknoloji Analiz Belgeseli" olmaya en uygun TEK BİR konuyu seçmek.

        KESİN KURALLAR:
        1. TEKRAR YASAĞI: 'SON 30 GÜNDE İŞLENEN KONULAR' listesinde yer alan konularla aynı olayı anlatan haberleri (başlığı farklı olsa bile) KESİNLİKLE SEÇME.
        2. KALİTE FİLTRESİ: Dedikodu, spor sonuçları, yerel adli vakalar, indirim haberleri veya yüzeysel ürün tanıtımlarını ele. Yapısal kırılmalar, yapay zeka regülasyonları, makroekonomik krizler, enerji teknolojileri veya bilimsel sıçramaları seç.
        3. LEISTUNGSSCHUTZRECHT (TELİF) UYUMU: 'core_event_summary' ve 'editorial_angle' alanlarını kendi analitik cümlelerinle, {self.language} dilinde yaz. Kaynak haberin cümle yapısını kopyalama.
        """

        prompt = f"""
        === SON 30 GÜNDE İŞLENEN KONULAR (BUNLARI SEÇME) ===
        {past_context}

        === GÜNCEL HABER ADAYLARI ===
        {candidates_text}

        Yukarıdaki adayları değerlendir ve şemaya uygun olarak en yüksek analitik potansiyele sahip konuyu seç.
        """

        logger.info("Gemini 2.5 Flash ile en iyi konu seçiliyor...")
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=SelectedTopicSchema,
                temperature=0.3,
            ),
        )

        selected_topic = SelectedTopicSchema.model_validate_json(response.text)
        logger.info(f"SEÇİLEN KONU: {selected_topic.topic_id} (Viral Skor: {selected_topic.viral_score})")

        if auto_save_to_memory:
            self.memory.save_topic(selected_topic, self.language)

        return selected_topic


if __name__ == "__main__":
    print(f"--- Ajan 1 Testi Başlatılıyor (Dil: {TARGET_LANGUAGE}) ---")
    agent = NewsIntelligenceAgent(language=TARGET_LANGUAGE)
    result = agent.select_best_topic(auto_save_to_memory=True)
    print("\n[BAŞARILI] Ajan 1 Çıktısı:")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
