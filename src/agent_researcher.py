import json
import logging
from datetime import datetime, timedelta
from typing import List
import feedparser
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    TARGET_LANGUAGE,
    HISTORY_FILE,
    HISTORY_RETENTION_DAYS,
    get_active_config,
    generate_content_with_fallback
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Agent1_MysteryResearcher")


class SelectedTopicSchema(BaseModel):
    topic_id: str = Field(description="Konuyu özetleyen benzersiz, küçük harfli ve tireli slug (örn: oppenheimer-hidden-soviet-file).")
    original_headline: str = Field(description="İlham alınan güncel trend veya haber başlığı.")
    source_url: str = Field(description="Haberin veya referansın kaynak linki.")
    historical_or_mystery_hook: str = Field(description="Güncel olayın bağlandığı derin tarihsel sır, gizli istihbarat dosyası, unutulmuş bilim insanı veya tarihi eser/finansal entrika.")
    core_event_summary: str = Field(description="Konunun tarihsel arka planı, gizli kalmış detayları, gerçek isimler, tarihler ve arşiv gerçekleri.")
    editorial_angle: str = Field(description="Lemmino, Fern ve Arte tarzı karanlık, merak uyandırıcı, sinematik araştırma-belgesel anlatım açısı.")
    viral_score: int = Field(description="1-100 arası Evergreen (yıllarca izlenme) potansiyeli skoru.", ge=1, le=100)


class NewsIntelligenceAgent:
    def __init__(self, language: str = TARGET_LANGUAGE):
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY bulunamadı!")
        self.language = language.upper()
        self.config = get_active_config(self.language)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def load_recent_history(self) -> List[dict]:
        if not HISTORY_FILE.exists():
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            cutoff = datetime.now() - timedelta(days=HISTORY_RETENTION_DAYS)
            recent = []
            for item in data:
                try:
                    dt = datetime.fromisoformat(item.get("date", "2000-01-01"))
                    if dt >= cutoff:
                        recent.append(item)
                except ValueError:
                    recent.append(item)
            return recent
        except Exception as e:
            logger.warning(f"History okunamadı: {e}")
            return []

    def save_to_history(self, topic: SelectedTopicSchema):
        history = self.load_recent_history()
        history.append({
            "topic_id": topic.topic_id,
            "headline": topic.historical_or_mystery_hook,
            "date": datetime.now().isoformat(),
            "language": self.language
        })
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"Konu hafızaya kaydedildi: {topic.topic_id}")

    def fetch_rss_headlines(self, max_per_feed: int = 12) -> List[dict]:
        articles = []
        for url in self.config["rss_feeds"]:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_per_feed]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "summary": entry.get("summary", "")[:300]
                    })
            except Exception as e:
                logger.warning(f"RSS hatası ({url}): {e}")
        logger.info(f"Toplam {len(articles)} güncel trend sinyali toplandı.")
        return articles

    def select_best_topic(self, auto_save_to_memory: bool = True, commit_to_history: bool = True) -> SelectedTopicSchema:
        articles = self.fetch_rss_headlines()
        if not articles:
            raise RuntimeError("Hiçbir RSS kaynağından veri çekilemedi!")

        recent_history = self.load_recent_history()
        used_topics_str = "\n".join([f"- {h['topic_id']}: {h['headline']}" for h in recent_history]) or "Henüz işlenmiş konu yok."
        articles_str = "\n".join([f"[{i+1}] {a['title']} | {a['summary']} (URL: {a['link']})" for i, a in enumerate(articles)])

        system_instruction = """
        Sen MakroKontext Sinematik Belgesel Stüdyosu'nun Baş Araştırmacısısın (Ajan 1).
        Amacımız: Sıradan günlük haber videoları yapmak DEĞİL; yıllarca izlenecek (Evergreen), yüksek merak uyandıran, Lemmino / Arte / ColdFusion tarzı derinlemesine gizem ve tarih belgeselleri üretmektir.
        
        GÖREVİN:
        1. Güncel haber havuzundaki trendlere bak (örneğin çip krizi, enerji savaşları, yapay zeka, merkez bankaları, savunma sanayii, uzay yarışı).
        2. Bu güncel trendi bir SIÇRAMA TAHTASI olarak kullan ve arkasındaki ESRARENGİZ TARİHSEL KÖKENİ, GÖLGEDEKİ BİLİM İNSANINI, SİYASİ/İSTİHBARİ GİZLİ DOSYAYI veya TARİHİ ESER/FİNANSAL ENTRİKAYI bul.
           - Örnek: Gündemde enerji/elektrik krizi varsa -> Nikola Tesla'nın FBI tarafından el konulan kayıp Wardenclyffe dosyalarına veya 1973 gizli petrol anlaşmasına bağla.
           - Örnek: Gündemde yapay zeka/şifreleme varsa -> Alan Turing'in gizlenen Bletchley Park belgelerine veya Soğuk Savaş'taki Operation Rubicon (Crypto AG) casusluk operasyonuna bağla.
        3. KESİN KURAL: 'DAHA ÖNCE İŞLENEN KONULAR' listesindeki konularla aynı veya benzer bir konuyu ASLA seçme! Gerçek tarihsel olgulara, gerçek tarihlere ve dosyalara dayan.
        """

        prompt = f"""
        === DAHA ÖNCE İŞLENEN KONULAR (BUNLARI SEÇME) ===
        {used_topics_str}

        === İLHAM ALINACAK GÜNCEL TREND HAVUZU ===
        {articles_str}
        """

        response = generate_content_with_fallback(
            client=self.client,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=SelectedTopicSchema,
                temperature=0.5,
            )
        )

        selected = SelectedTopicSchema.model_validate_json(response.text)
        logger.info(f"Seçilen Zamansız Belgesel Konusu: {selected.topic_id} | {selected.historical_or_mystery_hook}")

        if auto_save_to_memory and commit_to_history:
            self.save_to_history(selected)

        return selected
