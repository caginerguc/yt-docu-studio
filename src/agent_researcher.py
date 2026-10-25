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
logger = logging.getLogger("Agent1_Researcher")


class SelectedTopicSchema(BaseModel):
    topic_id: str = Field(description="Konuyu özetleyen benzersiz, küçük harfli ve tireli İngilizce/Almanca slug (örn: vw-factory-closures-de).")
    original_headline: str = Field(description="Seçilen haberin orijinal başlığı.")
    source_url: str = Field(description="Haberin kaynak linki.")
    core_event_summary: str = Field(description="Haberin tamamen olgusal, objektif ve detaylı özeti.")
    editorial_angle: str = Field(description="DW/Arte tarzı belgesel için derinlemesine ekonomik/teknolojik analiz açısı.")
    viral_score: int = Field(description="1-100 arası izlenme potansiyeli skoru.", ge=1, le=100)


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
            "headline": topic.original_headline,
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
        logger.info(f"Toplam {len(articles)} güncel haber başlığı toplandı.")
        return articles

    def select_best_topic(self, auto_save_to_memory: bool = True) -> SelectedTopicSchema:
        articles = self.fetch_rss_headlines()
        if not articles:
            raise RuntimeError("Hiçbir RSS kaynağından haber çekilemedi!")

        recent_history = self.load_recent_history()
        used_topics_str = "\n".join([f"- {h['topic_id']}: {h['headline']}" for h in recent_history]) or "Henüz işlenmiş konu yok."
        articles_str = "\n".join([f"[{i+1}] {a['title']} | {a['summary']} (URL: {a['link']})" for i, a in enumerate(articles)])

        system_instruction = f"""
        Sen MakroKontext belgesel kanalının Baş Araştırmacısısın (Ajan 1).
        Görevin: Verilen güncel haber listesinden DACH bölgesi izleyicisi için en yüksek analitik derinliğe sahip ekonomi, sanayi veya teknoloji konusunu seçmek.
        KESİN KURAL: Aşağıdaki 'SON 30 GÜNDE İŞLENEN KONULAR' listesindeki konularla aynı veya çok benzer bir haberi ASLA seçme!
        """

        prompt = f"""
        === SON 30 GÜNDE İŞLENEN KONULAR (BUNLARI SEÇME) ===
        {used_topics_str}

        === GÜNCEL HABER HAVUZU ===
        {articles_str}
        """

        response = generate_content_with_fallback(
            client=self.client,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=SelectedTopicSchema,
                temperature=0.3,
            )
        )

        selected = SelectedTopicSchema.model_validate_json(response.text)
        logger.info(f"Seçilen Konu: {selected.topic_id} (Skor: {selected.viral_score})")

        if auto_save_to_memory:
            self.save_to_history(selected)

        return selected
