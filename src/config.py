import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
logger = logging.getLogger("Config_LLM_Router")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
MUSIC_DIR = ASSETS_DIR / "music"
FONTS_DIR = ASSETS_DIR / "fonts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MUSIC_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = DATA_DIR / "history.json"
HISTORY_RETENTION_DAYS = 30

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro"
]

TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "DE").upper()
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()

HN_FEED = "https://news.ycombinator.com/rss"

LANGUAGE_CONFIG = {
    "DE": {
        "name": "Deutsch",
        "voice": "de-DE-KillianNeural",
        "fallback_voices": ["de-DE-ConradNeural", "de-DE-FlorianMultilingualNeural"],
        "rss_feeds": [
            "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
            "https://www.heise.de/rss/heise-atom.xml",
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=de&gl=DE&ceid=DE:de",
            "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=de&gl=DE&ceid=DE:de",
            HN_FEED
        ],
    },
    "EN": {
        "name": "English",
        "voice": "en-US-ChristopherNeural",
        "fallback_voices": ["en-US-GuyNeural", "en-US-EricNeural"],
        "rss_feeds": [
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
            HN_FEED
        ],
    },
    "TR": {
        "name": "Türkçe",
        "voice": "tr-TR-AhmetNeural",
        "fallback_voices": ["tr-TR-EmelNeural"],
        "rss_feeds": [
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=tr&gl=tr&ceid=TR:tr",
            "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=tr&gl=tr&ceid=TR:tr",
            HN_FEED
        ],
    }
}

def get_active_config(lang: str = TARGET_LANGUAGE):
    return LANGUAGE_CONFIG.get(lang.upper(), LANGUAGE_CONFIG["DE"])

def generate_content_with_fallback(client: genai.Client, contents: str, config):
    """503/429 hatalarında sırayla yedek Gemini modellerine geçiş yapar."""
    last_err = None
    for round_idx in range(1, 4):
        for model_name in dict.fromkeys(GEMINI_FALLBACK_MODELS):
            try:
                logger.info(f"LLM isteği gönderiliyor (Model: {model_name}, Tur: {round_idx})...")
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except Exception as e:
                last_err = e
                logger.warning(f"Model {model_name} yanıt vermedi ({e}). 8 sn sonra yedek modele geçiliyor...")
                time.sleep(8)
    raise RuntimeError(f"Tüm Gemini modelleri yoğunluk nedeniyle başarısız oldu: {last_err}")
