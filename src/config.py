import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"

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
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=tr&gl=TR&ceid=TR:tr",
            "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=tr&gl=TR&ceid=TR:tr",
            HN_FEED
        ],
    }
}

def get_active_config(lang: str = TARGET_LANGUAGE):
    return LANGUAGE_CONFIG.get(lang.upper(), LANGUAGE_CONFIG["DE"])

HISTORY_RETENTION_DAYS = 30

