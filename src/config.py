import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Dizin Yolları
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"
HISTORY_FILE = DATA_DIR / "history.json"

# Klasörlerin varlığını garanti altına al
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "scenes").mkdir(exist_ok=True)
(OUTPUT_DIR / "audio").mkdir(exist_ok=True)

# API & Çalışma Parametreleri
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "DE").upper()  # "EN", "DE", "TR"
GEMINI_MODEL = "gemini-3.8-flash"
HISTORY_RETENTION_DAYS = 30

# Çoklu Dil, Ses (Edge-TTS) ve Haber Kaynağı (RSS) Matrisi
LANGUAGE_CONFIG = {
    "DE": {
        "name": "German (DACH Region)",
        "voice": "de-DE-KillianNeural",
        "rss_feeds": [
            "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=de&gl=DE&ceid=DE:de",
            "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=de&gl=DE&ceid=DE:de",
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=de&gl=DE&ceid=DE:de",
            "https://hnrss.org/frontpage?points=150"
        ]
    },
    "EN": {
        "name": "English (Global / US)",
        "voice": "en-US-ChristopherNeural",
        "rss_feeds": [
            "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en",
            "https://hnrss.org/frontpage?points=150"
        ]
    },
    "TR": {
        "name": "Turkish",
        "voice": "tr-TR-AhmetNeural",
        "rss_feeds": [
            "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=tr&gl=TR&ceid=TR:tr",
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=tr&gl=TR&ceid=TR:tr",
            "https://hnrss.org/frontpage?points=150"
        ]
    }
}

def get_active_config(lang: str = None) -> dict:
    """Aktif dilin konfigürasyonunu döndürür."""
    target = (lang or TARGET_LANGUAGE).upper()
    if target not in LANGUAGE_CONFIG:
        raise ValueError(f"Desteklenmeyen dil kodu: {target}. Geçerli olanlar: EN, DE, TR")
    return LANGUAGE_CONFIG[target]
