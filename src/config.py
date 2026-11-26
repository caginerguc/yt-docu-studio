"""MakroKontext V3 — merkezi konfigürasyon (Evergreen Gizem & Arşiv Serisi).

Kurallar:
  * Sırlar (API anahtarları) koda yazılmaz, yalnızca ortam değişkeninden okunur.
  * Varsayılanlar ücretsiz katmanlara göre seçilmiştir.
  * Her şey .env (yerel) veya GitHub Secrets/Variables (bulut) ile override edilebilir.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

try:  # yerel geliştirmede .env oku; CI'da gerekmez
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Ortam değişkeni yardımcıları
# --------------------------------------------------------------------------- #
def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = _env(name)
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else list(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Dizinler
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
MUSIC_DIR = ASSETS_DIR / "music"
FONTS_DIR = ASSETS_DIR / "fonts"
DATA_DIR = ROOT / "data"
HISTORY_FILE = DATA_DIR / "history.json"

OUTPUT_DIR = ROOT / "output"
SCENES_DIR = OUTPUT_DIR / "scenes"
AUDIO_DIR = OUTPUT_DIR / "audio"
CLIPS_DIR = OUTPUT_DIR / "clips"
CACHE_DIR = OUTPUT_DIR / "cache"          # checkpoint/resume için önbellek
THUMB_DIR = OUTPUT_DIR / "thumbnails"
LOG_DIR = OUTPUT_DIR / "logs"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

ALL_OUTPUT_DIRS = (SCENES_DIR, AUDIO_DIR, CLIPS_DIR, CACHE_DIR, THUMB_DIR, LOG_DIR)


def ensure_dirs() -> None:
    for d in (DATA_DIR, MUSIC_DIR, FONTS_DIR, *ALL_OUTPUT_DIRS):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Sırlar (hepsi ortam değişkeni)
# --------------------------------------------------------------------------- #
GEMINI_API_KEY = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")

PEXELS_API_KEY = _env("PEXELS_API_KEY")                 # ücretsiz stok video/foto
CF_ACCOUNT_ID = _env("CF_ACCOUNT_ID")                   # Cloudflare Workers AI (FLUX)
CF_API_TOKEN = _env("CF_API_TOKEN")
POLLINATIONS_API_KEY = _env("POLLINATIONS_API_KEY")     # opsiyonel (sk_...)
WIKI_CONTACT = _env("WIKI_CONTACT")                     # Wikimedia User-Agent için e-posta/URL

YT_CLIENT_ID = _env("YT_CLIENT_ID")
YT_CLIENT_SECRET = _env("YT_CLIENT_SECRET")
YT_REFRESH_TOKEN = _env("YT_REFRESH_TOKEN")

TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")         # opsiyonel hata bildirimi
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID")


# --------------------------------------------------------------------------- #
# LLM (metin) — model adları .env'den değiştirilebilir
# --------------------------------------------------------------------------- #
TEXT_MODEL = _env("TEXT_MODEL", "gemini-3.5-flash")
TEXT_FALLBACK_MODELS = _env_list(
    "TEXT_FALLBACK_MODELS", ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 5)
LLM_BACKOFF_BASE_S = _env_int("LLM_BACKOFF_BASE_S", 6)   # 6, 12, 24, 48 ...
LLM_BACKOFF_MAX_S = 60
# 503 "high demand": aynı modelde bu kadar denemeden sonra sıradaki modele geç (5 deneme ~3 dk bekletiyordu)
LLM_OVERLOAD_RETRIES = _env_int("LLM_OVERLOAD_RETRIES", 3)
# Gemini tamamen yanıt vermezse bu süre boyunca çağrılar doğrudan yedek sağlayıcıya (NVIDIA NIM) gider
LLM_PRIMARY_COOLDOWN_S = _env_int("LLM_PRIMARY_COOLDOWN_S", 600)


# --------------------------------------------------------------------------- #
# İçerik mimarisi: koleksiyonlar, hedef süre, perde planı
# --------------------------------------------------------------------------- #
COLLECTIONS = (
    {"id": "schatten_genies", "name": "Schattengenies",
     "brief": "Bilim insanlarının az bilinen, belgelenmiş yönleri ve el konulan/unutulan çalışmaları."},
    {"id": "dunkle_korridore", "name": "Dunkle Korridore der Macht",
     "brief": "Siyaset ve istihbaratın belgelenmiş gizli operasyonları (Gladio, Crypto AG gibi)."},
    {"id": "geheimnis_der_steine", "name": "Das Geheimnis der Steine",
     "brief": "Tarihi eserler ve yapılar: çözülmüş ve hâlâ tartışmalı sırlar."},
    {"id": "intrigen_und_zusammenbrueche", "name": "Intrigen & Zusammenbrüche",
     "brief": "Belgelenmiş finansal çöküşler ve entrikalar (1929, Doğu Hindistan Şirketi vb.)."},
)

# Konu havuzundan otomatik elenir: komplo efsaneleri / nefret söylemiyle bağlantılı anlatılar.
TOPIC_BLOCKLIST = (
    "rothschild", "weisen von zion", "protocols of the elders", "illuminati", "new world order",
    "neue weltordnung", "great reset", "chemtrail", "reptilian", "reptiloid", "flat earth",
    "flache erde", "qanon", "holocaust-leugn", "holocaust denial", "great replacement",
)

TARGET_MINUTES = (8, 11)                                 # hedef video süresi
WORDS_PER_MINUTE = 135                                   # Almanca belgesel anlatımı + duraklamalar
TARGET_WORDS = (TARGET_MINUTES[0] * WORDS_PER_MINUTE, TARGET_MINUTES[1] * WORDS_PER_MINUTE)

# 5 perdelik yapı: hedef kelime payı yaklaşık oransaldır.
ACTS = (
    {"id": 1, "name": "Der Haken",          "goal": "En esrarengiz anın ortasından başla; ilk 30 sn'de soru kur.", "share": 0.12},
    {"id": 2, "name": "Die bekannte Version", "goal": "Resmi/yaygın anlatı; izleyicinin bildiğini sandığı hikâye.", "share": 0.20},
    {"id": 3, "name": "Der Bruch",          "goal": "Belgelenmiş çelişki ve kırılma; kaynaklı detay.",           "share": 0.28},
    {"id": 4, "name": "Das tiefere Geheimnis", "goal": "Arkadaki motivasyon ve aktörler; belgelenmiş vs. iddia ayrımı.", "share": 0.25},
    {"id": 5, "name": "Das ungelöste Erbe", "goal": "Çözülmemiş sorular, bugüne yansıma, düşündüren final.",     "share": 0.15},
)

# Hikâye anlatımı kuralı: anlatıcı belgelenmiş olanı, iddia edileni ve spekülasyonu açıkça ayırır.
EVIDENCE_LABELS = ("belegt", "behauptet", "spekulativ")
MIN_SOURCES_PER_VIDEO = 3                                # yazıdan önce toplanması gereken kaynak sayısı


# --------------------------------------------------------------------------- #
# Kesit (shot) planı — sese göre dinamik hesaplanır
# --------------------------------------------------------------------------- #
SHOT_TARGET_S = 5.0                                      # ortalama kesit süresi
SHOT_MIN_S, SHOT_MAX_S = 3.5, 7.5
# Hedef görsel kaynak karışımı (toplam kesit içindeki pay)
SOURCE_MIX = {"stock_video": 0.45, "archive_photo": 0.35, "ai_image": 0.20}   # gerçek görüntü öncelikli
CROSSFADE_S = 0.5                                        # yalnızca "yumuşak" geçişlerde; varsayılan sert kesme
SENTENCE_GAP_S = 0.35                                    # cümleler arası sessizlik
TWIST_GAP_S = 0.90                                       # kırılma cümlelerinden önceki sessizlik


# --------------------------------------------------------------------------- #
# Görsel üretim zinciri (soldan sağa denenir)
# --------------------------------------------------------------------------- #
IMAGE_PROVIDERS = _env_list("IMAGE_PROVIDERS", ["cloudflare", "pollinations", "gemini"])
STOCK_ENABLED = _env_bool("STOCK_ENABLED", True)

CF_FLUX_MODEL = _env("CF_FLUX_MODEL", "@cf/black-forest-labs/flux-1-schnell")
CF_FLUX_STEPS = _env_int("CF_FLUX_STEPS", 6)             # schnell için 1-8
AI_IMAGE_W, AI_IMAGE_H = 1344, 768                       # üretim boyutu; render'da 1080p'ye ölçeklenir

# Pollinations: önce yeni API (gen.pollinations.ai, sk_ anahtar ister), sorun olursa eski "Classic" API.
POLLINATIONS_BASE_URL = _env("POLLINATIONS_BASE_URL", "https://gen.pollinations.ai/image")
POLLINATIONS_MODELS = _env_list("POLLINATIONS_MODELS", ["flux"])
POLLINATIONS_CLASSIC_URL = _env("POLLINATIONS_CLASSIC_URL", "https://image.pollinations.ai/prompt")
POLLINATIONS_CLASSIC_MODELS = _env_list("POLLINATIONS_CLASSIC_MODELS", ["flux", "turbo"])


def pollinations_routes() -> list[tuple[str, str, bool]]:
    """(adres, model, klasik_mi) sırası. Yeni API anahtarsız çalışmadığı için anahtar yoksa yalnızca Classic API."""
    new = [(POLLINATIONS_BASE_URL, m, False) for m in POLLINATIONS_MODELS] if POLLINATIONS_API_KEY else []
    return new + [(POLLINATIONS_CLASSIC_URL, m, True) for m in POLLINATIONS_CLASSIC_MODELS]

GEMINI_IMAGE_MODEL = _env("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")

MIN_IMAGE_BYTES = 5000                                   # altı = bozuk görsel
# Tek bir sanatsal dil: tüm AI görseller aynı "art direction" ekini alır.
ART_DIRECTION = (
    "cinematic 35mm film still, dark moody chiaroscuro lighting, vintage archival atmosphere, "
    "volumetric fog, dramatic shadows, shallow depth of field, no text, no watermark, no logos"
)

# Wikimedia Commons: yalnızca telif atfı gerektirmeyen lisanslar otomatik kullanılır.
WIKIMEDIA_ALLOWED_LICENSES = ("public domain", "pd", "cc0")
WIKI_PAGEVIEW_PROJECTS = ("de.wikipedia", "en.wikipedia")


# --------------------------------------------------------------------------- #
# Video / Shorts
# --------------------------------------------------------------------------- #
VIDEO_W, VIDEO_H, FPS = 1920, 1080, 30
KB_SUPERSAMPLE_W, KB_SUPERSAMPLE_H = 3840, 2160          # titremesiz zoompan için
SHORTS_W, SHORTS_H = 1080, 1920
SHORTS_MAX_SECONDS = 58

MUSIC_VOLUME = 0.125                                     # ~ -18 dB
DUCK_VOLUME = 0.06                                       # konuşma sırasında müzik
TARGET_LUFS = -14
MUSIC_FILES = {
    "serious_news": "serious_news.mp3",
    "ambient_tech": "ambient_tech.mp3",
    "cinematic_tension": "cinematic_tension.mp3",
}
# Güvenli sentez filtresi (karmaşık chorus filtresi exit 234 üretiyordu)
MUSIC_SYNTH_FILTER = "lowpass=f=800,tremolo=f=0.25:d=0.5,volume=0.20"

CAMERA_EFFECTS = ("zoom_in", "zoom_out", "pan_left_to_right", "pan_right_to_left")


# --------------------------------------------------------------------------- #
# Ses (edge-tts)
# --------------------------------------------------------------------------- #
TTS_RETRIES = _env_int("TTS_RETRIES", 4)
TTS_SCENE_PAUSE_S = 1.5
# NoAudioReceived'ı önlemek için SSML rate/pitch KULLANMA — doğal hız.
# Duraklamalar metinde değil, cümle başına TTS + FFmpeg sessizlik ile yapılır.


# --------------------------------------------------------------------------- #
# Diller / trend sinyalleri
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LangConfig:
    code: str
    name: str
    hl: str
    gl: str
    ceid: str
    voices: tuple[str, ...]
    yt_language: str
    news_queries: tuple[str, ...]
    trends_rss: str = ""


LANGS: dict[str, LangConfig] = {
    "de": LangConfig(
        code="de", name="Deutsch", hl="de", gl="DE", ceid="DE:de",
        voices=("de-DE-KillianNeural", "de-DE-ConradNeural", "de-DE-FlorianMultilingualNeural"),
        yt_language="de",
        news_queries=(
            "Akten freigegeben when:7d",
            "Archiv entdeckt Dokumente when:7d",
            "Nachlass Forscher entschlüsseln when:7d",
            "Geheimdienst Dokumente veröffentlicht when:7d",
            "Archäologen Fund Entdeckung when:7d",
        ),
        trends_rss="https://trends.google.com/trending/rss?geo=DE",
    ),
    "en": LangConfig(
        code="en", name="English", hl="en", gl="US", ceid="US:en",
        voices=("en-US-AndrewNeural", "en-GB-RyanNeural"),
        yt_language="en",
        news_queries=(
            "declassified files when:7d",
            "archive discovered documents when:7d",
            "researchers decipher manuscript when:7d",
            "archaeologists discovery when:7d",
        ),
        trends_rss="https://trends.google.com/trending/rss?geo=US",
    ),
    "tr": LangConfig(
        code="tr", name="Türkçe", hl="tr", gl="TR", ceid="TR:tr",
        voices=("tr-TR-AhmetNeural",),
        yt_language="tr",
        news_queries=("gizli belgeler açıklandı when:7d", "arkeologlar keşif when:7d"),
        trends_rss="https://trends.google.com/trending/rss?geo=TR",
    ),
}
DEFAULT_LANG = "de"


def get_lang(code: str | None) -> LangConfig:
    return LANGS.get((code or DEFAULT_LANG).lower(), LANGS[DEFAULT_LANG])


def google_news_url(query: str, lang: LangConfig) -> str:
    return (
        f"https://news.google.com/rss/search?q={quote_plus(query)}"
        f"&hl={lang.hl}&gl={lang.gl}&ceid={lang.ceid}"
    )


# Konu hafızası (tekrar önleyici) — evergreen içerikte pencere uzun tutulur
HISTORY_DAYS = 365
TOPIC_SIMILARITY_THRESHOLD = 0.55                        # token Jaccard eşiği


# --------------------------------------------------------------------------- #
# Thumbnail (Pillow ile kompozit — metni Flux'a bırakma, kendimiz çiziyoruz)
# --------------------------------------------------------------------------- #
THUMB_W, THUMB_H = 1280, 720
THUMB_MAX_BYTES = 2_000_000                              # YouTube sınırı 2 MB
THUMB_VARIANTS = _env_int("THUMB_VARIANTS", 3)
THUMB_FONT_CANDIDATES = (
    FONTS_DIR / "Anton-Regular.ttf",
    FONTS_DIR / "BebasNeue-Regular.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("C:/Windows/Fonts/impact.ttf"),
)
CHANNEL_WATERMARK = "MAKROKONTEXT"


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #
UPLOAD_ENABLED = _env_bool("UPLOAD_ENABLED", True)
# DİKKAT: API denetiminden (audit) geçmemiş projelerde YouTube her videoyu private'a kilitler.
UPLOAD_PRIVACY = _env("UPLOAD_PRIVACY", "public").lower()   # varsayılan: doğrudan yayında
# Kalite kapısı (varsayılan KAPALI: her video doğrudan public). QUALITY_GATE=true yapılırsa public istenip
# video bu eşikleri geçemediğinde private yüklenir (elle inceleme için).
QUALITY_GATE = _env_bool("QUALITY_GATE", False)
PUBLIC_MIN_SCORE = int(_env("PUBLIC_MIN_SCORE", "5"))            # eleştirmen genel puanı (0-10)
PUBLIC_MAX_WEAK_VISUALS = float(_env("PUBLIC_MAX_WEAK_VISUALS", "0.25"))  # tekrar/boş kesit payı üst sınırı
YT_CATEGORY_ID = _env("YT_CATEGORY_ID", "27")            # 27 = Education
YT_MADE_FOR_KIDS = False
YT_CONTAINS_SYNTHETIC_MEDIA = True                       # sentetik medya beyanı (zorunlu)
PUBLISH_TZ = _env("PUBLISH_TZ", "Europe/Berlin")
PUBLISH_LOCAL_TIME = _env("PUBLISH_LOCAL_TIME", "18:00")  # yerel yayın saati (HH:MM)
VIDEOS_PER_WEEK = 2                                      # bilgi amaçlı; cron workflow'da


def now_local() -> datetime:
    return datetime.now(ZoneInfo(PUBLISH_TZ))


def today_local_str() -> str:
    return now_local().strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Doğrulama & durum
# --------------------------------------------------------------------------- #
def provider_status() -> dict[str, bool]:
    return {
        "gemini_text": bool(GEMINI_API_KEY),
        "pexels_stock": bool(PEXELS_API_KEY) and STOCK_ENABLED,
        "cloudflare_flux": bool(CF_ACCOUNT_ID and CF_API_TOKEN),
        "pollinations": True,                            # anahtarsız da denenir
        "gemini_image": bool(GEMINI_API_KEY),
        "wikimedia": bool(WIKI_CONTACT),
        "youtube_upload": bool(YT_CLIENT_ID and YT_CLIENT_SECRET and YT_REFRESH_TOKEN),
        "telegram_notify": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    }


def validate(upload: bool = True) -> tuple[list[str], list[str]]:
    """(hatalar, uyarılar) döndürür. Hata varsa pipeline başlamamalı."""
    errors: list[str] = []
    warnings: list[str] = []
    st = provider_status()

    if not st["gemini_text"]:
        errors.append("GEMINI_API_KEY eksik (zorunlu).")
    if upload and UPLOAD_ENABLED and not st["youtube_upload"]:
        errors.append("YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN eksik (yükleme açık).")
    if UPLOAD_PRIVACY not in {"private", "unlisted", "public"}:
        errors.append(f"UPLOAD_PRIVACY geçersiz: {UPLOAD_PRIVACY!r}")

    if not st["pexels_stock"]:
        warnings.append("PEXELS_API_KEY yok: gerçek stok b-roll kapalı, yalnızca AI görseller kullanılacak.")
    if not st["cloudflare_flux"]:
        warnings.append("Cloudflare Workers AI anahtarı yok: FLUX yedeği kapalı.")
    if not st["wikimedia"]:
        warnings.append("WIKI_CONTACT yok: Wikimedia arşiv görselleri ve Wikipedia araştırması için User-Agent'ta iletişim bilgisi gerekir.")
    if UPLOAD_PRIVACY == "public":
        warnings.append(
            "UPLOAD_PRIVACY=public: YouTube API denetiminden geçmemiş projede video private'a kilitlenir."
        )
    return errors, warnings


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "run.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("makrokontext")

MAX_SENTENCE_WORDS = 25
MAX_CANDIDATES_TRIED = 4
MIN_FACTS_PER_VIDEO = 12
LLM_MIN_INTERVAL_S = 4.0
PAGEVIEW_TOP_N = 100
MAX_SOURCE_CHARS = 15000
MAX_REVISIONS = 2

WORD_COUNT_TOLERANCE = 0.4

WORD_COUNT_TOLERANCE = 0.5

MIN_WORD_RATIO = 0.5
