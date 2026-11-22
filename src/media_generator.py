"""Aşama 3 — Medya tedariki: anlatıcı sesi + gerçek görüntü.

Her cümle için:
  * Ses   : edge-tts ile cümle başına MP3  ->  output/audio/<sentence_id>.mp3
Her kesit (shot) için, plan.source'a göre öncelik sırası:
  * archive_photo : Wikimedia Commons (yalnızca Public Domain / CC0)  -> Pexels foto -> AI
  * stock_video   : Pexels VIDEO (gerçek hareketli görüntü)          -> Pexels foto -> AI
  * ai_image      : Pollinations (tek sanat yönetimi)                -> Pexels foto -> boş tuval
Dosya adları: output/scenes/<sentence_id>_shot<N>.mp4  (video)  veya  .jpg  (durağan)
Aynı görsel/video bir videoda iki kez kullanılmaz. Var olan dosyalar yeniden indirilmez (resume).
Kullanılan dış kaynaklar output/credits.json dosyasına yazılır (açıklama metni için).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from src import config
from src.plan import PlanShot, ProductionPlan

log = logging.getLogger("makrokontext.media")

HTTP_TIMEOUT = 20
AI_TIMEOUT = 90                 # Pollinations üretimi yavaş olabilir
MAX_IMAGE_EDGE = 3840           # renderer 4K supersample kullanır; daha büyüğü gereksiz
MIN_IMAGE_WIDTH = 900           # bundan küçük arşiv görseli 1080p'de bulanık görünür
STOCK_MIN_W, STOCK_MAX_W = 1280, 1920
STOCK_MIN_SECONDS = 4
TTS_CONCURRENCY = 4


@dataclass
class Credit:
    kind: str       # wikimedia | pexels_video | pexels_photo | pollinations
    title: str
    url: str
    license: str


class MediaGenerator:
    def __init__(self, plan: ProductionPlan):
        self.plan = plan
        self.lang = config.get_lang(plan.language)
        self.scenes_dir = config.SCENES_DIR
        self.audio_dir = config.AUDIO_DIR
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            f"MakroKontextBot/3.1 (https://github.com/caginerguc/yt-docu-studio; "
            f"{config.WIKI_CONTACT or 'contact-not-set'})"
        )
        self._used: set[str] = set()
        self._lock = threading.Lock()
        self.credits: list[Credit] = []
        self.stats: dict[str, int] = {}

    # ================================================================== #
    # Yardımcılar
    # ================================================================== #
    def _claim(self, key: str) -> bool:
        """Aynı kaynağın aynı videoda tekrar kullanılmasını engeller."""
        with self._lock:
            if key in self._used:
                return False
            self._used.add(key)
            return True

    def _count(self, name: str) -> None:
        with self._lock:
            self.stats[name] = self.stats.get(name, 0) + 1

    def _credit(self, c: Credit) -> None:
        with self._lock:
            self.credits.append(c)

    @staticmethod
    def _save_image(data: bytes, path: Path, min_width: int = 0) -> bool:
        """Görseli doğrular, RGB JPEG'e çevirir, gereksiz büyükse küçültür."""
        if len(data) < config.MIN_IMAGE_BYTES:
            return False
        try:
            with Image.open(BytesIO(data)) as im:
                im.load()
                if im.width < min_width:
                    return False
                if im.mode != "RGB":
                    im = im.convert("RGB")
                if max(im.size) > MAX_IMAGE_EDGE:
                    im.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS)
                path.parent.mkdir(parents=True, exist_ok=True)
                im.save(path, "JPEG", quality=92)
            return True
        except Exception as e:  # bozuk/HTML/desteklenmeyen dosya
            log.debug("Görsel reddedildi (%s): %s", path.name, e)
            return False

    def _get(self, url: str, **kw) -> requests.Response | None:
        try:
            r = self.session.get(url, timeout=kw.pop("timeout", HTTP_TIMEOUT), **kw)
            return r if r.status_code == 200 else None
        except requests.RequestException as e:
            log.debug("HTTP hatası %s: %s", url[:80], e)
            return None

    # ================================================================== #
    # Wikimedia Commons (yalnızca PD / CC0)
    # ================================================================== #
    @staticmethod
    def _license_ok(meta: dict) -> tuple[bool, str]:
        short = str(meta.get("LicenseShortName", {}).get("value", "")).lower()
        lic = str(meta.get("License", {}).get("value", "")).lower()
        ok = ("public domain" in short or "cc0" in short or "cc0" in lic or lic.startswith("pd")
              or any(a == short or a == lic for a in config.WIKIMEDIA_ALLOWED_LICENSES))
        return ok, short or lic

    def _wikimedia(self, query: str, path: Path) -> bool:
        if not query:
            return False
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 15,
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 2560,
        }
        r = self._get("https://commons.wikimedia.org/w/api.php", params=params)
        if not r:
            return False
        pages = sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
        for p in pages:
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png", "image/tiff", "image/webp"):
                continue
            if int(info.get("width", 0)) < MIN_IMAGE_WIDTH:
                continue
            ok, lic = self._license_ok(info.get("extmetadata", {}))
            if not ok:
                continue
            url = info.get("thumburl") or info.get("url")
            if not url or not self._claim(url):
                continue
            img = self._get(url, timeout=30)
            if img and self._save_image(img.content, path, MIN_IMAGE_WIDTH):
                self._credit(Credit("wikimedia", p.get("title", ""), info.get("descriptionurl", url), lic))
                self._count("archive_photo")
                return True
        return False

    # ================================================================== #
    # Pexels: video ve foto
    # ================================================================== #
    def _pexels_video(self, query: str, path: Path) -> bool:
        if not (config.PEXELS_API_KEY and query):
            return False
        r = self._get("https://api.pexels.com/videos/search",
                      params={"query": query, "per_page": 12, "orientation": "landscape", "size": "medium"},
                      headers={"Authorization": config.PEXELS_API_KEY})
        if not r:
            return False
        for v in r.json().get("videos", []):
            if int(v.get("duration", 0)) < STOCK_MIN_SECONDS:
                continue
            files = [f for f in v.get("video_files", [])
                     if f.get("file_type") == "video/mp4" and f.get("width")
                     and STOCK_MIN_W <= int(f["width"]) <= STOCK_MAX_W and int(f["width"]) > int(f.get("height") or 0)]
            if not files:
                continue
            best = max(files, key=lambda f: int(f["width"]))
            if not self._claim(f"pexels-video-{v.get('id')}"):
                continue
            data = self._get(best["link"], timeout=120)
            if data and len(data.content) > 100_000:
                path.write_bytes(data.content)
                self._credit(Credit("pexels_video", v.get("user", {}).get("name", "Pexels"),
                                    v.get("url", ""), "Pexels License"))
                self._count("stock_video")
                return True
        return False

    def _pexels_photo(self, query: str, path: Path) -> bool:
        if not (config.PEXELS_API_KEY and query):
            return False
        r = self._get("https://api.pexels.com/v1/search",
                      params={"query": query, "per_page": 15, "orientation": "landscape"},
                      headers={"Authorization": config.PEXELS_API_KEY})
        if not r:
            return False
        for ph in r.json().get("photos", []):
            if not self._claim(f"pexels-photo-{ph.get('id')}"):
                continue
            img = self._get(ph.get("src", {}).get("large2x") or ph.get("src", {}).get("original", ""), timeout=30)
            if img and self._save_image(img.content, path, MIN_IMAGE_WIDTH):
                self._credit(Credit("pexels_photo", ph.get("photographer", "Pexels"), ph.get("url", ""),
                                    "Pexels License"))
                self._count("stock_photo")
                return True
        return False

    # ================================================================== #
    # Yapay zekâ görseli (son çare) ve boş tuval
    # ================================================================== #
    def _pollinations(self, prompt: str, path: Path, seed: int) -> bool:
        if not prompt:
            return False
        full = quote(f"{prompt}, {config.ART_DIRECTION}")
        headers = {"Authorization": f"Bearer {config.POLLINATIONS_API_KEY}"} if config.POLLINATIONS_API_KEY else {}
        for model in config.POLLINATIONS_MODELS:
            url = (f"{config.POLLINATIONS_BASE_URL}/{full}?width={config.AI_IMAGE_W}&height={config.AI_IMAGE_H}"
                   f"&seed={seed}&model={model}&nologo=true&enhance=false")
            r = self._get(url, headers=headers, timeout=AI_TIMEOUT)
            if r and self._save_image(r.content, path):
                self._credit(Credit("pollinations", prompt[:80], "https://pollinations.ai", "AI generated"))
                self._count("ai_image")
                return True
        return False

    def _canvas(self, path: Path, index: int) -> None:
        c = (14 + index * 7 % 20, 16 + index * 5 % 20, 22 + index * 11 % 24)
        Image.new("RGB", (config.VIDEO_W, config.VIDEO_H), c).save(path, "JPEG", quality=90)
        self._count("fallback_canvas")
        log.warning("Görsel bulunamadı, boş tuval kullanıldı: %s", path.name)

    # ================================================================== #
    # Kesit başına tedarik
    # ================================================================== #
    @staticmethod
    def shot_paths(sentence_id: str, idx: int) -> tuple[Path, Path]:
        base = config.SCENES_DIR / f"{sentence_id}_shot{idx}"
        return base.with_suffix(".mp4"), base.with_suffix(".jpg")

    def fetch_shot(self, sentence_id: str, idx: int, shot: PlanShot, seed: int, n: int) -> str:
        vid, img = self.shot_paths(sentence_id, idx)
        if vid.exists() and vid.stat().st_size > 100_000:
            return "cached"
        if img.exists() and img.stat().st_size > config.MIN_IMAGE_BYTES:
            return "cached"
        stock_q = shot.stock_query_en or shot.archive_query_en
        if shot.source == "archive_photo":
            if self._wikimedia(shot.archive_query_en, img) or self._pexels_photo(stock_q, img):
                return "ok"
        elif shot.source == "stock_video":
            if (config.STOCK_ENABLED and self._pexels_video(shot.stock_query_en, vid)) or self._pexels_photo(stock_q, img):
                return "ok"
        if self._pollinations(shot.ai_prompt_en or stock_q, img, seed):
            return "ok"
        if self._pexels_photo(stock_q or "dark archive room", img):
            return "ok"
        self._canvas(img, n)
        return "canvas"

    def prepare_all_visuals(self) -> None:
        seed0 = int(hashlib.sha1(self.plan.topic_id.encode()).hexdigest()[:6], 16) % 100000
        tasks = []
        n = 0
        for s in self.plan.sentences():
            for i, sh in enumerate(s.shots, 1):
                n += 1
                tasks.append((s.id, i, sh, seed0 + n, n))
        log.info("Görseller tedarik ediliyor: %d kesit (Wikimedia PD -> Pexels video/foto -> AI)", len(tasks))
        t0 = time.monotonic()
        # Arşiv ve stok önce: AI yalnızca gerçek görüntü bulunamazsa devreye girer.
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(self.fetch_shot, *t) for t in tasks]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # tek kesit tüm üretimi durdurmasın
                    log.warning("Kesit tedarik hatası: %s", e)
        log.info("Görsel tedariki bitti (%.0f sn). Dağılım: %s", time.monotonic() - t0, self.stats)

        thumb_bg = self.scenes_dir / "thumb_bg.jpg"
        if not thumb_bg.exists():
            if not self._pollinations(self.plan.seo.thumbnail_prompt_en, thumb_bg, seed0 + 999):
                if not self._pexels_photo("dark mysterious archive", thumb_bg):
                    self._canvas(thumb_bg, 99)

    # ================================================================== #
    # Anlatıcı sesi (edge-tts)
    # ================================================================== #
    @staticmethod
    def audio_path(sentence_id: str) -> Path:
        return config.AUDIO_DIR / f"{sentence_id}.mp3"

    async def _tts_one(self, sem: asyncio.Semaphore, sid: str, text: str) -> None:
        import edge_tts  # tembel import: test ortamında şart değil

        out = self.audio_path(sid)
        if out.exists() and out.stat().st_size > 2000:
            return
        voices = list(self.lang.voices)
        last: Exception | None = None
        async with sem:
            for attempt in range(config.TTS_RETRIES):
                voice = voices[0] if attempt < config.TTS_RETRIES - 1 else voices[-1]
                tmp = out.with_suffix(".part.mp3")
                try:
                    await edge_tts.Communicate(text, voice).save(str(tmp))
                    if tmp.exists() and tmp.stat().st_size > 2000:
                        tmp.replace(out)
                        return
                    raise RuntimeError("boş ses dosyası")
                except Exception as e:
                    last = e
                    await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"TTS başarısız ({sid}): {last}")

    async def _tts_all(self) -> None:
        sem = asyncio.Semaphore(TTS_CONCURRENCY)
        await asyncio.gather(*(self._tts_one(sem, s.id, s.text) for s in self.plan.sentences()))

    def generate_narration(self) -> None:
        n = len(self.plan.sentences())
        log.info("Anlatıcı sesi üretiliyor: %d cümle, ses: %s", n, self.lang.voices[0])
        t0 = time.monotonic()
        asyncio.run(self._tts_all())
        log.info("Anlatım hazır (%.0f sn).", time.monotonic() - t0)

    # ================================================================== #
    def write_credits(self) -> Path:
        path = config.OUTPUT_DIR / "credits.json"
        uniq = {(c.kind, c.url): c.__dict__ for c in self.credits}
        path.write_text(json.dumps(list(uniq.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def run(self) -> None:
        log.info("Medya üretimi başlatılıyor...")
        self.generate_narration()
        self.prepare_all_visuals()
        self.write_credits()
        total = sum(self.stats.values()) or 1
        ai_share = self.stats.get("ai_image", 0) / total
        log.info("Tüm medya hazır. Yapay zekâ görsel payı: %%%d", round(ai_share * 100))
