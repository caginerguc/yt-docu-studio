"""Aşama 3 — Medya tedariki: anlatıcı sesi + gerçek görüntü.

Her cümle için:
  * Ses   : edge-tts ile cümle başına MP3  ->  output/audio/<sentence_id>.mp3
Her kesit (shot) için, plan.source'a göre öncelik sırası (gerçek görüntü her zaman AI'den önce):
  * archive_photo : Wikimedia (PD/CC0, arşiv sorgusu) -> Pexels foto -> Wikimedia (stok sorgusu) -> AI
  * stock_video   : Pexels VIDEO -> Pexels foto -> Wikimedia (stok sorgusu) -> AI
  * ai_image      : AI (Pollinations) -> Pexels foto -> Wikimedia
Hiçbiri bulunamazsa boş tuval yerine en yakın kesitin görüntüsü yeniden kullanılır (renderer farklı kamera
hareketi ve farklı başlangıç noktası uygular); hiç görüntü yoksa son çare koyu tuval.

Teşhis: her sağlayıcının ilk hatası HTTP koduyla WARNING olarak loglanır; Pexels başta tek istekle sınanır,
anahtar reddedilirse çalışma boyunca kapatılır. Özet output/media_report.json dosyasına yazılır.
Pollinations anonim kullanımda aynı anda tek isteğe izin verir: istekler sıraya alınır, 429'da beklenir.
Dosya adları: output/scenes/<sentence_id>_shot<N>.mp4 (video) veya .jpg (durağan).
Kullanılan dış kaynaklar output/credits.json dosyasına yazılır (açıklama metni için).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
FETCH_WORKERS = 4
MAX_REUSE_PER_SOURCE = 3        # eksik kesit doldururken aynı görüntü en fazla bu kadar tekrar
AI_BUDGET_S = float(os.environ.get("AI_BUDGET_S", "1800"))          # AI görsellere toplam en fazla 30 dk
AI_MIN_INTERVAL_S = float(os.environ.get("AI_MIN_INTERVAL_S", "3" if config.POLLINATIONS_API_KEY else "12"))
AI_MAX_CONSEC_FAIL = 8          # arka arkaya bu kadar başarısız AI denemesi -> bu çalışmada AI kapalı

_STOP = {
    "the", "and", "with", "from", "into", "onto", "over", "under", "of", "in", "on", "at", "to", "for", "by",
    "an", "its", "his", "her", "their", "this", "that", "these", "those", "while", "where", "which", "who",
    "is", "are", "was", "were", "be", "being", "very", "some", "lit", "dimly", "softly", "slowly", "scene",
    "shot", "view", "image", "photo", "cinematic", "dramatic", "moody", "dark", "light", "lighting", "shadows",
    "close", "wide", "angle", "background", "foreground", "atmosphere", "atmospheric", "single", "one",
}


def keywords(text: str, n: int = 4) -> str:
    """AI istem metninden stok arama sorgusu çıkarır: 'A dimly lit study with candles...' -> 'study candles ...'."""
    out: list[str] = []
    for w in re.findall(r"[a-zA-Z][a-zA-Z'-]+", (text or "").lower()):
        if len(w) < 3 or w in _STOP or w in out:
            continue
        out.append(w)
        if len(out) == n:
            break
    return " ".join(out)


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
        # Wikimedia, iletişim bilgisi içeren User-Agent ister; diğer servisler için standart oturum.
        self.wiki = requests.Session()
        self.wiki.headers["User-Agent"] = (
            f"MakroKontextBot/3.2 (https://github.com/caginerguc/yt-docu-studio; "
            f"{config.WIKI_CONTACT or 'contact-not-set'})"
        )
        self.http = requests.Session()
        self._used: set[str] = set()
        self._lock = threading.Lock()
        self._ai_lock = threading.Lock()
        self._cache: dict[tuple[str, str], list] = {}
        self.credits: list[Credit] = []
        self.stats: Counter = Counter()
        self.errors: Counter = Counter()
        self.pexels_ok = bool(config.PEXELS_API_KEY) and config.STOCK_ENABLED
        self.pexels_note = "" if self.pexels_ok else "anahtar yok veya STOCK_ENABLED=0"
        self.ai_ok = True
        self.ai_spent = 0.0
        self._ai_last = 0.0
        self._ai_auth_fail = 0
        self._ai_consec_fail = 0

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
            self.stats[name] += 1

    def _credit(self, c: Credit) -> None:
        with self._lock:
            self.credits.append(c)

    def _fail(self, provider: str, code: str, detail: str = "") -> None:
        key = f"{provider} {code}"
        with self._lock:
            self.errors[key] += 1
            first = self.errors[key] == 1
        if first:   # her (sağlayıcı, kod) çifti için bir kez: log kirlenmez ama sebep görünür
            log.warning("[%s] istek başarısız: %s %s", provider, code, " ".join(detail.split())[:180])

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

    @staticmethod
    def _body(r: requests.Response) -> str:
        ctype = r.headers.get("Content-Type", "")
        return r.text[:200] if ("text" in ctype or "json" in ctype) else ""

    def _fetch(self, url: str, provider: str, session: requests.Session | None = None,
               **kw) -> requests.Response | None:
        """Her durum kodunda yanıtı döndürür; ağ hatasında None (hata kaydedilir)."""
        try:
            return (session or self.http).get(url, timeout=kw.pop("timeout", HTTP_TIMEOUT), **kw)
        except requests.RequestException as e:
            self._fail(provider, type(e).__name__, str(e))
            return None

    def _get(self, url: str, provider: str, session: requests.Session | None = None,
             **kw) -> requests.Response | None:
        """Yalnızca 200 yanıtı döndürür; diğerlerini sağlayıcı/kod bazında kaydeder."""
        r = self._fetch(url, provider, session, **kw)
        if r is None:
            return None
        if r.status_code != 200:
            self._fail(provider, f"HTTP {r.status_code}", self._body(r))
            return None
        return r

    # ================================================================== #
    # Ön kontrol
    # ================================================================== #
    @staticmethod
    def _pexels_headers() -> dict:
        return {"Authorization": config.PEXELS_API_KEY}

    def check_providers(self) -> None:
        """Pexels'i tek istekle sınar; anahtar reddedilirse 100'lerce boş istek atmadan kapatır."""
        if not self.pexels_ok:
            log.warning("Pexels kapalı (%s): stok video yok, yalnızca arşiv ve AI görselleri.", self.pexels_note)
            return
        r = self._fetch("https://api.pexels.com/v1/search", "pexels", params={"query": "forest", "per_page": 1},
                        headers=self._pexels_headers())
        if r is None:
            log.warning("Pexels ön kontrolü ağ hatası verdi; yine de denenecek.")
            return
        if r.status_code == 200:
            log.info("Pexels bağlantısı tamam (bu saat kalan istek: %s).", r.headers.get("X-Ratelimit-Remaining", "?"))
            return
        body = " ".join(self._body(r)[:160].split())
        if r.status_code in (401, 403):
            self.pexels_ok = False
            self.pexels_note = f"HTTP {r.status_code}: anahtar reddedildi"
            log.error("Pexels anahtarı REDDEDİLDİ (HTTP %d: %s). GitHub > Settings > Secrets > PEXELS_API_KEY "
                      "değerini kontrol et (yalnızca anahtarın kendisi, başında 'Bearer' olmadan).", r.status_code, body)
        elif r.status_code == 429:
            self.pexels_ok = False
            self.pexels_note = "HTTP 429: saatlik istek limiti dolu"
            log.error("Pexels saatlik istek limiti dolu (HTTP 429); bu çalışmada Pexels kapalı.")
        else:
            log.warning("Pexels ön kontrolü HTTP %d döndü (%s); yine de denenecek.", r.status_code, body)

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

    def _wiki_candidates(self, query: str) -> list[dict]:
        key = ("wiki", query.lower())
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 20,
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 2560,
        }
        r = self._get("https://commons.wikimedia.org/w/api.php", "wikimedia", self.wiki, params=params)
        out: list[dict] = []
        if r:
            try:
                pages = sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
            except ValueError:
                pages = []
            for p in pages:
                info = (p.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ("image/jpeg", "image/png", "image/tiff", "image/webp"):
                    continue
                if int(info.get("width", 0)) < MIN_IMAGE_WIDTH:
                    continue
                ok, lic = self._license_ok(info.get("extmetadata", {}))
                url = info.get("thumburl") or info.get("url")
                if ok and url:
                    out.append({"url": url, "title": p.get("title", ""), "page": info.get("descriptionurl", url),
                                "license": lic})
        with self._lock:
            self._cache[key] = out
        return out

    def _wikimedia(self, query: str, path: Path) -> bool:
        if not query:
            return False
        for c in self._wiki_candidates(query):
            if not self._claim(c["url"]):
                continue
            img = self._get(c["url"], "wikimedia-dosya", self.wiki, timeout=30)
            if img and self._save_image(img.content, path, MIN_IMAGE_WIDTH):
                self._credit(Credit("wikimedia", c["title"], c["page"], c["license"]))
                self._count("archive_photo")
                return True
        return False

    # ================================================================== #
    # Pexels: video ve foto (arama sonuçları önbelleğe alınır: aynı sorgu = tek istek)
    # ================================================================== #
    def _pexels_search(self, kind: str, query: str) -> list[dict]:
        key = (f"pexels-{kind}", query.lower())
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if kind == "video":
            url, params, field = ("https://api.pexels.com/videos/search",
                                  {"query": query, "per_page": 15, "orientation": "landscape", "size": "medium"},
                                  "videos")
        else:
            url, params, field = ("https://api.pexels.com/v1/search",
                                  {"query": query, "per_page": 20, "orientation": "landscape"}, "photos")
        r = self._fetch(url, "pexels", params=params, headers=self._pexels_headers())
        items: list[dict] = []
        if r is not None:
            if r.status_code == 200:
                try:
                    items = r.json().get(field, [])
                except ValueError:
                    items = []
            else:
                self._fail("pexels", f"HTTP {r.status_code}", self._body(r))
                if r.status_code in (401, 403, 429):
                    self.pexels_ok = False
                    self.pexels_note = f"çalışma sırasında HTTP {r.status_code}"
                    log.error("Pexels HTTP %d döndü; bu çalışmada Pexels kapatıldı.", r.status_code)
        with self._lock:
            self._cache[key] = items
        return items

    def _pexels_video(self, query: str, path: Path) -> bool:
        if not (self.pexels_ok and query):
            return False
        for v in self._pexels_search("video", query):
            if int(v.get("duration") or 0) < STOCK_MIN_SECONDS:
                continue
            files = [f for f in v.get("video_files", [])
                     if f.get("file_type") == "video/mp4" and f.get("width") and f.get("link")
                     and STOCK_MIN_W <= int(f["width"]) <= STOCK_MAX_W and int(f["width"]) > int(f.get("height") or 0)]
            if not files:
                continue
            best = max(files, key=lambda f: int(f["width"]))
            if not self._claim(f"pexels-video-{v.get('id')}"):
                continue
            data = self._get(best["link"], "pexels-dosya", timeout=120)
            if data and len(data.content) > 100_000:
                path.write_bytes(data.content)
                self._credit(Credit("pexels_video", (v.get("user") or {}).get("name", "Pexels"),
                                    v.get("url", ""), "Pexels License"))
                self._count("stock_video")
                return True
        return False

    def _pexels_photo(self, query: str, path: Path) -> bool:
        if not (self.pexels_ok and query):
            return False
        for ph in self._pexels_search("photo", query):
            src = ph.get("src") or {}
            url = src.get("large2x") or src.get("original")
            if not url or not self._claim(f"pexels-photo-{ph.get('id')}"):
                continue
            img = self._get(url, "pexels-dosya", timeout=30)
            if img and self._save_image(img.content, path, MIN_IMAGE_WIDTH):
                self._credit(Credit("pexels_photo", ph.get("photographer", "Pexels"), ph.get("url", ""),
                                    "Pexels License"))
                self._count("stock_photo")
                return True
        return False

    # ================================================================== #
    # Yapay zekâ görseli (sıralı, hız sınırına saygılı)
    # ================================================================== #
    def _pollinations(self, prompt: str, path: Path, seed: int) -> bool:
        if not prompt or not self.ai_ok:
            return False
        full = quote(f"{prompt}, {config.ART_DIRECTION}")
        headers = {"Authorization": f"Bearer {config.POLLINATIONS_API_KEY}"} if config.POLLINATIONS_API_KEY else {}
        with self._ai_lock:                       # anonim kullanımda aynı anda tek istek
            for model in config.POLLINATIONS_MODELS:
                for attempt in range(3):
                    if not self.ai_ok:
                        return False
                    if self.ai_spent >= AI_BUDGET_S:
                        self.ai_ok = False
                        log.warning("AI görsel zaman bütçesi doldu (%.0f sn); kalan kesitler gerçek görüntü/tekrar "
                                    "kullanımla doldurulacak.", self.ai_spent)
                        return False
                    t0 = time.monotonic()
                    wait = AI_MIN_INTERVAL_S - (t0 - self._ai_last)
                    if wait > 0:
                        time.sleep(wait)
                    url = (f"{config.POLLINATIONS_BASE_URL}/{full}?width={config.AI_IMAGE_W}&height={config.AI_IMAGE_H}"
                           f"&seed={seed + attempt * 101}&model={model}&nologo=true&enhance=false")
                    r = self._fetch(url, "pollinations", headers=headers, timeout=AI_TIMEOUT)
                    self._ai_last = time.monotonic()
                    code = r.status_code if r is not None else 0
                    if code == 200 and self._save_image(r.content, path):
                        self.ai_spent += time.monotonic() - t0
                        self._ai_auth_fail = self._ai_consec_fail = 0
                        self._credit(Credit("pollinations", prompt[:80], "https://pollinations.ai", "AI generated"))
                        self._count("ai_image")
                        return True
                    if code and code != 200:
                        self._fail("pollinations", f"HTTP {code}", self._body(r))
                    self._ai_consec_fail += 1
                    if self._ai_consec_fail >= AI_MAX_CONSEC_FAIL:
                        self.ai_ok = False
                        self.ai_spent += time.monotonic() - t0
                        log.error("Pollinations arka arkaya %d kez başarısız (son kod: %s); AI görselleri bu çalışmada "
                                  "kapalı.", self._ai_consec_fail, code or "ağ hatası")
                        return False
                    if code in (401, 402, 403):
                        self._ai_auth_fail += 1
                        if self._ai_auth_fail >= 3:
                            self.ai_ok = False
                            log.error("Pollinations erişimi reddediyor (HTTP %d); AI görselleri bu çalışmada kapalı. "
                                      "Ücretsiz anahtar: https://enter.pollinations.ai -> POLLINATIONS_API_KEY", code)
                        self.ai_spent += time.monotonic() - t0
                        break                       # sıradaki model
                    if code in (0, 429, 500, 502, 503, 504):
                        ra = r.headers.get("Retry-After", "") if r is not None else ""
                        back = float(ra) if ra.isdigit() else 10 * 2 ** attempt
                        time.sleep(min(back, 60))
                        self.ai_spent += time.monotonic() - t0
                        continue                    # aynı model, yeni deneme
                    self.ai_spent += time.monotonic() - t0
                    break                           # 400/404 veya bozuk görsel: sıradaki model
        return False

    def _canvas(self, path: Path, index: int) -> None:
        c = (14 + index * 7 % 20, 16 + index * 5 % 20, 22 + index * 11 % 24)
        Image.new("RGB", (config.VIDEO_W, config.VIDEO_H), c).save(path, "JPEG", quality=90)
        self._count("fallback_canvas")

    # ================================================================== #
    # Kesit başına tedarik
    # ================================================================== #
    @staticmethod
    def shot_paths(sentence_id: str, idx: int) -> tuple[Path, Path]:
        base = config.SCENES_DIR / f"{sentence_id}_shot{idx}"
        return base.with_suffix(".mp4"), base.with_suffix(".jpg")

    @staticmethod
    def _existing(sentence_id: str, idx: int) -> Path | None:
        vid, img = MediaGenerator.shot_paths(sentence_id, idx)
        if vid.exists() and vid.stat().st_size > 100_000:
            return vid
        if img.exists() and img.stat().st_size > config.MIN_IMAGE_BYTES:
            return img
        return None

    def fetch_shot(self, sentence_id: str, idx: int, shot: PlanShot, seed: int, n: int) -> str:
        if self._existing(sentence_id, idx):
            return "cached"
        vid, img = self.shot_paths(sentence_id, idx)
        archive_q = shot.archive_query_en
        stock_q = shot.stock_query_en or archive_q or keywords(shot.ai_prompt_en)
        ai_prompt = shot.ai_prompt_en or stock_q
        if shot.source == "archive_photo":
            chain = (lambda: self._wikimedia(archive_q, img),
                     lambda: self._pexels_photo(stock_q, img),
                     lambda: stock_q != archive_q and self._wikimedia(stock_q, img),
                     lambda: self._pollinations(ai_prompt, img, seed))
        elif shot.source == "stock_video":
            chain = (lambda: self._pexels_video(shot.stock_query_en or stock_q, vid),
                     lambda: self._pexels_photo(stock_q, img),
                     lambda: self._wikimedia(stock_q, img),
                     lambda: self._pollinations(ai_prompt, img, seed))
        else:
            chain = (lambda: self._pollinations(ai_prompt, img, seed),
                     lambda: self._pexels_photo(stock_q, img),
                     lambda: self._wikimedia(stock_q, img))
        for step in chain:
            if step():
                return "ok"
        return "missing"

    def _fill_missing(self, tasks: list[tuple], status: list[str]) -> None:
        """Eksik kesitleri en yakın başarılı kesitin görüntüsüyle doldurur (boş tuval son çare)."""
        good = [(i, p) for i, p in ((i, self._existing(t[0], t[1])) for i, t in enumerate(tasks)) if p is not None]
        reuse: Counter = Counter()
        for i, t in enumerate(tasks):
            if status[i] != "missing":
                continue
            vid, img = self.shot_paths(t[0], t[1])
            pick = None
            # önce aynı cümle, sonra en yakın kesit; bir görüntü en fazla MAX_REUSE_PER_SOURCE kez
            for _, p in sorted(good, key=lambda g: (g[1].name.split("_shot")[0] != t[0], abs(g[0] - i))):
                if reuse[p] < MAX_REUSE_PER_SOURCE:
                    pick = p
                    break
            if pick is None and good:
                pick = min(good, key=lambda g: reuse[g[1]])[1]
            if pick is not None:
                shutil.copyfile(pick, vid if pick.suffix == ".mp4" else img)
                reuse[pick] += 1
                self._count("reused")
            else:
                self._canvas(img, t[4])

    def prepare_all_visuals(self) -> None:
        self.check_providers()
        seed0 = int(hashlib.sha1(self.plan.topic_id.encode()).hexdigest()[:6], 16) % 100000
        tasks = []
        n = 0
        for s in self.plan.sentences():
            for i, sh in enumerate(s.shots, 1):
                n += 1
                tasks.append((s.id, i, sh, seed0 + n, n))
        log.info("Görseller tedarik ediliyor: %d kesit (Wikimedia PD -> Pexels video/foto -> AI)", len(tasks))
        t0 = time.monotonic()
        status = ["missing"] * len(tasks)
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
            futs = {ex.submit(self.fetch_shot, *t): k for k, t in enumerate(tasks)}
            for f, k in futs.items():
                try:
                    status[k] = f.result()
                except Exception as e:  # tek kesit tüm üretimi durdurmasın
                    log.warning("Kesit tedarik hatası (%s_shot%d): %s", tasks[k][0], tasks[k][1], e)
        missing = status.count("missing")
        if missing:
            log.warning("%d/%d kesit için görüntü bulunamadı; en yakın kesitlerin görüntüleriyle dolduruluyor.",
                        missing, len(tasks))
            self._fill_missing(tasks, status)
        log.info("Görsel tedariki bitti (%.0f sn). Dağılım: %s", time.monotonic() - t0, dict(self.stats))

        thumb_bg = self.scenes_dir / "thumb_bg.jpg"
        if not thumb_bg.exists():
            q = keywords(self.plan.seo.thumbnail_prompt_en) or "dark mysterious archive"
            if not (self._pollinations(self.plan.seo.thumbnail_prompt_en, thumb_bg, seed0 + 999)
                    or self._pexels_photo(q, thumb_bg) or self._wikimedia(q, thumb_bg)):
                stills = [p for p in (self._existing(t[0], t[1]) for t in tasks) if p is not None and p.suffix == ".jpg"]
                if stills:
                    shutil.copyfile(stills[0], thumb_bg)
                else:
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

    def report(self) -> dict:
        total = sum(self.stats.values()) or 1
        real = sum(self.stats.get(k, 0) for k in ("stock_video", "stock_photo", "archive_photo"))
        weak = self.stats.get("reused", 0) + self.stats.get("fallback_canvas", 0)
        return {
            "distribution": dict(self.stats),
            "real_footage_share": round(real / total, 2),
            "ai_share": round(self.stats.get("ai_image", 0) / total, 2),
            "reused_or_blank_share": round(weak / total, 2),
            "provider_errors": dict(self.errors),
            "pexels": "ok" if self.pexels_ok else f"kapalı ({self.pexels_note})",
            "ai": "ok" if self.ai_ok else "kapalı",
            "ai_seconds": round(self.ai_spent),
        }

    def run(self) -> None:
        log.info("Medya üretimi başlatılıyor...")
        self.generate_narration()
        self.prepare_all_visuals()
        self.write_credits()
        rep = self.report()
        (config.OUTPUT_DIR / "media_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
        log.info("Tüm medya hazır. Gerçek görüntü payı: %%%d, yapay zekâ: %%%d, tekrar/boş: %%%d",
                 round(rep["real_footage_share"] * 100), round(rep["ai_share"] * 100),
                 round(rep["reused_or_blank_share"] * 100))
        if rep["reused_or_blank_share"] > 0.25:
            log.warning("Görsel kalitesi zayıf: kesitlerin %%%d'i tekrar kullanım veya boş. Sağlayıcı hataları: %s",
                        round(rep["reused_or_blank_share"] * 100), rep["provider_errors"] or "-")
