"""Aşama 6 — YouTube yükleme (tarayıcısız, refresh token ile).

  * Video (sürdürülebilir/resumable yükleme, geçici hatalarda yeniden deneme)
  * Özel kapak (kanalın telefon doğrulaması gerekir; yoksa uyarı verip devam eder)
  * Altyazı (SRT) — token'da youtube.force-ssl kapsamı yoksa uyarı verip devam eder
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src import config

log = logging.getLogger("makrokontext.youtube")

TOKEN_URI = "https://oauth2.googleapis.com/token"
RETRY_STATUS = {500, 502, 503, 504}
MAX_RETRIES = 8
CHUNK = 16 * 1024 * 1024


class UploadError(RuntimeError):
    pass


@dataclass
class VideoMeta:
    title: str
    description: str
    tags: list[str]
    language: str
    privacy: str = "private"
    category_id: str = config.YT_CATEGORY_ID


def _clean_tags(tags: list[str]) -> list[str]:
    """YouTube: etiketlerin toplamı 500 karakteri geçemez, '<' '>' yasak."""
    out, total = [], 0
    for t in tags:
        t = t.replace("<", "").replace(">", "").replace("#", "").strip()
        if not t or t.lower() in {x.lower() for x in out}:
            continue
        cost = len(t) + (2 if " " in t else 0) + 1
        if total + cost > 480:
            break
        out.append(t)
        total += cost
    return out


class YouTubeUploader:
    def __init__(self) -> None:
        self.client_id = config.YT_CLIENT_ID
        self.client_secret = config.YT_CLIENT_SECRET
        self.refresh_token = config.YT_REFRESH_TOKEN
        self._yt = None

    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.refresh_token)

    def _service(self):
        if self._yt is not None:
            return self._yt
        if not self.is_configured():
            raise UploadError("YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN eksik.")
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        # scopes verilmez: token hangi kapsamlarla alındıysa onlarla yenilenir (invalid_scope riski yok).
        creds = Credentials(token=None, refresh_token=self.refresh_token, token_uri=TOKEN_URI,
                            client_id=self.client_id, client_secret=self.client_secret)
        try:
            creds.refresh(Request())
        except Exception as e:
            raise UploadError(f"YouTube token yenilenemedi (refresh token iptal/süresi dolmuş olabilir): {e}") from e
        self._yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
        return self._yt

    # ------------------------------------------------------------------ #
    def upload_video(self, video: Path, meta: VideoMeta) -> str:
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        if not video.exists():
            raise UploadError(f"Video bulunamadı: {video}")
        if meta.privacy not in {"private", "unlisted", "public"}:
            raise UploadError(f"Geçersiz gizlilik: {meta.privacy}")
        lang = meta.language.lower()
        body = {
            "snippet": {
                "title": meta.title.replace("<", "").replace(">", "")[:100],
                "description": meta.description.replace("<", "").replace(">", "")[:4900],
                "tags": _clean_tags(meta.tags),
                "categoryId": meta.category_id,
                "defaultLanguage": lang,
                "defaultAudioLanguage": lang,
            },
            "status": {
                "privacyStatus": meta.privacy,
                "selfDeclaredMadeForKids": config.YT_MADE_FOR_KIDS,
                "containsSyntheticMedia": config.YT_CONTAINS_SYNTHETIC_MEDIA,
            },
        }
        media = MediaFileUpload(str(video), mimetype="video/mp4", chunksize=CHUNK, resumable=True)
        request = self._service().videos().insert(part="snippet,status", body=body, media_body=media)
        log.info("YouTube'a yükleniyor: '%s' (%s, %.0f MB)", body["snippet"]["title"], meta.privacy,
                 video.stat().st_size / 1e6)

        response, retries, last_pct = None, 0, -1
        while response is None:
            try:
                status, response = request.next_chunk()
                retries = 0
                if status:
                    pct = int(status.progress() * 100)
                    if pct // 10 != last_pct // 10:
                        log.info("Yükleme: %%%d", pct)
                        last_pct = pct
            except HttpError as e:
                if e.resp.status not in RETRY_STATUS:
                    raise UploadError(f"YouTube reddetti ({e.resp.status}): {e}") from e
                retries += 1
                if retries > MAX_RETRIES:
                    raise UploadError(f"Yükleme {MAX_RETRIES} denemeden sonra başarısız: {e}") from e
                wait = min(2 ** retries, 64) + random.uniform(0, 1)
                log.warning("Geçici YouTube hatası %s, %.0f sn sonra tekrar", e.resp.status, wait)
                time.sleep(wait)
            except (ConnectionError, TimeoutError, OSError) as e:
                retries += 1
                if retries > MAX_RETRIES:
                    raise UploadError(f"Bağlantı hatası: {e}") from e
                time.sleep(min(2 ** retries, 64))

        if "id" not in response:
            raise UploadError(f"Beklenmeyen yanıt: {response}")
        vid = response["id"]
        log.info("Yüklendi: https://youtu.be/%s", vid)
        return vid

    def set_thumbnail(self, video_id: str, thumb: Path | None) -> bool:
        if not thumb or not thumb.exists():
            return False
        from googleapiclient.http import MediaFileUpload
        try:
            self._service().thumbnails().set(
                videoId=video_id, media_body=MediaFileUpload(str(thumb), mimetype="image/jpeg")).execute()
            log.info("Kapak ayarlandı: %s", thumb.name)
            return True
        except Exception as e:
            log.warning("Kapak yüklenemedi (kanal telefon doğrulaması gerekir): %s", str(e)[:200])
            return False

    def add_captions(self, video_id: str, srt: Path | None, language: str) -> bool:
        if not srt or not srt.exists() or srt.stat().st_size == 0:
            return False
        from googleapiclient.http import MediaFileUpload
        try:
            self._service().captions().insert(
                part="snippet",
                body={"snippet": {"videoId": video_id, "language": language.lower(), "name": "", "isDraft": False}},
                media_body=MediaFileUpload(str(srt), mimetype="application/octet-stream"),
            ).execute()
            log.info("Altyazı eklendi (%s).", language)
            return True
        except Exception as e:
            log.warning("Altyazı eklenemedi (token'a youtube.force-ssl kapsamı gerekebilir): %s", str(e)[:200])
            return False
