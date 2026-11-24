"""YouTube API teşhisi: token, izinler (scope), kanal ve son yüklemelerin gizlilik durumu.

  python check_youtube.py          # tam rapor (GitHub Actions: 'YouTube API Kontrol' workflow'u)
  preflight()                      # üretim başlamadan önce hızlı kontrol (main.py)

Not: API projesinin denetimden (audit) geçip geçmediği API ile sorgulanamaz. Anlamanın yolu: API ile
'public' yüklenen bir videonun YouTube'da 'private' görünmesi. Rapor bunu son yüklemeler üzerinden kontrol eder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import requests

from src import config, history

log = logging.getLogger("makrokontext.youtube_check")

TOKEN_URI = "https://oauth2.googleapis.com/token"
TOKENINFO_URI = "https://oauth2.googleapis.com/tokeninfo"
S_UPLOAD = "https://www.googleapis.com/auth/youtube.upload"
S_YOUTUBE = "https://www.googleapis.com/auth/youtube"
S_FORCE_SSL = "https://www.googleapis.com/auth/youtube.force-ssl"
TESTING_HINT = ("Google Cloud Console > APIs & Services > OAuth consent screen (Audience) bölümünde yayın durumu "
                "'Testing' ise refresh token 7 günde bir geçersiz olur: 'Publish app' ile 'In production' yapın, "
                "sonra get_yt_token.py ile yeni token alıp YT_REFRESH_TOKEN secret'ını güncelleyin.")


@dataclass
class Report:
    lines: list[tuple[str, str, str]] = field(default_factory=list)   # (seviye, başlık, ayrıntı)

    def add(self, level: str, title: str, detail: str = "") -> None:
        self.lines.append((level, title, detail))

    @property
    def ok(self) -> bool:
        return not any(lv == "error" for lv, _, _ in self.lines)

    def text(self) -> str:
        icon = {"ok": "✅", "warn": "⚠️", "error": "❌", "info": "ℹ️"}
        out = []
        for lv, title, detail in self.lines:
            out.append(f"{icon.get(lv, '-')} {title}" + (f"\n    {detail}" if detail else ""))
        return "\n".join(out)


def _credentials():
    from google.oauth2.credentials import Credentials

    return Credentials(token=None, refresh_token=config.YT_REFRESH_TOKEN, token_uri=TOKEN_URI,
                       client_id=config.YT_CLIENT_ID, client_secret=config.YT_CLIENT_SECRET)


def _refresh(rep: Report):
    from google.auth.transport.requests import Request

    if not (config.YT_CLIENT_ID and config.YT_CLIENT_SECRET and config.YT_REFRESH_TOKEN):
        rep.add("error", "YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN eksik",
                "GitHub > Settings > Secrets and variables > Actions altında üçü de tanımlı olmalı.")
        return None
    creds = _credentials()
    try:
        creds.refresh(Request())
    except Exception as e:
        msg = str(e)
        if "invalid_grant" in msg:
            rep.add("error", "Refresh token geçersiz (invalid_grant): token iptal edilmiş veya süresi dolmuş",
                    TESTING_HINT)
        elif "invalid_client" in msg or "unauthorized_client" in msg:
            rep.add("error", "Client ID / Client Secret hatalı (invalid_client)",
                    "YT_CLIENT_ID ve YT_CLIENT_SECRET, refresh token'ı alırken kullanılan OAuth istemcisine ait olmalı.")
        else:
            rep.add("error", "Token yenilenemedi", msg[:300])
        return None
    rep.add("ok", "Token yenilendi (refresh token geçerli)")
    return creds


def _scopes(creds, rep: Report) -> set[str]:
    try:
        r = requests.get(TOKENINFO_URI, params={"access_token": creds.token}, timeout=15)
        scopes = set((r.json().get("scope") or "").split()) if r.status_code == 200 else set()
    except (requests.RequestException, ValueError):
        scopes = set()
    if not scopes:
        rep.add("warn", "Token izinleri okunamadı (tokeninfo)")
        return scopes
    short = ", ".join(sorted(s.rsplit("/", 1)[-1] for s in scopes))
    if scopes & {S_UPLOAD, S_YOUTUBE, S_FORCE_SSL}:
        rep.add("ok", "Video yükleme izni var", f"İzinler: {short}")
    else:
        rep.add("error", "Token'da video yükleme izni yok", f"İzinler: {short}. get_yt_token.py ile yeni token alın.")
    if S_FORCE_SSL in scopes:
        rep.add("ok", "Altyazı yükleme izni var (youtube.force-ssl)")
    else:
        rep.add("warn", "Altyazı yükleme izni yok (youtube.force-ssl)",
                "Videolar yüklenir ama SRT altyazı eklenemez. Güncel get_yt_token.py ile yeni token alıp "
                "YT_REFRESH_TOKEN secret'ını güncelleyin.")
    return scopes


def _api_error(e: Exception, rep: Report, what: str) -> None:
    msg = str(e)
    if "accessNotConfigured" in msg or "has not been used" in msg or "is disabled" in msg:
        rep.add("error", "YouTube Data API v3 bu Google Cloud projesinde kapalı",
                "Google Cloud Console > APIs & Services > Library > 'YouTube Data API v3' > Enable.")
    elif "quotaExceeded" in msg:
        rep.add("error", "Günlük YouTube API kotası doldu",
                "Kota Pasifik saatiyle gece yarısı sıfırlanır (Türkiye saatiyle ~10:00-11:00).")
    else:
        rep.add("error", f"{what} başarısız", msg[:300])


def _channel(yt, rep: Report) -> dict | None:
    try:
        res = yt.channels().list(part="snippet,status,contentDetails", mine=True).execute()
    except Exception as e:
        _api_error(e, rep, "Kanal bilgisi alma")
        return None
    items = res.get("items") or []
    if not items:
        rep.add("error", "Bu Google hesabında YouTube kanalı bulunamadı",
                "Token alınırken tarayıcıda kanalın bağlı olduğu hesap (veya marka hesabı) seçilmeli.")
        return None
    ch = items[0]
    st = ch.get("status", {})
    rep.add("ok", f"Kanal: {ch['snippet'].get('title')} ({ch['id']})")
    long_up = st.get("longUploadsStatus", "?")
    if long_up == "allowed":
        rep.add("ok", "Kanal telefonla doğrulanmış: özel kapak ve 15 dk üstü video serbest")
    else:
        rep.add("warn", f"Kanal doğrulaması eksik (longUploadsStatus={long_up})",
                "Özel kapak yüklenemez ve 15 dakikadan uzun video reddedilir. "
                "https://www.youtube.com/verify adresinden telefonla doğrulayın.")
    return ch


def _recent_uploads(yt, ch: dict, rep: Report, n: int) -> None:
    uploads = (ch.get("contentDetails", {}).get("relatedPlaylists") or {}).get("uploads")
    if not uploads:
        return
    try:
        items = yt.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=n).execute()
        ids = [i["contentDetails"]["videoId"] for i in items.get("items", [])]
        vids = yt.videos().list(part="snippet,status", id=",".join(ids)).execute().get("items", []) if ids else []
    except Exception as e:
        _api_error(e, rep, "Son yüklemeleri okuma")
        return
    wanted = {h.get("video_id"): h for h in history.load() if h.get("video_id")}
    if not vids:
        rep.add("info", "Kanalda henüz video yok")
    locked = 0
    for v in vids:
        st, title = v.get("status", {}), v["snippet"].get("title", "")[:60]
        privacy, upload = st.get("privacyStatus"), st.get("uploadStatus")
        extra = st.get("rejectionReason") or st.get("failureReason") or ""
        detail = f"https://youtu.be/{v['id']} | gizlilik={privacy} | durum={upload}" + (f" | neden={extra}" if extra else "")
        ours = wanted.get(v["id"])
        if ours and ours.get("requested_privacy") == "public" and privacy == "private":
            locked += 1
            rep.add("error", f"Public istendi ama private: {title}", detail)
        elif upload in ("rejected", "failed"):
            rep.add("error", f"Yükleme reddedildi: {title}", detail)
        else:
            rep.add("info", title, detail)
    if locked:
        rep.add("error", "API projesi denetimden (audit) geçmemiş görünüyor",
                "Denetimsiz projelerden yüklenen videolar YouTube tarafından private'a kilitlenir. Başvuru: "
                "https://support.google.com/youtube/contact/yt_api_form (form türü: 'API Services - Audit and "
                "Quota Extension'). Onaylanana kadar videoları YouTube Studio'dan elle public yapabilirsiniz.")


def diagnose(recent: int = 5) -> Report:
    rep = Report()
    creds = _refresh(rep)
    if creds is None:
        return rep
    _scopes(creds, rep)
    from googleapiclient.discovery import build

    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    ch = _channel(yt, rep)
    if ch is not None:
        _recent_uploads(yt, ch, rep, recent)
    rep.add("info", "Denetim (audit) durumu API'den okunamaz",
            "Anlamanın yolu: public yüklenen bir videonun YouTube'da private görünmesi (yukarıdaki liste).")
    return rep


def preflight() -> str | None:
    """Üretimden önce: token + kanal. Sorun varsa okunur bir hata metni döndürür (yoksa None)."""
    try:
        rep = Report()
        creds = _refresh(rep)
        if creds is not None:
            from googleapiclient.discovery import build

            _channel(build("youtube", "v3", credentials=creds, cache_discovery=False), rep)
    except Exception as e:     # teşhis aracı üretimi kendi hatasıyla durdurmamalı
        log.warning("YouTube ön kontrolü yapılamadı: %s", e)
        return None
    for lv, title, detail in rep.lines:
        if lv == "warn":
            log.warning("YouTube: %s. %s", title, detail)
    errs = [f"{t}. {d}".strip() for lv, t, d in rep.lines if lv == "error"]
    return " | ".join(errs) or None
