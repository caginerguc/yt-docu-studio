"""Kanalı düzenler (tek seferlik, tekrar çalıştırılabilir): GitHub Actions'ta 'Kanal Düzenle' workflow'u.

  1. Kanal ayarları: açıklama, anahtar kelimeler, ülke, varsayılan dil
  2. Abone olmayanlara gösterilen kanal fragmanı: en çok izlenen belgesel (3 dk üstü)
  3. Koleksiyon oynatma listeleri + mevcut belgeselleri listelere ekleme
  4. Ana sayfa rafları: en yeni videolar + dolu oynatma listeleri
  5. Abone ol filigranı (assets/branding/watermark_150.png)

Her adım ayrı denenir; biri başarısız olursa diğerleri yapılır ve rapor edilir. Tekrar çalıştırmak güvenlidir:
var olan liste, raf veya liste öğesi ikinci kez eklenmez.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from src import config, history
from src.youtube_check import _refresh, Report
from src.youtube_uploader import YouTubeUploader

WATERMARK = Path("assets/branding/watermark_150.png")
# Geçmişte koleksiyonu kaydedilmemiş ilk belgeseller (sonrakiler history.collection_id'den okunur)
LEGACY_COLLECTIONS = {
    "oHvVEAB9iic": "dunkle_korridore",              # Flick-Imperium
    "p-33ESoqeBw": "schatten_genies",               # Konrad Zuses V4
    "awDbO0amPBg": "intrigen_und_zusammenbrueche",  # Der Hase des Königs (Mary Toft)
}


def _seconds(iso: str) -> int:
    m = re.fullmatch(r"P(?:\d+D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    return 0 if not m else int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)


def _set_watermark(creds, channel_id: str, png: Path) -> None:
    """watermarks.set: JSON ayarları + PNG tek bir multipart/related istekte, doğrudan HTTP ile."""
    import json
    import uuid

    from google.auth.transport.requests import AuthorizedSession

    meta = {"timing": {"type": "offsetFromStart", "offsetMs": 0},
            "position": {"type": "corner", "cornerPosition": "topRight"}}
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{json.dumps(meta)}\r\n--{boundary}\r\nContent-Type: image/png\r\n\r\n").encode() \
        + png.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    r = AuthorizedSession(creds).post(
        "https://www.googleapis.com/upload/youtube/v3/watermarks/set",
        params={"channelId": channel_id, "uploadType": "multipart"}, data=body,
        headers={"Content-Type": f"multipart/related; boundary={boundary}"}, timeout=60)
    if r.status_code not in (200, 204):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")


def main() -> int:
    config.setup_logging()
    rep = Report()
    creds = _refresh(rep)
    if creds is None:
        print(rep.text())
        return 1
    from googleapiclient.discovery import build

    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    up = YouTubeUploader()
    up._yt = yt
    ch = yt.channels().list(part="snippet,brandingSettings,contentDetails", mine=True).execute()["items"][0]
    cid = ch["id"]
    rep.add("info", f"Kanal: {ch['snippet']['title']} ({cid})")

    # Kanaldaki videolar (yükleme listesinden)
    uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    ids, token = [], None
    while True:
        res = yt.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=50,
                                      pageToken=token).execute()
        ids += [i["contentDetails"]["videoId"] for i in res.get("items", [])]
        token = res.get("nextPageToken")
        if not token:
            break
    videos = []
    for i in range(0, len(ids), 50):
        videos += yt.videos().list(part="snippet,contentDetails,statistics,status",
                                   id=",".join(ids[i:i + 50])).execute().get("items", [])
    docs = [v for v in videos if _seconds(v["contentDetails"]["duration"]) >= 180
            and v["status"].get("privacyStatus") == "public"]

    # 1-2) Kanal ayarları + fragman
    try:
        bs = ch.get("brandingSettings", {})
        chan = bs.setdefault("channel", {})
        chan.update({"description": config.CHANNEL_DESCRIPTION, "keywords": config.CHANNEL_KEYWORDS,
                     "country": config.CHANNEL_COUNTRY, "defaultLanguage": "de"})
        trailer = max(docs, key=lambda v: int(v["statistics"].get("viewCount", 0)), default=None)
        if trailer:
            chan["unsubscribedTrailer"] = trailer["id"]
        yt.channels().update(part="brandingSettings", body={"id": cid, "brandingSettings": bs}).execute()
        rep.add("ok", "Kanal ayarları güncellendi: açıklama, anahtar kelimeler, ülke DE, dil Almanca")
        if trailer:
            rep.add("ok", f"Kanal fragmanı (abone olmayanlara): {trailer['snippet']['title']}",
                    f"https://youtu.be/{trailer['id']} ({trailer['statistics'].get('viewCount', 0)} izlenme)")
    except Exception as e:
        rep.add("error", "Kanal ayarları güncellenemedi", str(e)[:300])

    # 3) Oynatma listeleri + mevcut belgeseller
    col_of = dict(LEGACY_COLLECTIONS)
    for h in history.load():
        if h.get("video_id") and h.get("collection_id"):
            col_of[h["video_id"]] = h["collection_id"]
    playlist_ids: dict[str, str] = {}
    filled: set[str] = set()
    for col in config.COLLECTIONS:
        try:
            pid = up.find_or_create_playlist(col["playlist"], col.get("playlist_desc", ""))
            playlist_ids[col["id"]] = pid
            added = []
            for v in docs:
                if col_of.get(v["id"]) == col["id"]:
                    filled.add(col["id"])
                    if up.add_to_playlist(pid, v["id"]):
                        added.append(v["snippet"]["title"][:50])
            rep.add("ok", f"Oynatma listesi: {col['playlist']}",
                    ("eklendi: " + "; ".join(added)) if added else "hazır (yeni eklenen video yok)")
        except Exception as e:
            rep.add("error", f"Oynatma listesi oluşturulamadı: {col['playlist']}", str(e)[:300])

    # 4) Ana sayfa rafları (YouTube türleri küçük harfle döndürüyor: 'recentuploads')
    try:
        existing = yt.channelSections().list(part="snippet,contentDetails", mine=True).execute().get("items", [])
        # Önceki bir çalıştırmanın eklediği mükerrer "en yeni videolar" rafını kaldır (ilki kalır)
        recents = [s for s in existing if s["snippet"]["type"].lower() == "recentuploads"]
        removed = 0
        for s in recents[1:]:
            yt.channelSections().delete(id=s["id"]).execute()
            removed += 1
        if removed:
            rep.add("ok", f"Mükerrer 'en yeni videolar' rafı kaldırıldı ({removed})")
            existing = [s for s in existing if s not in recents[1:]]
        have_types = {s["snippet"]["type"].lower() for s in existing}
        have_pls = {p for s in existing for p in (s.get("contentDetails", {}).get("playlists") or [])}
        pos, made = len(existing), []
        if "recentuploads" not in have_types:
            yt.channelSections().insert(part="snippet", body={"snippet": {"type": "recentUploads",
                                                                         "position": 0}}).execute()
            made.append("En yeni videolar")
        for col in config.COLLECTIONS:
            pid = playlist_ids.get(col["id"])
            if pid and col["id"] in filled and pid not in have_pls:
                yt.channelSections().insert(part="snippet,contentDetails", body={
                    "snippet": {"type": "singlePlaylist", "position": pos},
                    "contentDetails": {"playlists": [pid]}}).execute()
                pos += 1
                made.append(col["name"])
        rep.add("ok", "Ana sayfa rafları", ("eklendi: " + ", ".join(made)) if made else "zaten düzenli")
    except Exception as e:
        rep.add("warn", "Ana sayfa rafları API ile tamamlanamadı (YouTube raf sınırı dolu olabilir)",
                "YouTube Studio > Özelleştirme > Düzen bölümünden oynatma listesi rafları elle eklenebilir. "
                "Hata: " + str(e)[:150])

    # 5) Abone ol filigranı: Google istemci kütüphanesi (httplib2 0.32) bu istekte 'range() arg 3 must not be
    # zero' hatası verdiği için istek doğrudan (requests + yetkili oturum) multipart olarak gönderilir.
    try:
        if not WATERMARK.exists():
            raise FileNotFoundError(WATERMARK)
        _set_watermark(creds, cid, WATERMARK)
        rep.add("ok", "Abone ol filigranı ayarlandı (anahtar deliği simgesi, tüm video boyunca)")
    except Exception as e:
        rep.add("warn", "Filigran API ile ayarlanamadı",
                "YouTube Studio > Özelleştirme > Marka > Video filigranı bölümünden assets/branding/"
                "watermark_150.png yüklenebilir. Hata: " + str(e)[:300])

    text = rep.text()
    print("\n=== Kanal Düzenleme Raporu ===\n" + text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Kanal Düzenleme Raporu\n\n```\n" + text + "\n```\n")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
