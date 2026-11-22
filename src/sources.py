"""Araştırma araçları (hepsi ücretsiz, anahtarsız): Wikipedia/Wikimedia, Google Trends & News RSS.

Wikimedia politikası gereği her istekte iletişim bilgisi içeren User-Agent gönderilir (WIKI_CONTACT).
Ağ hataları pipeline'ı kırmaz; ilgili sinyal boş döner.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from src import config
from src.grounding import shingles

log = logging.getLogger("makrokontext.sources")


@dataclass
class Source:
    id: str
    lang: str
    title: str
    url: str
    text: str
    _shingles: set[str] | None = field(default=None, repr=False)

    def shingle_set(self) -> set[str]:
        if self._shingles is None:
            self._shingles = shingles(self.text)
        return self._shingles


def user_agent() -> str:
    return f"MakroKontextBot/3.0 (contact: {config.WIKI_CONTACT or 'unset'}) python-requests"


def get_json(url: str, params: dict[str, Any] | None = None, tries: int = 3, timeout: int = 30) -> Any | None:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": user_agent(), "Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (2 ** i))
                continue
            log.warning("HTTP %s: %s", r.status_code, url)
            return None
        except (requests.RequestException, ValueError) as e:
            log.warning("İstek hatası (%s): %s", url, str(e)[:120])
            time.sleep(2 * (2 ** i))
    return None


# --------------------------------------------------------------------------- #
# Trend sinyalleri
# --------------------------------------------------------------------------- #
_SKIP_TITLES = {"main page", "hauptseite", "wikipedia", "-", "undefined"}


def top_pageviews(project: str, limit: int | None = None, start_days_back: int = 2) -> list[dict[str, Any]]:
    """En çok görüntülenen Wikipedia maddeleri (son yayımlanan gün). [{'title','views'}]"""
    limit = limit or config.PAGEVIEW_TOP_N
    for back in range(start_days_back, start_days_back + 4):
        d = datetime.now(timezone.utc) - timedelta(days=back)
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{project}/all-access/"
               f"{d.year}/{d.month:02d}/{d.day:02d}")
        data = get_json(url, tries=2)
        try:
            arts = data["items"][0]["articles"]
        except (TypeError, KeyError, IndexError):
            continue
        out = []
        for a in arts:
            title = str(a.get("article", "")).replace("_", " ")
            if not title or ":" in title or title.lower() in _SKIP_TITLES:
                continue
            out.append({"title": title, "views": int(a.get("views", 0))})
            if len(out) >= limit:
                break
        if out:
            return out
    log.warning("Pageview verisi alınamadı: %s", project)
    return []


def _feed_entries(url: str) -> list[Any]:
    try:
        import feedparser  # tembel import

        r = requests.get(url, timeout=25, headers={"User-Agent": user_agent()})
        if r.status_code != 200:
            return []
        return list(feedparser.parse(r.content).entries)
    except Exception as e:  # ağ/ayrıştırma
        log.warning("RSS okunamadı (%s): %s", url, str(e)[:100])
        return []


def google_trends_terms(lang: config.LangConfig, limit: int = 25) -> list[str]:
    if not lang.trends_rss:
        return []
    return [e.get("title", "").strip() for e in _feed_entries(lang.trends_rss)[:limit] if e.get("title")]


def google_news_titles(lang: config.LangConfig, per_query: int = 6) -> list[str]:
    titles: list[str] = []
    for q in lang.news_queries:
        for e in _feed_entries(config.google_news_url(q, lang))[:per_query]:
            t = re.sub(r"\s+-\s+[^-]+$", "", e.get("title", "")).strip()
            if t:
                titles.append(t)
    return list(dict.fromkeys(titles))


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #
def _api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def wiki_search(lang: str, query: str, limit: int = 5) -> list[str]:
    data = get_json(_api(lang), {"action": "query", "list": "search", "srsearch": query,
                                 "srlimit": limit, "format": "json", "formatversion": 2})
    try:
        return [r["title"] for r in data["query"]["search"]]
    except (TypeError, KeyError):
        return []


def wiki_fetch(lang: str, title: str, min_chars: int = 800) -> Source | None:
    """Maddenin düz metnini getirir. Anlam ayrımı sayfaları ve kısa taslaklar elenir."""
    data = get_json(_api(lang), {
        "action": "query", "prop": "extracts|info|pageprops", "ppprop": "disambiguation",
        "explaintext": 1, "exsectionformat": "plain", "redirects": 1, "inprop": "url",
        "format": "json", "formatversion": 2, "titles": title,
    })
    try:
        page = data["query"]["pages"][0]
    except (TypeError, KeyError, IndexError):
        return None
    if page.get("missing") or "disambiguation" in (page.get("pageprops") or {}):
        return None
    text = (page.get("extract") or "").strip()
    if len(text) < min_chars:
        return None
    return Source(id="", lang=lang, title=page.get("title", title),
                  url=page.get("fullurl", f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                  text=text[: config.MAX_SOURCE_CHARS])


def resolve_source(lang: str, title: str, queries: list[str] | None = None) -> Source | None:
    """Önce tam başlık, olmazsa arama sonuçlarının en iyisi."""
    src = wiki_fetch(lang, title)
    if src:
        return src
    for q in [title, *(queries or [])]:
        for found in wiki_search(lang, q, limit=3):
            src = wiki_fetch(lang, found)
            if src:
                return src
    return None
