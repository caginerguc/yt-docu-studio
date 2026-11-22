"""Konu hafızası (data/history.json). Eski liste biçimiyle uyumludur.

Durumlar: selected -> produced (video üretildi, yüklenmedi) | uploaded | failed
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from src import config

log = logging.getLogger("makrokontext.history")
DONE_STATUSES = {"uploaded", "produced"}
# Zamanlanmış çalışmanın "bugün zaten üretildi" diye atlaması için yalnızca gerçekten yüklenmiş videolar sayılır;
# dry-run ("produced") aynı günkü gerçek yüklemeyi engellememeli.
SKIP_STATUSES = {"uploaded"}
_STOP = {"der", "die", "das", "und", "the", "and", "von", "mit", "des", "den", "ein", "eine"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w{4,}", (text or "").lower()) if t not in _STOP}


def load() -> list[dict[str, Any]]:
    path = config.HISTORY_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        bak = path.with_suffix(".json.bak")
        try:
            path.replace(bak)
        except OSError:
            pass
        log.warning("history.json okunamadı (%s); yedeklendi: %s", e, bak.name)
        return []
    if isinstance(data, dict):
        data = data.get("topics", [])
    return [d for d in data if isinstance(d, dict)]


def save(items: list[dict[str, Any]]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.HISTORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _date_of(item: dict[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(str(item.get("date", ""))[:19])
    except ValueError:
        return None


def recent(days: int | None = None, lang: str | None = None) -> list[dict[str, Any]]:
    days = days or config.HISTORY_DAYS
    cutoff = datetime.now() - timedelta(days=days)
    out = []
    for it in load():
        if it.get("status") == "failed":
            continue
        if lang and it.get("language") and it["language"].lower() != lang.lower():
            continue
        d = _date_of(it)
        if d is None or d >= cutoff:
            out.append(it)
    return out


def add(entry: dict[str, Any]) -> None:
    items = load()
    items = [i for i in items if i.get("topic_id") != entry.get("topic_id")]
    items.append(entry)
    save(items)


def set_status(topic_id: str, status: str, **extra: Any) -> None:
    items = load()
    for it in items:
        if it.get("topic_id") == topic_id:
            it["status"] = status
            it.update(extra)
    save(items)


def is_done_today(lang: str | None = None) -> bool:
    today = config.today_local_str()
    for it in load():
        status = it.get("status", "uploaded")  # eski kayıtlar yüklenmiş sayılır
        if status in SKIP_STATUSES and str(it.get("date", ""))[:10] == today:
            if not lang or not it.get("language") or it["language"].lower() == lang.lower():
                return True
    return False


def find_similar(topic_id: str, title: str, wiki_main: str, entries: list[dict[str, Any]],
                 threshold: float | None = None) -> dict[str, Any] | None:
    """Aynı konu mu? (aynı slug / aynı ana Wikipedia maddesi / yüksek kelime örtüşmesi)"""
    threshold = config.TOPIC_SIMILARITY_THRESHOLD if threshold is None else threshold
    cand = _tokens(f"{topic_id} {title}")
    for it in entries:
        if it.get("topic_id") == topic_id:
            return it
        if wiki_main and it.get("wiki_main", "").lower() == wiki_main.lower():
            return it
        other = _tokens(f"{it.get('topic_id', '')} {it.get('headline', '')} {it.get('title', '')}")
        if cand and other and len(cand & other) / len(cand | other) >= threshold:
            return it
    return None
