"""Uydurmayı (hallucination) kod seviyesinde yakalayan doğrulama araçları.

  * quote_supported : modelin verdiği alıntı gerçekten kaynak metinde var mı?
  * numbers_missing : cümledeki yıl/sayı kaynak metinlerin hiçbirinde geçmiyor mu?
Bunlar LLM'e güvenmeden çalışır; LLM'in "doğruladım" demesi tek başına yeterli sayılmaz.
"""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_NUM = re.compile(r"\d[\d.,\u00a0\u202f']*\d|\d")


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_PUNCT.sub(" ", t).split())


def shingles(text: str, n: int = 4) -> set[str]:
    words = norm(text).split()
    return {" ".join(words[i: i + n]) for i in range(max(0, len(words) - n + 1))}


def quote_supported(quote: str, source_shingles: set[str], n: int = 4, min_cov: float = 0.75) -> bool:
    """Alıntının 4'lü kelime kümelerinin en az %75'i kaynakta varsa kabul (küçük biçim farkları tolere edilir)."""
    words = norm(quote).split()
    if len(words) < n + 1:
        return False
    grams = [" ".join(words[i: i + n]) for i in range(len(words) - n + 1)]
    hits = sum(1 for g in grams if g in source_shingles)
    return hits / len(grams) >= min_cov


def numbers_in(text: str, min_digits: int = 3) -> set[str]:
    out: set[str] = set()
    for m in _NUM.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= min_digits:
            out.add(digits)
    return out


def numbers_missing(sentence: str, corpus_numbers: set[str]) -> list[str]:
    """Cümlede geçen ama kaynaklarda hiç bulunmayan yıl/sayılar (en az 3 haneli)."""
    return sorted(numbers_in(sentence) - corpus_numbers)
