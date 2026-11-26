"""Ajan 1 — Baş Araştırmacı.

Akış:
  1. Sinyal topla: Wikipedia'da en çok okunan maddeler (de+en), Google Trends, keşif/arşiv haberleri.
  2. LLM, sinyali "zamansız bir hikâyeye" bağlayan aday konular önerir (koleksiyonlara göre).
  3. Kod filtreler: engel listesi (komplo/nefret anlatıları) ve geçmiş konularla benzerlik.
  4. Aday için Wikipedia kaynak dosyası (>= 3 madde) indirilir.
  5. LLM olguları çıkarır; HER olgu kaynaktan alıntıyla desteklenmek zorundadır ve kod bu alıntıyı
     kaynak metinde doğrular. Doğrulanamayan olgu atılır. Yeterli olgu yoksa sıradaki aday denenir.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src import config, history, sources as src_mod
from src.grounding import quote_supported
from src.llm import LLM

log = logging.getLogger("makrokontext.researcher")


# ----------------------------- LLM şemaları -------------------------------- #
class WikiRef(BaseModel):
    lang: str
    title: str


class Candidate(BaseModel):
    topic_id: str
    working_title: str
    collection_id: str
    trend_link: str
    evergreen_reason: str
    wiki_refs: list[WikiRef]
    search_queries: list[str]
    evidence_risk: Literal["low", "medium", "high"]
    viral_score: int


class CandidateList(BaseModel):
    candidates: list[Candidate]


class FactOut(BaseModel):
    id: str
    claim: str
    source_id: str
    evidence_quote: str
    status: Literal["belegt", "behauptet", "spekulativ"]
    year: str


class DossierOut(BaseModel):
    facts: list[FactOut]
    key_figures: list[str]
    open_questions: list[str]
    one_line_hook: str


# ----------------------------- Dosya ---------------------------------------- #
@dataclass
class Dossier:
    candidate: Candidate
    sources: list[src_mod.Source]
    facts: list[FactOut]
    key_figures: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    hook: str = ""

    def corpus(self) -> str:
        return "\n".join(s.text for s in self.sources)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or (config.CACHE_DIR / "dossier.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "candidate": self.candidate.model_dump(),
            "sources": [{"id": s.id, "lang": s.lang, "title": s.title, "url": s.url, "text": s.text}
                        for s in self.sources],
            "facts": [f.model_dump() for f in self.facts],
            "key_figures": self.key_figures, "open_questions": self.open_questions, "hook": self.hook,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Dossier":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            candidate=Candidate.model_validate(d["candidate"]),
            sources=[src_mod.Source(**s) for s in d["sources"]],
            facts=[FactOut.model_validate(f) for f in d["facts"]],
            key_figures=d.get("key_figures", []), open_questions=d.get("open_questions", []),
            hook=d.get("hook", ""),
        )


# ----------------------------- Ajan ----------------------------------------- #
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RISK_RANK = {"low": 0, "medium": 1, "high": 2}

SELECT_SYSTEM = f"""You are the chief researcher of {config.CHANNEL_NAME}, a documentary channel for EVERGREEN stories:
hidden history, forgotten scientists, documented secret operations, historic artifacts, financial intrigues.
Today's trend signals are only a springboard: you connect them to a timeless, well-documented story that will
still be watched in five years. You never pick breaking news, never pick conspiracy myths, and never pick
anything whose core claim is not backed by published scholarship or archives. Mystery must come from real,
documented gaps and contradictions, not from invention."""


class ResearchAgent:
    def __init__(self, llm: LLM | None = None, lang_code: str | None = None):
        self.llm = llm or LLM()
        self.lang = config.get_lang(lang_code)

    # -- 1. sinyaller ------------------------------------------------------- #
    def collect_signals(self) -> dict[str, list]:
        wiki: list[dict] = []
        for project in config.WIKI_PAGEVIEW_PROJECTS:
            for item in src_mod.top_pageviews(project):
                wiki.append({**item, "project": project})
        return {
            "wiki": wiki,
            "trends": src_mod.google_trends_terms(self.lang),
            "news": src_mod.google_news_titles(self.lang),
        }

    # -- 2/3. aday seçimi ---------------------------------------------------- #
    @staticmethod
    def _blocked(text: str) -> str | None:
        low = text.lower()
        return next((b for b in config.TOPIC_BLOCKLIST if b in low), None)

    def _validate_candidates(self, obj: CandidateList) -> None:
        valid_ids = {c["id"] for c in config.COLLECTIONS}
        problems = []
        if not obj.candidates:
            raise ValueError("candidates list is empty")
        for c in obj.candidates:
            if not _SLUG.match(c.topic_id):
                problems.append(f"{c.topic_id!r}: topic_id must be a lowercase-hyphen slug")
            if c.collection_id not in valid_ids:
                problems.append(f"{c.topic_id}: collection_id must be one of {sorted(valid_ids)}")
            if len(c.wiki_refs) < 3:
                problems.append(f"{c.topic_id}: provide at least 3 wiki_refs (main article + related)")
        if problems:
            raise ValueError("; ".join(problems[:6]))

    def shortlist(self, signals: dict[str, list] | None = None) -> list[Candidate]:
        signals = signals or self.collect_signals()
        recent = history.recent(lang=self.lang.code)
        past = "\n".join(f"- {h.get('topic_id', '')}: {h.get('headline') or h.get('title', '')}" for h in recent[-80:]) \
            or "(none yet)"
        wiki_lines = "\n".join(f"{i + 1}. {w['title']} ({w['project'][:2]}, {w['views']:,} views)"
                               for i, w in enumerate(signals["wiki"][:120])) or "(unavailable)"
        collections = "\n".join(f"- {c['id']}: {c['name']} — {c['brief']}" for c in config.COLLECTIONS)

        prompt = f"""OUTPUT LANGUAGE for titles/reasons: {self.lang.name}. Wikipedia titles must be exact, real article titles.

COLLECTIONS:
{collections}

TREND SIGNALS
Most-read Wikipedia articles yesterday:
{wiki_lines}
Search trends: {', '.join(signals['trends'][:25]) or '(unavailable)'}
Archive/discovery headlines this week:
{chr(10).join('- ' + t for t in signals['news'][:30]) or '(unavailable)'}

ALREADY COVERED (never repeat or paraphrase these):
{past}

TASK: propose 6 candidate documentaries (8-11 minutes each). For each: a trend_link explaining which signal
inspired it (a signal may be only loosely related; if no signal fits, say 'evergreen pick'), why it stays
relevant for years, 3-5 wiki_refs (lang 'de' or 'en', exact titles: main article first, then related
articles/people/events), 2-3 search_queries, evidence_risk (low = rich documentation), viral_score 1-100
(curiosity + search demand). Spread candidates across collections. topic_id: lowercase-hyphen slug."""

        res = self.llm.generate_json(prompt, CandidateList, system=SELECT_SYSTEM, temperature=0.8,
                                     validate=self._validate_candidates, label="shortlist")
        kept: list[Candidate] = []
        for c in res.candidates:
            blob = " ".join([c.topic_id, c.working_title, c.trend_link, c.evergreen_reason,
                             *(r.title for r in c.wiki_refs)])
            hit = self._blocked(blob)
            if hit:
                log.warning("Aday elendi (engel listesi: %s): %s", hit, c.topic_id)
                continue
            dup = history.find_similar(c.topic_id, c.working_title, c.wiki_refs[0].title, recent)
            if dup:
                log.info("Aday elendi (geçmişte benzeri var: %s): %s", dup.get("topic_id"), c.topic_id)
                continue
            kept.append(c)
        kept.sort(key=lambda c: (_RISK_RANK[c.evidence_risk], -c.viral_score))
        log.info("Kısa liste: %s", [c.topic_id for c in kept])
        return kept

    # -- 4/5. kaynak dosyası ve olgular ------------------------------------- #
    def _gather_sources(self, cand: Candidate) -> list[src_mod.Source]:
        found: list[src_mod.Source] = []
        seen: set[str] = set()

        def add(s: src_mod.Source | None) -> None:
            if s and s.url not in seen and len(found) < 6:
                seen.add(s.url)
                found.append(s)

        for ref in cand.wiki_refs:
            add(src_mod.resolve_source(ref.lang, ref.title, cand.search_queries))
        if len(found) < config.MIN_SOURCES_PER_VIDEO:  # takviye: arama sorguları
            for q in cand.search_queries:
                for lang in dict.fromkeys([self.lang.code, "en"]):
                    for title in src_mod.wiki_search(lang, q, limit=2):
                        add(src_mod.wiki_fetch(lang, title))
        for i, s in enumerate(found, 1):
            s.id = f"S{i}"
        return found

    def _extract_facts(self, cand: Candidate, srcs: list[src_mod.Source]) -> DossierOut:
        blocks = "\n\n".join(f"=== SOURCE {s.id} | {s.title} ({s.lang}.wikipedia) ===\n{s.text}" for s in srcs)
        prompt = f"""TOPIC: {cand.working_title}
OUTPUT LANGUAGE for claim/hook/questions: {self.lang.name} (evidence_quote stays in the source language).

{blocks}

TASK: extract 14-30 distinct, narratively useful facts about the topic, strictly from the sources above.
Rules:
- id: F1, F2, ... ; source_id: the SOURCE id the fact comes from.
- evidence_quote: an EXACT verbatim excerpt (8-30 words) copied from that source that supports the claim.
  Never paraphrase the quote; it will be machine-checked against the source text.
- status: belegt = the source states it as established fact; behauptet = the source attributes it to someone
  or calls it disputed; spekulativ = the source says possible/unclear/hypothesis.
- year: the 4-digit year if the fact is dated, else an empty string.
- Prefer surprising, concrete, checkable details (names, dates, places, numbers, documents) and contradictions.
- key_figures: main people/institutions; open_questions: genuine unresolved questions found in the sources;
  one_line_hook: the most startling documented moment, as one sentence."""
        return self.llm.generate_json(prompt, DossierOut, temperature=0.2, label="facts")

    def build_dossier(self, cand: Candidate) -> Dossier | None:
        srcs = self._gather_sources(cand)
        if len(srcs) < config.MIN_SOURCES_PER_VIDEO:
            log.warning("%s: yalnızca %d kaynak bulundu, atlanıyor.", cand.topic_id, len(srcs))
            return None
        out = self._extract_facts(cand, srcs)
        by_id = {s.id: s for s in srcs}
        verified: list[FactOut] = []
        for f in out.facts:
            s = by_id.get(f.source_id)
            if s and quote_supported(f.evidence_quote, s.shingle_set()):
                verified.append(f)
            else:
                log.info("Olgu atıldı (alıntı kaynakta doğrulanamadı): %s", f.id)
        log.info("%s: %d/%d olgu doğrulandı.", cand.topic_id, len(verified), len(out.facts))
        if len(verified) < config.MIN_FACTS_PER_VIDEO:
            return None
        for i, f in enumerate(verified, 1):  # kimlikleri art arda yeniden numaralandır
            f.id = f"F{i}"
        return Dossier(cand, srcs, verified, out.key_figures, out.open_questions, out.one_line_hook)

    # -- ana akış ------------------------------------------------------------ #
    def run(self) -> Dossier:
        for cand in self.shortlist()[: config.MAX_CANDIDATES_TRIED]:
            log.info("Aday deneniyor: %s", cand.topic_id)
            dossier = self.build_dossier(cand)
            if dossier:
                history.add({
                    "topic_id": cand.topic_id, "headline": cand.working_title,
                    "wiki_main": dossier.sources[0].title, "collection": cand.collection_id,
                    "date": config.now_local().isoformat(timespec="seconds"),
                    "language": self.lang.code, "status": "selected",
                })
                dossier.save()
                return dossier
        raise RuntimeError("Hiçbir aday yeterli doğrulanmış kaynak/olguya ulaşamadı.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Araştırmacıyı tek başına çalıştır")
    ap.add_argument("--lang", default=config.DEFAULT_LANG)
    a = ap.parse_args()
    config.ensure_dirs()
    config.setup_logging()
    d = ResearchAgent(lang_code=a.lang).run()
    print(f"{d.candidate.topic_id}: {len(d.sources)} kaynak, {len(d.facts)} doğrulanmış olgu -> {config.CACHE_DIR / 'dossier.json'}")
