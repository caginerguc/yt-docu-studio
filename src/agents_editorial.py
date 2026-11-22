"""Ajan 2-5 — Yazar ekibi.

  Mimar        : 5 perdelik anlatı planı (video başına değişen anlatı yapısı, şablon izlenimini kırar)
  Başyazar     : perde perde yazar; her cümle kanıt etiketi (belegt/behauptet/spekulativ/ueberleitung) taşır
  Eleştirmen   : acımasız puanlama, zayıf perdeleri geri gönderir (en fazla MAX_REVISIONS tur)
  Doğrulayıcı  : kod (sayı/yıl kaynakta var mı) + LLM; desteklenmeyen cümle düzeltilir veya silinir
  Sinematograf : cümle başına kesit planı (stok video / arşiv fotoğrafı / AI görsel)
  SEO          : başlık, kapak metni, etiketler, kaynak listesi
"""
from __future__ import annotations

import json
import logging
import zlib
from typing import Literal

from pydantic import BaseModel

from src import config
from src.agent_researcher import Dossier, ResearchAgent
from src.grounding import numbers_in, numbers_missing
from src.llm import LLM
from src.plan import PlanAct, PlanSentence, PlanShot, ProductionPlan, SeoPlan, SourceRef

log = logging.getLogger("makrokontext.editorial")

Evidence = Literal["belegt", "behauptet", "spekulativ", "ueberleitung"]
Pause = Literal["none", "short", "long"]
Camera = Literal["zoom_in", "zoom_out", "pan_left_to_right", "pan_right_to_left"]
_RANK = {"belegt": 2, "behauptet": 1, "spekulativ": 0}

STRUCTURE_VARIANTS = {
    "ermittlung": "INVESTIGATION: the narrator follows the evidence trail like a detective; every act answers one "
                  "question and opens a bigger one.",
    "zeitsprung": "TWO TIMELINES: alternate between the historical moment and its later rediscovery, converging "
                  "at the turning point.",
    "gegenspieler": "RIVAL VIEWPOINTS: tell the central conflict through two opposing figures; the twist reveals what "
                    "each of them did not know.",
    "rueckwaerts": "REVERSE REVEAL: start with the consequence, then step backwards through the causes to the origin.",
}


def pick_variant(topic_id: str) -> str:
    keys = sorted(STRUCTURE_VARIANTS)
    return keys[zlib.crc32(topic_id.encode()) % len(keys)]


# ----------------------------- LLM şemaları -------------------------------- #
class ArcAct(BaseModel):
    act_id: int
    beats: list[str]
    fact_ids: list[str]
    closing_question: str


class ArcPlan(BaseModel):
    working_title: str
    twist_fact_id: str
    acts: list[ArcAct]


class SentenceOut(BaseModel):
    text: str
    evidence: Evidence
    fact_ids: list[str]
    pause_before: Pause


class ActDraft(BaseModel):
    sentences: list[SentenceOut]


class Issue(BaseModel):
    sentence_id: str
    problem: str
    fix: str


class Critique(BaseModel):
    hook_score: int
    pacing_score: int
    clarity_score: int
    originality_score: int
    overall: int
    acts_to_rewrite: list[int]
    issues: list[Issue]


class FactFlag(BaseModel):
    sentence_id: str
    verdict: Literal["unsupported", "overstated", "contradicted"]
    reason: str
    replacement_text: str
    replacement_evidence: Evidence


class FactCheckResult(BaseModel):
    flags: list[FactFlag]


class ShotOut(BaseModel):
    source: Literal["stock_video", "archive_photo", "ai_image"]
    stock_query_en: str
    archive_query_en: str
    ai_prompt_en: str
    camera: Camera


class SentenceShots(BaseModel):
    sentence_id: str
    shots: list[ShotOut]


class ActShots(BaseModel):
    sentences: list[SentenceShots]


class SeoOut(BaseModel):
    title: str
    description_intro: str
    chapter_titles: list[str]
    tags: list[str]
    hashtags: list[str]
    thumbnail_text: str
    thumbnail_tag: str
    thumbnail_prompt_en: str
    music_mood: Literal["cinematic_tension", "serious_news", "ambient_tech"]
    shorts_hook: str


# ----------------------------- Yardımcılar --------------------------------- #
def facts_block(d: Dossier, with_quotes: bool = False) -> str:
    lines = []
    for f in d.facts:
        line = f"{f.id} | {f.status} | {f.year or '-'} | {f.claim}"
        if with_quotes:
            line += f'  [{f.source_id}: "{f.evidence_quote}"]'
        lines.append(line)
    return "\n".join(lines)


def script_view(acts: list[PlanAct]) -> str:
    out = []
    for a in acts:
        out.append(f"## ACT {a.act_id}: {a.name}")
        for s in a.sentences:
            out.append(f"{s.id} [{s.evidence}|{','.join(s.fact_ids) or '-'}|{s.pause_before}] {s.text}")
    return "\n".join(out)


def renumber(acts: list[PlanAct]) -> None:
    for a in acts:
        for i, s in enumerate(a.sentences):
            s.id, s.act_id = f"A{a.act_id}-{i:02d}", a.act_id


def clean_sentence(text: str) -> str:
    t = " ".join(text.replace("\n", " ").split()).strip(' "«»„“”')
    return t if t and t[-1] in ".!?…" else t + "."


def shots_for(sentence: PlanSentence) -> int:
    secs = sentence.words * 60 / config.WORDS_PER_MINUTE
    return max(1, min(4, round(secs / config.SHOT_TARGET_S)))


NARRATION_RULES = f"""NARRATION RULES
- Spoken documentary voice: calm, precise, quietly gripping. Short, strong sentences (8-25 words, never more than
  {config.MAX_SENTENCE_WORDS}). Occasionally one very short sentence after a long one.
- Written for text-to-speech: no abbreviations (write words out), no parentheses, no markdown, no stage directions,
  no emojis, no lists.
- Every factual statement must come from the FACTS list. Never invent documents, quotes, dates, names or numbers.
- Label every sentence with evidence: belegt (documented fact), behauptet (a claim; attribute it explicitly to a
  person or source inside the sentence), spekulativ (open question or hypothesis; say clearly it is unresolved),
  ueberleitung (bridge or atmosphere with NO factual claim). A sentence can never be labelled stronger than its
  weakest fact (a fact marked 'behauptet' can only appear in a 'behauptet' or 'spekulativ' sentence).
- Mystery comes from real gaps, contradictions and unanswered questions in the record. Never insinuate a
  conspiracy without evidence. Never demonize groups of people.
- No filler ('in this video', 'let's dive in'), no calls to like/subscribe, no clichés.
- pause_before: 'long' only for 1-2 genuine turning points per act, 'short' for dramatic beats, else 'none'."""


# ----------------------------- Ekip ----------------------------------------- #
class EditorialCrew:
    def __init__(self, llm: LLM | None = None, lang_code: str | None = None):
        self.llm = llm or LLM()
        self.lang = config.get_lang(lang_code)
        self.system = ("You are the writing room of MakroKontext, a documentary channel in the tradition of Arte, "
                       f"Lemmino and Fern. Output language: {self.lang.name}. Accuracy beats drama.")

    # -- 1. mimari ----------------------------------------------------------- #
    def plan_arc(self, d: Dossier, variant: str, target: int) -> ArcPlan:
        ids = {f.id for f in d.facts}
        acts_txt = "\n".join(f"Act {a['id']} '{a['name']}' (~{round(target * a['share'])} words): {a['goal']}"
                             for a in config.ACTS)

        def validate(p: ArcPlan) -> None:
            if sorted(a.act_id for a in p.acts) != [1, 2, 3, 4, 5]:
                raise ValueError("acts must contain exactly act_id 1,2,3,4,5")
            if p.twist_fact_id not in ids:
                raise ValueError(f"twist_fact_id must be one of {sorted(ids)}")
            for a in p.acts:
                bad = [f for f in a.fact_ids if f not in ids]
                if bad or not (3 <= len(a.beats) <= 6) or not a.fact_ids:
                    raise ValueError(f"act {a.act_id}: need 3-6 beats and valid fact_ids (bad: {bad})")

        prompt = f"""TOPIC: {d.candidate.working_title}
NARRATIVE DEVICE FOR THIS VIDEO: {STRUCTURE_VARIANTS[variant]}
HOOK CANDIDATE: {d.hook}
OPEN QUESTIONS IN THE RECORD: {'; '.join(d.open_questions) or '-'}

FACTS (id | status | year | claim):
{facts_block(d)}

FIVE-ACT STRUCTURE:
{acts_txt}

TASK: design the arc. For every act give 3-6 beats (one line each, in {self.lang.name}), the fact_ids it will use
(spread the facts over the acts; every strong fact should be used once), and a closing_question that opens the
next act. twist_fact_id = the single fact that changes how the viewer understands everything (Act 3).
working_title: a precise, intriguing working title."""
        return self.llm.generate_json(prompt, ArcPlan, system=self.system, temperature=0.6,
                                      validate=None, label="arc")

    # -- 2. yazım ------------------------------------------------------------- #
    def write_act(self, act_cfg: dict, arc_act: ArcAct, d: Dossier, budget: int, tail: str,
                  used: set[str], variant: str, feedback: str = "",
                  previous: list[PlanSentence] | None = None) -> list[PlanSentence]:
        status = {f.id: f.status for f in d.facts}

        def validate(draft: ActDraft) -> None:
            s = draft.sentences
            if not s:
                raise ValueError("no sentences")
            words = sum(len(x.text.split()) for x in s)
            lo, hi = int(budget * 0.7), int(budget * 1.35)
            if not lo <= words <= hi:
                log.warning(f"act has {words} words; required {lo}-{hi} (target {budget})")
            errs, longs, bridges = [], 0, 0
            for i, x in enumerate(s):
                n = len(x.text.split())
                if n > config.MAX_SENTENCE_WORDS:
                    errs.append(f"sentence {i} has {n} words (max {config.MAX_SENTENCE_WORDS})")
                unknown = [f for f in x.fact_ids if f not in status]
                if unknown:
                    errs.append(f"sentence {i}: unknown fact ids {unknown}")
                    continue
                if x.evidence == "ueberleitung":
                    bridges += 1
                    continue
                if not x.fact_ids:
                    errs.append(f"sentence {i}: evidence '{x.evidence}' needs fact_ids")
                    continue
                weakest = min(_RANK[status[f]] for f in x.fact_ids)
                if _RANK[x.evidence] > weakest:
                    errs.append(f"sentence {i}: label '{x.evidence}' is stronger than its facts allow")
                longs += x.pause_before == "long"
            if longs > 2:
                errs.append("at most 2 'long' pauses per act")
            if bridges > 0.25 * len(s):
                errs.append("too many ueberleitung sentences (max 25%)")
            if errs:
                raise ValueError("; ".join(errs[:6]))

        use = [f for f in d.facts if f.id in set(arc_act.fact_ids)]
        reserve = [f for f in d.facts if f.id not in set(arc_act.fact_ids)]
        extra = ""
        if act_cfg["id"] == 1:
            extra = ("Open mid-scene or with the single most startling documented fact; pose the central question "
                     "within the first two sentences.")
        if act_cfg["id"] == 5:
            extra = "End on an unresolved, thought-provoking final sentence. No moralizing summary."
        prior = ""
        if previous:
            prior = "\nPREVIOUS DRAFT OF THIS ACT (improve it, do not repeat its weaknesses):\n" + \
                    "\n".join(f"{s.id}: {s.text}" for s in previous)
        prompt = f"""NARRATIVE DEVICE: {STRUCTURE_VARIANTS[variant]}
{NARRATION_RULES}

WRITE ACT {act_cfg['id']}: '{act_cfg['name']}' — {act_cfg['goal']} {extra}
WORD BUDGET: {budget} words (acceptable {int(budget * 0.7)}-{int(budget * 1.35)}).
BEATS TO COVER:
{chr(10).join('- ' + b for b in arc_act.beats)}
END THE ACT BY OPENING THIS QUESTION: {arc_act.closing_question}

FACTS FOR THIS ACT (id | status | year | claim):
{facts_block(Dossier(d.candidate, d.sources, use))}
RESERVE FACTS (use sparingly, only if needed):
{facts_block(Dossier(d.candidate, d.sources, reserve)) or '-'}
ALREADY USED IN EARLIER ACTS (avoid repeating): {', '.join(sorted(used)) or '-'}
LAST SENTENCES OF THE PREVIOUS ACT (continue smoothly, do not recap): {tail or '(this is the opening act)'}
{('REVIEWER FEEDBACK TO FIX:' + chr(10) + feedback) if feedback else ''}{prior}

Return the sentences of this act only."""
        draft = self.llm.generate_json(prompt, ActDraft, system=self.system, temperature=0.75,
                                       validate=None, label=f"act{act_cfg['id']}")
        return [PlanSentence(id="", act_id=act_cfg["id"], text=clean_sentence(x.text), evidence=x.evidence,
                             fact_ids=x.fact_ids, pause_before=x.pause_before) for x in draft.sentences]

    # -- 3. eleştirmen -------------------------------------------------------- #
    def critique(self, acts: list[PlanAct], d: Dossier) -> Critique:
        ids = {s.id for s in (x for a in acts for x in a.sentences)}

        def validate(c: Critique) -> None:
            if any(a not in range(1, 6) for a in c.acts_to_rewrite):
                raise ValueError("acts_to_rewrite must contain act ids 1-5")
            bad = [i.sentence_id for i in c.issues if i.sentence_id not in ids]
            if bad:
                raise ValueError(f"unknown sentence ids in issues: {bad[:5]}")

        prompt = f"""You are the merciless chief editor. Judge this documentary script (voice-over, {self.lang.name}).

{script_view(acts)}

SCORE 1-10 (be strict; 8 means broadcast quality): hook_score (do the first 30 seconds create a question the
viewer must have answered?), pacing_score (rhythm, variety, no sag in the middle), clarity_score (easy to follow
by ear, no jargon, no clutter), originality_score (fresh angle, no cliché, no generic AI phrasing), overall.
List concrete issues (sentence_id, problem, fix). acts_to_rewrite: only acts scoring below 8 in your judgment
(at most 3, worst first); empty list if the script is good."""
        return self.llm.generate_json(prompt, Critique, system=self.system, temperature=0.3,
                                      validate=None, label="critique")

    # -- 4. doğrulayıcı ------------------------------------------------------- #
    def fact_check(self, acts: list[PlanAct], d: Dossier) -> dict[str, int]:
        corpus_nums = numbers_in(d.corpus(), min_digits=1)
        by_id = {s.id: s for a in acts for s in a.sentences}
        det = {sid: numbers_missing(s.text, corpus_nums) for sid, s in by_id.items()}
        det = {k: v for k, v in det.items() if v}

        def validate(r: FactCheckResult) -> None:
            bad = [f.sentence_id for f in r.flags if f.sentence_id not in by_id]
            if bad:
                raise ValueError(f"unknown sentence ids: {bad[:5]}")

        det_txt = "\n".join(f"{k}: numbers not found in any source: {v}" for k, v in det.items()) or "(none)"
        prompt = f"""You are the fact-checker. Check the voice-over script against the verified facts.

FACTS (id | status | year | claim [source: exact quote]):
{facts_block(d, with_quotes=True)}

SCRIPT:
{script_view(acts)}

AUTOMATIC FINDINGS (must be addressed; the numbers/years below appear in NO source):
{det_txt}

Flag a sentence only if it (a) states something the facts do not support, (b) sounds more certain than the fact's
status allows, (c) contradicts a fact, or (d) contains a number/year not in the sources.
For each flag give replacement_text: same language, same narrative function, at most 25 words, fixed by hedging,
attributing or removing the unsupported detail; replacement_evidence: the correct label.
Return an empty flags list if everything is supported."""
        res = self.llm.generate_json(prompt, FactCheckResult, system=self.system, temperature=0.1,
                                     validate=None, label="factcheck")
        stats = {"flagged": len(set(det) | {f.sentence_id for f in res.flags}), "replaced": 0, "deleted": 0}
        unfixed: set[str] = set()
        for f in res.flags:
            s = by_id[f.sentence_id]
            new = clean_sentence(f.replacement_text) if f.replacement_text.strip() else ""
            if new and len(new.split()) <= config.MAX_SENTENCE_WORDS and not numbers_missing(new, corpus_nums):
                s.text, s.evidence = new, f.replacement_evidence
                stats["replaced"] += 1
            else:
                unfixed.add(s.id)
        # düzeltilemeyen veya hâlâ kaynaksız sayı içeren cümleleri sil (perde başına en az 3 cümle kalsın)
        for a in acts:
            for s in list(a.sentences):
                if (s.id in unfixed or numbers_missing(s.text, corpus_nums)) and len(a.sentences) > 3:
                    a.sentences.remove(s)
                    stats["deleted"] += 1
        renumber(acts)
        log.info("Doğrulama: %s", stats)
        return stats

    # -- 5. sinematograf ------------------------------------------------------ #
    def direct_act(self, act: PlanAct, d: Dossier) -> None:
        want = {s.id: shots_for(s) for s in act.sentences}
        ids = set(want)

        def validate(r: ActShots) -> None:
            got = {x.sentence_id for x in r.sentences}
            if got != ids:
                raise ValueError(f"sentence_id mismatch; missing={sorted(ids - got)[:4]} unknown={sorted(got - ids)[:4]}")

        lines = "\n".join(f"{s.id} (needs {want[s.id]} shot(s)): {s.text}" for s in act.sentences)
        mix = config.SOURCE_MIX
        prompt = f"""You are the cinematographer of a dark, elegant historical documentary. Plan the shots for Act
{act.act_id} '{act.name}' of '{d.candidate.working_title}'.

SENTENCES:
{lines}

For each sentence return EXACTLY the number of shots it needs. Each shot has a source:
- stock_video (~{int(mix['stock_video'] * 100)}%): generic filmable footage from a free stock library. stock_query_en = 2-4 concrete English
  words (e.g. 'candle flame dark room', 'old library shelves', 'rain city night', 'typewriter close up').
  Never ask for named people, brands, or specific historic events.
- archive_photo (~{int(mix['archive_photo'] * 100)}%): ONLY when the sentence names a real person, object, building, document or place that
  likely has a public-domain historical photo. archive_query_en = the precise subject (e.g. 'Nikola Tesla portrait').
- ai_image (~{int(mix['ai_image'] * 100)}%): atmospheric reconstruction. ai_prompt_en = one vivid scene description (composition, light,
  subject). Rules: never show a real, named person's face (use silhouettes, hands, back views, objects,
  architecture, landscapes); no readable text, no logos; do NOT add style words (a house style is appended later).
Unused query fields must be empty strings. camera: vary the movement; match the mood (zoom_in for tension,
pan for reveals). Alternate source types so no more than 2 consecutive shots share a source."""
        res = self.llm.generate_json(prompt, ActShots, system=self.system, temperature=0.6,
                                     validate=None, label=f"shots-act{act.act_id}")
        by_id = {x.sentence_id: x.shots for x in res.sentences}
        for s in act.sentences:
            shots = list(by_id[s.id])
            while len(shots) < want[s.id]:       # eksikse son kesiti farklı kamerayla çoğalt
                dup = shots[-1].model_copy()
                dup.camera = config.CAMERA_EFFECTS[(config.CAMERA_EFFECTS.index(dup.camera) + 1) % 4]
                shots.append(dup)
            s.shots = [PlanShot(source=x.source, stock_query_en=x.stock_query_en.strip(),
                                archive_query_en=x.archive_query_en.strip(), ai_prompt_en=x.ai_prompt_en.strip(),
                                camera=x.camera) for x in shots[: want[s.id]]]

    @staticmethod
    def finalize_shots(acts: list[PlanAct]) -> dict[str, float]:
        prev = ""
        counts = {"stock_video": 0, "archive_photo": 0, "ai_image": 0}
        for a in acts:
            first = True
            for s in a.sentences:
                for sh in s.shots:
                    if sh.camera == prev:   # arka arkaya aynı kamera hareketi olmasın
                        sh.camera = config.CAMERA_EFFECTS[(config.CAMERA_EFFECTS.index(sh.camera) + 1) % 4]
                    prev = sh.camera
                    sh.transition = "crossfade" if (first and a.act_id > 1) else "cut"
                    first = False
                    counts[sh.source] += 1
                    # alanlar kaynakla uyumlu: boş sorgu varsa AI'ye düş
                    if sh.source == "stock_video" and not sh.stock_query_en:
                        sh.source = "ai_image"
                    if sh.source == "archive_photo" and not sh.archive_query_en:
                        sh.source = "ai_image"
                    if sh.source == "ai_image" and not sh.ai_prompt_en:
                        sh.ai_prompt_en = sh.stock_query_en or sh.archive_query_en or "empty dark archive room"
        total = max(1, sum(counts.values()))
        return {k: round(v / total, 2) for k, v in counts.items()}

    # -- 6. SEO --------------------------------------------------------------- #
    def build_seo(self, acts: list[PlanAct], d: Dossier, arc: ArcPlan, variant: str) -> SeoOut:
        def validate(o: SeoOut) -> None:
            errs = []
            if not 20 <= len(o.title) <= 95:
                errs.append("title must be 20-95 characters")
            if len(o.chapter_titles) != 5:
                errs.append("chapter_titles needs exactly 5 entries")
            if len(o.tags) < 8:
                errs.append("at least 8 tags")
            if not 1 <= len(o.thumbnail_text.split()) <= 4 or len(o.thumbnail_text) > 26:
                errs.append("thumbnail_text: 1-4 words, max 26 characters")
            hit = next((b for b in config.TOPIC_BLOCKLIST if b in f"{o.title} {o.thumbnail_text}".lower()), None)
            if hit:
                errs.append(f"forbidden term: {hit}")
            if errs:
                raise ValueError("; ".join(errs))

        opening = " ".join(a.sentences[0].text for a in acts if a.sentences)
        prompt = f"""You are the YouTube packaging editor. Output language: {self.lang.name}.
TOPIC: {arc.working_title}
OPENING LINES OF EACH ACT: {opening}
TWIST FACT: {next((f.claim for f in d.facts if f.id == arc.twist_fact_id), '')}

TASK
- title: curiosity-driven but FULLY answerable inside the video; no false promise, no ALL-CAPS, no 'SCHOCK' bait;
  max 95 characters. Pattern: concrete subject + open question/tension.
- description_intro: 2-3 sentences, no timestamps, no hashtags.
- chapter_titles: exactly 5 short chapter names (one per act, spoiler-free).
- tags: 12-15 niche tags; hashtags: exactly 3 (with #).
- thumbnail_text: 1-3 striking words that create a question (e.g. 'VERBOTENE AKTE', 'WER WAR ER?'), uppercase.
- thumbnail_tag: a short truthful label for the corner banner (e.g. 'AKTE 1943', 'UNGELÖST'); no invented claims.
- thumbnail_prompt_en: ONE dramatic, high-contrast, dark background image: a single strong subject lit from the
  side, deep shadows, empty space on the left for text. No real person's face (silhouette, hands, object, door,
  document stack). No text in the image.
- music_mood: choose the best fit. shorts_hook: one-line cliffhanger for a 45-second Short."""
        o = self.llm.generate_json(prompt, SeoOut, system=self.system, temperature=0.7,
                                   validate=None, label="seo")
        if "no text" not in o.thumbnail_prompt_en.lower():
            o.thumbnail_prompt_en += ", no text, no watermark"
        return o

    # -- ana akış -------------------------------------------------------------- #
    def produce(self, d: Dossier) -> ProductionPlan:
        cand = d.candidate
        variant = pick_variant(cand.topic_id)
        target = max(config.TARGET_WORDS[0], min(config.TARGET_WORDS[1], len(d.facts) * 55))
        log.info("Üretim: %s | yapı=%s | hedef=%d kelime", cand.topic_id, variant, target)

        arc = self.plan_arc(d, variant, target)
        arc_by_id = {a.act_id: a for a in arc.acts}
        acts: list[PlanAct] = []
        used: set[str] = set()
        tail = ""
        for cfg in config.ACTS:
            sents = self.write_act(cfg, arc_by_id[cfg["id"]], d, round(target * cfg["share"]), tail, used, variant)
            acts.append(PlanAct(act_id=cfg["id"], name=cfg["name"], sentences=sents))
            renumber(acts)
            used |= {f for s in sents for f in s.fact_ids}
            tail = " ".join(s.text for s in sents[-3:])

        scores: dict = {}
        for rnd in range(config.MAX_REVISIONS):
            crit = self.critique(acts, d)
            scores = crit.model_dump(exclude={"issues"})
            log.info("Eleştirmen turu %d: genel=%d, yeniden yazılacak=%s", rnd + 1, crit.overall, crit.acts_to_rewrite)
            if crit.overall >= 8 or not crit.acts_to_rewrite:
                break
            for act_id in crit.acts_to_rewrite[:3]:
                act = next(a for a in acts if a.act_id == act_id)
                fb = "\n".join(f"- {i.sentence_id}: {i.problem} -> {i.fix}" for i in crit.issues
                               if i.sentence_id.startswith(f"A{act_id}-"))
                cfg = next(c for c in config.ACTS if c["id"] == act_id)
                prior_used = {f for a in acts if a.act_id < act_id for s in a.sentences for f in s.fact_ids}
                idx = acts.index(act)
                tail_prev = " ".join(s.text for s in acts[idx - 1].sentences[-3:]) if idx else ""
                act.sentences = self.write_act(cfg, arc_by_id[act_id], d, round(target * cfg["share"]), tail_prev,
                                               prior_used, variant, feedback=fb or (crit.issues[0].fix if crit.issues else ""),
                                               previous=act.sentences)
                renumber(acts)

        stats = self.fact_check(acts, d)
        for a in acts:
            self.direct_act(a, d)
        mix = self.finalize_shots(acts)
        log.info("Kesit karışımı: %s (hedef %s)", mix, config.SOURCE_MIX)
        seo = self.build_seo(acts, d, arc, variant)
        for a, title in zip(acts, seo.chapter_titles):
            a.chapter_title = title

        plan = ProductionPlan(
            topic_id=cand.topic_id, language=self.lang.code, collection_id=cand.collection_id,
            working_title=arc.working_title, structure_variant=variant, trend_link=cand.trend_link,
            target_words=target,
            sources=[SourceRef(id=s.id, title=s.title, url=s.url, lang=s.lang) for s in d.sources],
            acts=acts,
            seo=SeoPlan(title=seo.title, description_intro=seo.description_intro, tags=seo.tags[:15],
                        hashtags=seo.hashtags[:3], thumbnail_text=seo.thumbnail_text,
                        thumbnail_tag=seo.thumbnail_tag, thumbnail_prompt_en=seo.thumbnail_prompt_en,
                        music_mood=seo.music_mood, shorts_hook=seo.shorts_hook),
            critique={**scores, "factcheck": stats, "source_mix": mix},
        )
        plan.recompute()
        log.info("Plan hazır: %d kelime, ~%.1f dk, %d kesit, %d cümle", plan.word_count, plan.est_minutes,
                 plan.shot_count(), len(plan.sentences()))
        return plan


# ----------------------------- Açıklama metni -------------------------------- #
_DISCLOSURE = {
    "de": "Hinweis: Recherche, Text, Bilder und Sprache dieses Videos entstanden mit KI-gestützten Werkzeugen. "
          "Alle Fakten basieren auf den genannten Quellen; Rekonstruktionen sind als solche gedacht.",
    "en": "Note: research, script, images and voice of this video were produced with AI-assisted tools. "
          "All facts are based on the sources listed; reconstructions are illustrative.",
    "tr": "Not: Bu videonun araştırması, metni, görselleri ve seslendirmesi yapay zekâ destekli araçlarla hazırlandı. "
          "Tüm olgular belirtilen kaynaklara dayanır; canlandırmalar örnekleyicidir.",
}
_SOURCES_LABEL = {"de": "Quellen", "en": "Sources", "tr": "Kaynaklar"}


def build_description(plan: ProductionPlan, chapter_starts: list[float] | None = None) -> str:
    parts = [plan.seo.description_intro.strip()]
    if chapter_starts and len(chapter_starts) == len(plan.acts):
        lines = []
        for t, act in zip(chapter_starts, plan.acts):
            m, s = divmod(int(t), 60)
            lines.append(f"{m:02d}:{s:02d} {act.chapter_title or act.name}")
        parts.append("\n".join(lines))
    parts.append(f"{_SOURCES_LABEL.get(plan.language, 'Sources')}:\n" +
                 "\n".join(f"- {s.title} ({s.lang}.wikipedia.org): {s.url}" for s in plan.sources))
    parts.append(_DISCLOSURE.get(plan.language, _DISCLOSURE["en"]))
    parts.append(" ".join(plan.seo.hashtags))
    return "\n\n".join(p for p in parts if p)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Araştırma + yazar ekibi (Adım 1 uçtan uca)")
    ap.add_argument("--lang", default=config.DEFAULT_LANG)
    ap.add_argument("--dossier", default=None, help="hazır dossier.json (araştırmayı atlar)")
    a = ap.parse_args()
    config.ensure_dirs()
    config.setup_logging()
    llm = LLM()
    dossier = Dossier.load(Path(a.dossier)) if a.dossier else ResearchAgent(llm, a.lang).run()
    result = EditorialCrew(llm, a.lang).produce(dossier)
    out = result.save()
    print(json.dumps({"plan": str(out), "words": result.word_count, "minutes": result.est_minutes,
                      "shots": result.shot_count(), "title": result.seo.title}, ensure_ascii=False, indent=2))


