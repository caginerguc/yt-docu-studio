"""Üretim planı: araştırma + yazar ekibinin çıktısı, medya/ses/render modüllerinin girdisi."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src import config


class PlanShot(BaseModel):
    source: Literal["stock_video", "archive_photo", "ai_image"]
    stock_query_en: str = ""
    archive_query_en: str = ""
    ai_prompt_en: str = ""
    camera: str = "zoom_in"
    transition: Literal["cut", "crossfade"] = "cut"


class PlanSentence(BaseModel):
    id: str
    act_id: int
    text: str
    evidence: str
    fact_ids: list[str] = Field(default_factory=list)
    pause_before: str = "none"          # none | short | long
    shots: list[PlanShot] = Field(default_factory=list)

    @property
    def words(self) -> int:
        return len(self.text.split())


class PlanAct(BaseModel):
    act_id: int
    name: str
    chapter_title: str = ""
    sentences: list[PlanSentence] = Field(default_factory=list)


class SourceRef(BaseModel):
    id: str
    title: str
    url: str
    lang: str


class SeoPlan(BaseModel):
    title: str
    description_intro: str
    tags: list[str]
    hashtags: list[str]
    thumbnail_text: str
    thumbnail_tag: str
    thumbnail_prompt_en: str
    music_mood: str
    shorts_hook: str


class ProductionPlan(BaseModel):
    topic_id: str
    language: str
    collection_id: str
    working_title: str
    structure_variant: str
    trend_link: str
    target_words: int
    word_count: int = 0
    est_minutes: float = 0.0
    sources: list[SourceRef]
    acts: list[PlanAct]
    seo: SeoPlan
    critique: dict = Field(default_factory=dict)

    def sentences(self) -> list[PlanSentence]:
        return [s for a in self.acts for s in a.sentences]

    def full_text(self) -> str:
        return " ".join(s.text for s in self.sentences())

    def shot_count(self) -> int:
        return sum(len(s.shots) for s in self.sentences())

    def recompute(self) -> None:
        sents = self.sentences()
        self.word_count = sum(s.words for s in sents)
        speech = self.word_count * 60 / config.WORDS_PER_MINUTE
        gaps = sum(config.SENTENCE_GAP_S + (config.TWIST_GAP_S if s.pause_before == "long" else 0) for s in sents)
        self.est_minutes = round((speech + gaps) / 60, 1)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or (config.OUTPUT_DIR / "plan.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ProductionPlan":
        path = Path(path or (config.OUTPUT_DIR / "plan.json"))
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
