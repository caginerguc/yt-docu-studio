"""Gemini sarmalayıcı: hız sınırlama, retry/backoff, model yedekleme, şema doğrulama.

Davranış özeti
  * 503/429/5xx  -> üstel bekleme ile aynı modelde tekrar dene, tükenirse sıradaki modele geç.
  * 404 / "model yok" / günlük kota bitti -> modeli çalışma boyunca devre dışı bırak, zinciri kır(ma).
  * Şema veya doğrulama hatası -> hatayı modele geri bildirerek yeniden iste (aynı model).
  * Kimlik doğrulama gibi kalıcı hatalar -> döngüye girmeden hemen yükselt.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from src import config

log = logging.getLogger("makrokontext.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Tüm modeller/denemeler tükendi veya kalıcı hata alındı."""


class _EmptyResponse(Exception):
    pass


_DEAD_PAT = re.compile(
    r"not found|is not supported|no longer available|deprecated|does not exist|unsupported model|"
    r"shut ?down|retired",
    re.I,
)
_RETRY_PAT = re.compile(
    r"unavailable|overloaded|high demand|resource_exhausted|deadline|timed? ?out|temporarily|internal|disconnected|connection|protocol|reset|eof",
    re.I,
)
_DAILY_PAT = re.compile(r"per ?day|daily|requests_per_day", re.I)
_RETRY_CODES = {408, 429, 500, 502, 503, 504}


def _code_of(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    m = re.search(r"\b([45]\d\d)\b", str(exc))
    return int(m.group(1)) if m else None


def classify_error(exc: BaseException) -> str:
    """'retry' | 'dead' | 'fatal'"""
    code, msg = _code_of(exc), str(exc)
    if code == 404:
        return "dead"
    if code == 429:
        return "dead" if _DAILY_PAT.search(msg) else "retry"
    if code in (400, 403) and _DEAD_PAT.search(msg):
        return "dead"
    if code in _RETRY_CODES or _RETRY_PAT.search(msg):
        return "retry"
    return "fatal"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _default_config_builder(**kw: Any) -> Any:
    from google.genai import types  # tembel import: test ortamında SDK şart değil

    safety = None
    try:
        cats = ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        # Savaş/istihbarat/soykırım gibi tarihsel konular için eşik gevşetilir (Gemini politikaları içinde).
        safety = [types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH") for c in cats]
    except Exception:  # eski SDK
        safety = None
    if safety:
        kw["safety_settings"] = safety
    return types.GenerateContentConfig(**kw)


class LLM:
    def __init__(self, client: Any = None, models: list[str] | None = None,
                 config_builder: Callable[..., Any] | None = None, sleep: Callable[[float], None] = time.sleep):
        if client is None:
            from google import genai  # tembel import

            if not config.GEMINI_API_KEY:
                raise LLMError("GEMINI_API_KEY eksik.")
            client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.client = client
        chain = models or [config.TEXT_MODEL, *config.TEXT_FALLBACK_MODELS]
        self.models = list(dict.fromkeys(m for m in chain if m))
        self.dead: set[str] = set()
        self._build = config_builder or _default_config_builder
        self._sleep = sleep
        self._last_call = 0.0
        self.calls = 0

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        wait = config.LLM_MIN_INTERVAL_S - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def _backoff(self, attempt: int) -> float:
        return min(config.LLM_BACKOFF_BASE_S * 2 ** (attempt - 1), config.LLM_BACKOFF_MAX_S) + random.uniform(0, 2)

    def _run(self, prompt: str, cfg_kw: dict[str, Any], parse: Callable[[str], Any], label: str) -> Any:
        models = [m for m in self.models if m not in self.dead]
        if not models:
            raise LLMError("Kullanılabilir model kalmadı (hepsi devre dışı).")
        last_err: BaseException | None = None

        for model in models:
            feedback = ""
            for attempt in range(1, config.LLM_MAX_RETRIES + 1):
                self._throttle()
                try:
                    self.calls += 1
                    resp = self.client.models.generate_content(
                        model=model, contents=prompt + feedback, config=self._build(**cfg_kw)
                    )
                    text = (getattr(resp, "text", None) or "").strip()
                    if not text:
                        raise _EmptyResponse("boş yanıt")
                except _EmptyResponse as e:
                    last_err = e
                    log.warning("[%s] %s boş yanıt (deneme %d)", label, model, attempt)
                    self._sleep(self._backoff(attempt))
                    continue
                except Exception as e:  # SDK hataları
                    last_err = e
                    kind = classify_error(e)
                    if kind == "dead":
                        self.dead.add(model)
                        log.warning("[%s] %s devre dışı bırakıldı: %s", label, model, str(e)[:160])
                        break
                    if kind == "retry":
                        wait = self._backoff(attempt)
                        log.warning("[%s] %s geçici hata (deneme %d/%d), %.0fs bekleniyor: %s",
                                    label, model, attempt, config.LLM_MAX_RETRIES, wait, str(e)[:120])
                        self._sleep(wait)
                        continue
                    raise LLMError(f"[{label}] kalıcı hata ({model}): {e}") from e

                try:
                    return parse(text)
                except (ValueError, TypeError) as e:  # pydantic.ValidationError ve JSONDecodeError ValueError alt sınıfıdır
                    last_err = e
                    feedback = (
                        "\n\n[SYSTEM NOTE: your previous answer was rejected: "
                        f"{str(e)[:700]} Fix every point and return the complete answer again.]"
                    )
                    log.warning("[%s] %s doğrulama reddi (deneme %d): %s", label, model, attempt, str(e)[:160])
            else:
                log.warning("[%s] %s denemeleri tükendi, sıradaki modele geçiliyor.", label, model)

        raise LLMError(f"[{label}] tüm modeller/denemeler başarısız. Son hata: {last_err}")

    # ------------------------------------------------------------------ #
    def generate_text(self, prompt: str, *, system: str | None = None, temperature: float = 0.7,
                      max_output_tokens: int | None = None, label: str = "text") -> str:
        kw: dict[str, Any] = {"temperature": temperature}
        if system:
            kw["system_instruction"] = system
        if max_output_tokens:
            kw["max_output_tokens"] = max_output_tokens
        return self._run(prompt, kw, lambda t: t, label)

    def generate_json(self, prompt: str, schema: type[T], *, system: str | None = None,
                      temperature: float = 0.5, validate: Callable[[T], None] | None = None,
                      max_output_tokens: int | None = None, label: str = "json") -> T:
        kw: dict[str, Any] = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        if system:
            kw["system_instruction"] = system
        if max_output_tokens:
            kw["max_output_tokens"] = max_output_tokens

        def parse(text: str) -> T:
            obj = schema.model_validate_json(_strip_fences(text))
            if validate:
                validate(obj)  # ValueError fırlatırsa modele geri bildirim gider
            return obj

        return self._run(prompt, kw, parse, label)

