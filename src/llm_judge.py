"""NIM ekibi: Eleştirmen, Doğrulayıcı, Sinematograf ve SEO ajanları Gemini'den FARKLI model ailesinde (NVIDIA NIM,
ücretsiz) çalışır. Gemini'nin kısıtlı günlük kotası araştırma ve senaryo yazımına kalır.

Neden: metni yazan model kendi metnini puanlarsa kendine kayırma olur ve kendi hatalarını görmez. Hakem başka bir
aileden olunca eleştiri ve olgu denetimi gerçekten bağımsız olur.

Güvenlik ağı (hat asla hakem yüzünden durmaz):
  * NVIDIA_API_KEY yoksa veya JUDGE_ENABLED=0 ise her çağrı doğrudan Gemini'ye (fallback LLM) gider.
  * NIM modellerinin hepsi başarısız olursa ilgili çağrı Gemini'ye düşer.
  * Anahtar reddedilirse (401) hakem kapatılır, kalan çağrılar Gemini'ye gider.

Ortam değişkenleri (hepsi isteğe bağlı):
  NVIDIA_API_KEY   NIM anahtarı (nvapi-...)
  JUDGE_MODELS     virgülle ayrılmış model listesi, ilk model tercih edilir
  JUDGE_ENABLED    "0" ise hakem kapalı
  JUDGE_TIMEOUT_S  tek isteğin okuma zaman aşımı (varsayılan 300)
  JUDGE_BUDGET_S   tüm çalışma boyunca NIM'de harcanabilecek toplam süre (varsayılan 3600); dolunca Gemini'ye geçilir
  NIM_BASE_URL     varsayılan https://integrate.api.nvidia.com/v1
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Callable, TypeVar

import requests
from pydantic import BaseModel

from src.llm import LLM, LLMError

log = logging.getLogger("makrokontext.judge")
T = TypeVar("T", bound=BaseModel)

# Sıra gerçek çalışmaya göre: nemotron yanıt veriyor (ama uzun yanıtta kesilebiliyor -> yüksek belirteç sınırı),
# Mistral Large büyük isteklerde sık sık zaman aşımına düşüyor -> sona alındı.
DEFAULT_JUDGE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "z-ai/glm-5.3-flash",
    "mistralai/mistral-large-4-1t-a52b",
]

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


class _Transient(Exception):
    """Geçici hata: bekle ve tekrar dene."""


class _Slow(Exception):
    """Zaman aşımı / boş yanıt: aynı modeli tekrar deneme, sıradaki modele geç."""


class _DeadModel(Exception):
    """Bu model bu çalışmada kullanılamaz (404, 400, 403 ...)."""


class _Auth(Exception):
    """Anahtar reddedildi: hakemi tamamen kapat."""


class _Unavailable(Exception):
    """NIM yolu tükendi: çağrı Gemini'ye düşer."""


def extract_json(text: str) -> str:
    """Model yanıtından ilk tam JSON nesnesini çıkarır (düşünme blokları, kod çitleri, ön/son metin temizlenir)."""
    t = _THINK.sub("", text or "").strip()
    fence = _FENCE.search(t)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object found in the answer")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    raise ValueError("JSON object is not closed (answer was cut off?)")


class JudgeLLM:
    """LLM ile aynı generate_json / generate_text imzası; önce NIM, olmazsa Gemini."""

    def __init__(self, fallback: LLM, api_key: str | None = None, models: list[str] | None = None,
                 session: Any = None, sleep: Callable[[float], None] = time.sleep,
                 max_attempts: int = 3, min_interval_s: float = 2.5):
        self.fallback = fallback
        self.api_key = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "").strip()
        raw = models if models is not None else [m.strip() for m in os.environ.get("JUDGE_MODELS", "").split(",")]
        self.models = list(dict.fromkeys(m for m in raw if m)) or list(DEFAULT_JUDGE_MODELS)
        self.url = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/") + "/chat/completions"
        self.timeout = float(os.environ.get("JUDGE_TIMEOUT_S", "300"))
        self.budget_s = float(os.environ.get("JUDGE_BUDGET_S", "3600"))
        self.spent_s = 0.0
        self.session = session or requests.Session()
        self._sleep = sleep
        self.max_attempts = max_attempts
        self.min_interval_s = min_interval_s
        self.dead: set[str] = set()
        self.active = bool(self.api_key) and os.environ.get("JUDGE_ENABLED", "1") != "0"
        self.calls = 0
        self.fallback_calls = 0
        self.last_model = ""
        self._last_call = 0.0

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        if self.active:
            return f"NVIDIA NIM (eleştirmen, doğrulayıcı, sinematograf, SEO): {', '.join(self.models)}"
        return "Gemini (NVIDIA_API_KEY yok veya NIM kapalı: tüm roller aynı model ailesinde ve aynı kotada)"

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(5 * 2 ** (attempt - 1), 40) + random.uniform(0, 2)

    def _chat(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        try:
            r = self.session.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
                json={"model": model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, "stream": False},
                timeout=(15, self.timeout),
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            raise _Slow(f"zaman aşımı/bağlantı: {str(e)[:100]}") from e
        except requests.exceptions.RequestException as e:
            raise _Transient(f"ağ hatası: {str(e)[:120]}") from e
        code = r.status_code
        if code == 200:
            try:
                text = r.json()["choices"][0]["message"].get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
                raise _Transient(f"beklenmeyen yanıt biçimi: {e}") from e
            if not text.strip():
                raise _Slow("boş yanıt (düşünme belirteçleri bütçeyi yemiş olabilir)")
            return text
        body = (getattr(r, "text", "") or "")[:200]
        if code == 401:
            raise _Auth(body)
        if code in (400, 403, 404, 410, 422):
            raise _DeadModel(f"HTTP {code}: {body}")
        raise _Transient(f"HTTP {code}: {body}")

    def _run(self, system: str, prompt: str, temperature: float, max_tokens: int,
             parse: Callable[[str], Any], label: str) -> Any:
        models = [m for m in self.models if m not in self.dead]
        if not models:
            self.active = False
            raise _Unavailable("kullanılabilir NIM modeli kalmadı")
        last: BaseException | None = None
        for model in models:
            feedback = ""
            for attempt in range(1, self.max_attempts + 1):
                if self.spent_s >= self.budget_s:
                    self.active = False
                    raise _Unavailable(f"NIM zaman bütçesi doldu ({self.spent_s:.0f}s)")
                self._throttle()
                t0 = time.monotonic()
                try:
                    self.calls += 1
                    text = self._chat(model, system, prompt + feedback, temperature, max_tokens)
                except _Slow as e:
                    last = e
                    log.warning("[%s] %s yanıt vermedi, sıradaki modele geçiliyor: %s", label, model, e)
                    break
                except _Auth as e:
                    self.active = False
                    raise _Unavailable(f"NVIDIA anahtarı reddedildi: {e}") from e
                except _DeadModel as e:
                    self.dead.add(model)
                    last = e
                    log.warning("[%s] %s devre dışı: %s", label, model, e)
                    break
                except _Transient as e:
                    last = e
                    wait = self._backoff(attempt)
                    log.warning("[%s] %s geçici hata (deneme %d/%d), %.0fs bekleniyor: %s",
                                label, model, attempt, self.max_attempts, wait, e)
                    self._sleep(wait)
                    continue
                finally:
                    self.spent_s += time.monotonic() - t0
                try:
                    out = parse(text)
                    self.last_model = model
                    return out
                except (ValueError, TypeError) as e:
                    last = e
                    feedback = ("\n\n[SYSTEM NOTE: your previous answer was rejected: "
                                f"{str(e)[:600]} Return ONLY the corrected JSON object.]")
                    log.warning("[%s] %s doğrulama reddi (deneme %d): %s", label, model, attempt, str(e)[:140])
        raise _Unavailable(f"tüm NIM modelleri başarısız. Son hata: {last}")

    # ------------------------------------------------------------------ #
    def generate_json(self, prompt: str, schema: type[T], *, system: str | None = None,
                      temperature: float = 0.3, validate: Callable[[T], None] | None = None,
                      max_output_tokens: int | None = None, label: str = "json", allow_fallback: bool = True) -> T:
        if self.active:
            sys_txt = (system + "\n\n" if system else "") + (
                "OUTPUT FORMAT: reply with ONE JSON object and nothing else (no markdown, no commentary). "
                "It must validate against this JSON Schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False))

            def parse(text: str) -> T:
                obj = schema.model_validate_json(extract_json(text))
                if validate:
                    validate(obj)
                return obj

            try:
                return self._run(sys_txt, prompt, temperature, max_output_tokens or 16000, parse, label)
            except _Unavailable as e:
                if not allow_fallback:
                    raise LLMError(f"[{label}] NIM kullanılamadı: {e}") from e
                log.warning("[%s] hakem NIM yolu kullanılamadı, Gemini'ye düşülüyor: %s", label, e)
        if not allow_fallback:
            raise LLMError(f"[{label}] NIM kapalı (NVIDIA_API_KEY yok veya JUDGE_ENABLED=0)")
        self.fallback_calls += 1
        return self.fallback.generate_json(prompt, schema, system=system, temperature=temperature,
                                           validate=validate, max_output_tokens=max_output_tokens, label=label,
                                           use_secondary=False)

    def generate_text(self, prompt: str, *, system: str | None = None, temperature: float = 0.3,
                      max_output_tokens: int | None = None, label: str = "text", allow_fallback: bool = True) -> str:
        if self.active:
            try:
                return self._run(system or "", prompt, temperature, max_output_tokens or 16000,
                                 lambda t: _THINK.sub("", t).strip(), label)
            except _Unavailable as e:
                if not allow_fallback:
                    raise LLMError(f"[{label}] NIM kullanılamadı: {e}") from e
                log.warning("[%s] hakem NIM yolu kullanılamadı, Gemini'ye düşülüyor: %s", label, e)
        if not allow_fallback:
            raise LLMError(f"[{label}] NIM kapalı (NVIDIA_API_KEY yok veya JUDGE_ENABLED=0)")
        self.fallback_calls += 1
        return self.fallback.generate_text(prompt, system=system, temperature=temperature,
                                           max_output_tokens=max_output_tokens, label=label, use_secondary=False)
