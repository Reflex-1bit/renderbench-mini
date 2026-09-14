"""Model providers.

Two implementations behind one interface:

  LiveProvider    -- any OpenAI-compatible /chat/completions endpoint. GLM-5.3 is
                     the pinned model for reported results; the exact model
                     string and every request are logged.
  OfflineProvider -- draws from the replay pool in renderbench.pool so the loop
                     is demonstrable with no API key. Results carry
                     provider="offline" and are never a measurement of a model.

Only `requests` is required; the openai SDK is not a dependency.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


class RateLimiter:
    """Token bucket shared by every provider that shares one API key.

    A per-key rate limit (NVIDIA NIM's free tier: 40 requests/minute) belongs to
    the KEY, not to any one model, so concurrent pool jobs hitting four
    different models on the same key are all spending the same budget. Without
    this, running jobs in parallel just earns 429s faster than running them
    serially -- the limiter is what makes concurrency actually pay.

    Spaces request *starts* evenly (60/rpm apart) rather than allowing a burst
    then a stall, because bursts are what trip most providers' limiters.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.min_interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next_ok = 0.0
        self.waits = 0
        self.total_wait_s = 0.0

    def acquire(self) -> float:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_ok - now)
            self._next_ok = max(now, self._next_ok) + self.min_interval
            if wait > 0:
                self.waits += 1
                self.total_wait_s += wait
        if wait > 0:
            time.sleep(wait)
        return wait


# Permanent failures -- retrying these burns the call deadline for nothing.
# 410 Gone is real and observed: deepseek-v4-pro-0813 was retired from NVIDIA's
# catalogue while still listed on its models page.
PERMANENT_HTTP = {400, 401, 403, 404, 410, 422}


class PermanentProviderError(RuntimeError):
    """A failure retrying cannot fix: retired model, bad key, malformed request.
    Raised past the retry loop so it fails in seconds instead of burning the
    whole call deadline on four identical rejections."""


def _load_dotenv() -> None:
    """Populate os.environ from a .env file next to run_bench.py, without
    overwriting variables the shell already set. Runs once at import time, so
    RB_MODEL / RB_BASE_URL / RB_API_KEY from .env are visible before the
    PINNED_* constants below are read -- a .env-only RB_BASE_URL used to be
    silently ignored because only the API key had a .env fallback."""
    p = Path(__file__).resolve().parent.parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

# --------------------------------------------------------------------------
# Pinned model configuration
# --------------------------------------------------------------------------

# Per the plan: pin the exact model string, do not let it silently drift.
PINNED_MODEL = os.environ.get("RB_MODEL", "glm-5.3")
PINNED_BASE_URL = os.environ.get("RB_BASE_URL",
                                 "https://open.bigmodel.cn/api/paas/v4")
API_KEY_ENV = "RB_API_KEY"

# USD per million tokens, from the plan.
PRICE_IN_PER_M = float(os.environ.get("RB_PRICE_IN", "1.40"))
PRICE_OUT_PER_M = float(os.environ.get("RB_PRICE_OUT", "4.40"))

# Hard budget cap. The loop refuses to issue a call that would cross it.
BUDGET_USD = float(os.environ.get("RB_BUDGET_USD", "5.00"))


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens / 1e6 * PRICE_IN_PER_M
                + self.completion_tokens / 1e6 * PRICE_OUT_PER_M)


@dataclass
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    raw: dict = field(default_factory=dict)


class BudgetExceeded(RuntimeError):
    pass


class Provider:
    name = "abstract"
    model = "none"

    def complete(self, system: str, user: str, **kw) -> Completion:
        raise NotImplementedError


# --------------------------------------------------------------------------


class LiveProvider(Provider):
    """Speaks either an OpenAI-compatible /chat/completions endpoint (GLM-5.3
    direct, tokenrouter, most aggregators) or Anthropic's native /messages API
    (api.anthropic.com) -- auto-detected from base_url, since the two use
    different auth headers, request shapes, and response/usage field names.
    """

    name = "live"

    def __init__(self, model: str = PINNED_MODEL, base_url: str = PINNED_BASE_URL,
                 api_key: str | None = None, temperature: float = 0.2,
                 budget_usd: float = BUDGET_USD,
                 rate_limiter: "RateLimiter | None" = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(
                f"no API key: set {API_KEY_ENV} in the environment or put "
                f"{API_KEY_ENV}=... in a .env file next to run_bench.py")
        self.temperature = temperature
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.calls = 0
        self.backend = "anthropic" if "anthropic.com" in self.base_url else "openai"
        # Shared across every provider on the same key; see RateLimiter.
        self.rate_limiter = rate_limiter

    def _request(self, system: str, user: str, max_tokens: int):
        """Build (url, headers, body) for the configured backend."""
        if self.backend == "anthropic":
            # No `temperature`: Opus 5 rejects it as a deprecated parameter
            # (400 invalid_request_error) -- confirmed against the live API,
            # not assumed. Sampling is fixed server-side for this model.
            return (
                f"{self.base_url}/messages",
                {"x-api-key": self.api_key,
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
                {"model": self.model, "system": system,
                 "messages": [{"role": "user", "content": user}],
                 "max_tokens": max_tokens},
            )
        return (
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}",
             "Content-Type": "application/json"},
            {"model": self.model,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}],
             "temperature": self.temperature, "max_tokens": max_tokens},
        )

    def _parse(self, data: dict) -> tuple[str | None, Usage, str]:
        """Return (text_or_None, usage, diagnostic) for the configured backend."""
        if self.backend == "anthropic":
            u = data.get("usage", {}) or {}
            usage = Usage(int(u.get("input_tokens", 0)),
                          int(u.get("output_tokens", 0)))
            blocks = data.get("content") or []
            text = "".join(b.get("text", "") for b in blocks
                          if b.get("type") == "text") or None
            return text, usage, f"stop_reason={data.get('stop_reason')}"
        u = data.get("usage", {}) or {}
        usage = Usage(int(u.get("prompt_tokens", 0)),
                      int(u.get("completion_tokens", 0)))
        choice = data["choices"][0]
        msg = choice["message"]
        text = msg.get("content")
        if not text:
            # Reasoning models (GLM-5.3 included) can spend the whole
            # max_tokens budget on `reasoning_content` and return an
            # empty/None `content`. Not a transport failure -- retrying
            # identically just repeats it -- so fall back to scanning
            # reasoning_content for the kernel block before giving up.
            reasoning = msg.get("reasoning_content") or ""
            if extract_kernel(reasoning):
                text = reasoning
            else:
                return (None, usage,
                       f"finish_reason={choice.get('finish_reason')} "
                       f"reasoning_content={len(reasoning)} chars, no kernel in it")
        return text, usage, f"finish_reason={choice.get('finish_reason')}"

    def complete(self, system: str, user: str, max_tokens: int = 12000,
                deadline_s: float = 240) -> Completion:
        import requests

        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} of ${self.budget_usd:.2f} cap")

        url, headers, body = self._request(system, user, max_tokens)
        last = None
        # A single call is capped at deadline_s wall-clock TOTAL, across every
        # retry. `requests`' own `timeout=` only bounds the gap between bytes,
        # not the call as a whole -- a slow-but-technically-alive connection
        # can duck under that indefinitely. One live-mode call against a
        # tokenrouter free tier measurably ran ~200 minutes and still failed
        # before this existed. deadline_s is the real ceiling; per-request
        # timeout below is deliberately short so there is time left for a retry.
        started = time.monotonic()
        for attempt in range(4):
            remaining = deadline_s - (time.monotonic() - started)
            if remaining <= 5:
                last = last or "no time left in the call deadline"
                break
            try:
                if self.rate_limiter is not None:
                    self.rate_limiter.acquire()
                    remaining = deadline_s - (time.monotonic() - started)
                    if remaining <= 5:
                        last = last or "call deadline spent waiting on the rate limiter"
                        break
                r = requests.post(url, headers=headers, json=body,
                                  timeout=min(90, remaining))
                if r.status_code in PERMANENT_HTTP:
                    # No point retrying a retired model or a bad key -- fail
                    # now rather than spending the whole deadline on it.
                    raise PermanentProviderError(
                        f"permanent HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(min(2 ** attempt, max(0.0, remaining - 1)))
                    continue
                r.raise_for_status()
                data = r.json()
                text, usage, diag = self._parse(data)
                self.spent_usd += usage.cost_usd
                self.calls += 1
                if text is None:
                    raise RuntimeError(f"empty response content ({diag})")
                return Completion(text, usage, data)
            except PermanentProviderError:
                raise
            except Exception as e:  # noqa: BLE001
                last = repr(e)
                remaining = deadline_s - (time.monotonic() - started)
                time.sleep(min(2 ** attempt, max(0.0, remaining - 1)))
        elapsed = time.monotonic() - started
        raise RuntimeError(
            f"provider gave up after {elapsed:.0f}s (deadline {deadline_s:.0f}s): "
            f"{last}")


# --------------------------------------------------------------------------


class OfflineProvider(Provider):
    """Replays the candidate pool. The code it returns is real and is really
    judged; only the choice of what to write next is scripted."""

    name = "offline"
    model = "replay-pool"

    def __init__(self):
        from .. import pool
        self.pool = pool.POOL
        self._cursor: dict[str, int] = {}
        self.calls = 0
        self.spent_usd = 0.0

    def next_for(self, task_id: str) -> tuple[str, str]:
        entries = self.pool[task_id]
        i = min(self._cursor.get(task_id, 0), len(entries) - 1)
        self._cursor[task_id] = i + 1
        self.calls += 1
        return entries[i].source.strip(), entries[i].note

    def exhausted(self, task_id: str) -> bool:
        return self._cursor.get(task_id, 0) >= len(self.pool[task_id])

    def complete(self, system: str, user: str, **kw) -> Completion:
        raise RuntimeError("OfflineProvider is driven via next_for(), not complete()")


# --------------------------------------------------------------------------


def extract_kernel(text: str) -> str | None:
    """Pull the kernel source out of a model response.

    Response parsing has to work every time or the loop silently degrades into
    measuring the parser. Three fallbacks, most specific first.
    """
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, re.S)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.S)
    for b in blocks:
        if "def kernel" in b:
            return b.strip()
    if "def kernel" in text:
        # Unfenced: take from the first import or def to the end.
        m = re.search(r"^(?:import |from |def kernel|_[A-Z])", text, re.M)
        if m:
            return text[m.start():].strip()
    return None


def make_provider(mode: str) -> Provider:
    if mode == "offline":
        return OfflineProvider()
    if mode == "live":
        return LiveProvider()
    raise ValueError(f"unknown provider mode {mode!r}")
