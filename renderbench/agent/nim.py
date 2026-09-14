"""NVIDIA NIM free-endpoint provider registry.

Every model here was probed live on 2026-09-14 against
https://integrate.api.nvidia.com/v1 and either responded or didn't. The dead
ones are kept in DEAD with the reason, because "we checked and it's gone" is
worth more later than silently dropping it and re-probing in three weeks.

All of them return `reasoning_content` alongside `content` -- the same shape
that starved GLM-5.3-free of output budget earlier in this project, so the
max_tokens defaults here are deliberately generous.
"""
from __future__ import annotations

import os

from .llm import LiveProvider, RateLimiter, _load_dotenv

_load_dotenv()

BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY_ENV = "NVIDIA_API_KEY"
# Free tier: 40 requests/minute, per KEY (not per model).
DEFAULT_RPM = int(os.environ.get("NVIDIA_RPM", "40"))

# label -> (model string, trivial-prompt latency measured 2026-09-14)
WORKING = {
    "deepseek-flash": ("deepseek-ai/deepseek-v4-flash-0731", 3.7),
    "muse-glimmer":   ("meta/muse-glimmer-30b", 3.8),
    "nemotron-ultra": ("nvidia/nemotron-3-ultra-550b-a55b", 38.6),
    "kimi-k3":        ("moonshotai/kimi-k3", 55.6),
}

DEAD = {
    "deepseek-pro": ("deepseek-ai/deepseek-v4-pro-0813",
                     "HTTP 410 Gone -- retired from the catalogue while still "
                     "listed on the models page"),
    "laguna":       ("poolside/laguna-xs-2.1",
                     "HTTP 503 ResourceExhausted -- shared free capacity "
                     "saturated (122/32 workers). May recover; re-probe."),
    "nemotron-lightning": ("nvidia/nemotron-3.5-lightning-30b-a3b",
                     "ReadTimeout at 90s on a trivial prompt, despite being "
                     "the catalogue's 'fastest' 30B entry"),
}

# One limiter instance per process, shared by every provider below, because the
# 40 rpm ceiling is a property of the key rather than of any single model.
SHARED_LIMITER = RateLimiter(DEFAULT_RPM)


def provider(label: str, budget_usd: float = 0.0, **kw) -> LiveProvider:
    """Build a provider for one verified-working NIM model.

    budget_usd defaults to 0.0 because these endpoints are free; cost tracking
    stays wired up (tokens are still counted) but there is nothing to spend.
    """
    if label in DEAD:
        model, why = DEAD[label]
        raise ValueError(f"{label} ({model}) is not usable: {why}")
    if label not in WORKING:
        raise KeyError(f"unknown NIM label {label!r}; have {sorted(WORKING)}")
    model, _lat = WORKING[label]
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set {API_KEY_ENV} in .env or the environment")
    p = LiveProvider(model=model, base_url=BASE_URL, api_key=key,
                     budget_usd=budget_usd or 10.0,
                     rate_limiter=SHARED_LIMITER, **kw)
    p.name = f"nim:{label}"
    return p


def all_working(**kw) -> dict[str, LiveProvider]:
    return {label: provider(label, **kw) for label in WORKING}
