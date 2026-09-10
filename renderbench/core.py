"""Core types for RenderBench: task specs, correctness oracles, timing methodology.

Everything in this module is METHODOLOGY. Per the project plan it is frozen once
working -- correctness thresholds are declared here alongside their justification
so they cannot be quietly tuned after results are in.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------


def _gaussian_1d(sigma: float = 1.5, radius: int = 5) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _blur2d(im: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable 'valid' gaussian blur, no scipy dependency."""
    t = sliding_window_view(im, (k.size,), axis=1) @ k
    return sliding_window_view(t, (k.size,), axis=0) @ k


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Mean SSIM over a gaussian window (sigma=1.5, 11x11), averaged over channels.

    Follows the Wang et al. 2004 formulation used by scikit-image with
    gaussian_weights=True, so numbers are comparable to published values.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 0.0
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    if min(a.shape[0], a.shape[1]) < 11:
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0

    k = _gaussian_1d()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    scores = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mx, my = _blur2d(x, k), _blur2d(y, k)
        vx = _blur2d(x * x, k) - mx * mx
        vy = _blur2d(y * y, k) - my * my
        vxy = _blur2d(x * y, k) - mx * my
        num = (2 * mx * my + c1) * (2 * vxy + c2)
        den = (mx * mx + my * my + c1) * (vx + vy + c2)
        scores.append(float(np.mean(num / den)))
    return float(np.mean(scores))


@dataclass
class OracleResult:
    passed: bool
    mode: str
    detail: str
    metrics: dict[str, float] = field(default_factory=dict)


class Oracle:
    """Base class. Subclasses decide what 'correct' means for a task family."""

    mode = "abstract"
    justification = ""

    def check(self, got: np.ndarray, want: np.ndarray) -> OracleResult:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"mode": self.mode, "justification": self.justification,
                "thresholds": self.thresholds()}

    def thresholds(self) -> dict:
        return {}


class ExactMatch(Oracle):
    """Bit-identical output. Used for integer-exact ops: blit, composite, atlas
    lookup. There is a single defensible answer, so any deviation is a bug rather
    than a rounding difference."""

    mode = "exact"
    justification = (
        "Integer-domain op with a single defensible result; any deviation is a "
        "logic bug, not a rounding artifact."
    )

    def check(self, got, want):
        if got.shape != want.shape:
            return OracleResult(False, self.mode, f"shape {got.shape} != {want.shape}")
        if got.dtype != want.dtype:
            return OracleResult(False, self.mode, f"dtype {got.dtype} != {want.dtype}")
        bad = int(np.count_nonzero(got != want))
        if bad:
            idx = tuple(int(i) for i in np.argwhere(got != want)[0])
            return OracleResult(
                False, self.mode,
                f"{bad}/{got.size} elements differ; first at {idx} "
                f"got={got[idx]} want={want[idx]}",
                {"mismatched": float(bad), "mismatch_frac": bad / got.size},
            )
        return OracleResult(True, self.mode, "bit-identical", {"mismatched": 0.0})


class ToleranceMatch(Oracle):
    """Bounded absolute error. Used where a different-but-valid order of
    operations shifts the low bit, e.g. gamma via LUT vs. via powf."""

    mode = "tolerance"

    def __init__(self, atol: float, justification: str):
        self.atol = atol
        self.justification = justification

    def thresholds(self):
        return {"atol": self.atol}

    def check(self, got, want):
        if got.shape != want.shape:
            return OracleResult(False, self.mode, f"shape {got.shape} != {want.shape}")
        d = np.abs(got.astype(np.float64) - want.astype(np.float64))
        mx = float(d.max()) if d.size else 0.0
        return OracleResult(
            mx <= self.atol, self.mode,
            f"max abs err {mx:.6g} vs atol {self.atol:g}",
            {"max_abs_err": mx, "atol": self.atol,
             "over_tol_frac": float(np.count_nonzero(d > self.atol) / d.size)},
        )


class PerceptualMatch(Oracle):
    """SSIM floor plus a hard outlier cap.

    Anti-aliased glyph coverage is NOT bit-reproducible across sampling
    strategies -- that is the methodological problem this benchmark exists to
    solve. But SSIM alone is gameable: a kernel can score 0.99 while dropping one
    glyph entirely. So we pair a structural floor with a cap on the worst single
    pixel and on how much of the image may exceed it. All three must hold.
    """

    mode = "perceptual"

    def __init__(self, min_ssim: float, max_pixel_err: float,
                 max_bad_frac: float, justification: str):
        self.min_ssim = min_ssim
        self.max_pixel_err = max_pixel_err
        self.max_bad_frac = max_bad_frac
        self.justification = justification

    def thresholds(self):
        return {"min_ssim": self.min_ssim, "max_pixel_err": self.max_pixel_err,
                "max_bad_frac": self.max_bad_frac}

    def check(self, got, want):
        if got.shape != want.shape:
            return OracleResult(False, self.mode, f"shape {got.shape} != {want.shape}")
        g = got.astype(np.float64) / 255.0
        w = want.astype(np.float64) / 255.0
        s = ssim(g, w, data_range=1.0)
        d = np.abs(g - w)
        mx = float(d.max()) if d.size else 0.0
        bad = float(np.count_nonzero(d > self.max_pixel_err) / d.size)
        checks = {"ssim": s >= self.min_ssim,
                  "max_pixel_err": mx <= self.max_pixel_err,
                  "bad_frac": bad <= self.max_bad_frac}
        failed = [k for k, v in checks.items() if not v]
        detail = (f"ssim={s:.5f} (floor {self.min_ssim}) "
                  f"max_px={mx:.4f} (cap {self.max_pixel_err}) "
                  f"bad_frac={bad:.5f} (cap {self.max_bad_frac})")
        if failed:
            detail = "FAILED [" + ",".join(failed) + "] " + detail
        return OracleResult(not failed, self.mode, detail,
                            {"ssim": s, "max_pixel_err": mx, "bad_frac": bad})


# --------------------------------------------------------------------------
# Timing methodology
# --------------------------------------------------------------------------


@dataclass
class TimingConfig:
    """Declared up front, applied identically to baseline and candidate.

    Reported statistic is the MEDIAN, not the min. Minimum-of-N flatters kernels
    with bimodal timing (a fast path plus an occasional recompile or page fault)
    and is the usual way GPU kernel benchmarks overstate speedups. We also carry
    the interquartile range so a low-variance win can be told apart from noise.
    """

    warmup: int = 5
    trials: int = 30
    inner_reps: int = 1
    statistic: str = "median"


@dataclass
class TimingResult:
    median_ms: float
    mean_ms: float
    p05_ms: float
    p95_ms: float
    iqr_ms: float
    cv: float
    trials: int
    samples_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["samples_ms"] = [round(x, 6) for x in self.samples_ms[:200]]
        return d


def time_callable(fn: Callable[[], Any], cfg: TimingConfig,
                  sync: Callable[[], None] | None = None) -> TimingResult:
    sync = sync or (lambda: None)
    for _ in range(cfg.warmup):
        fn()
    sync()
    samples: list[float] = []
    for _ in range(cfg.trials):
        t0 = time.perf_counter()
        for _ in range(cfg.inner_reps):
            fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0 / cfg.inner_reps)
    samples.sort()
    n = len(samples)
    mean = statistics.fmean(samples)
    q1 = samples[n // 4]
    q3 = samples[min(n - 1, (3 * n) // 4)]
    sd = statistics.pstdev(samples)
    return TimingResult(
        median_ms=statistics.median(samples), mean_ms=mean,
        p05_ms=samples[max(0, int(0.05 * n))],
        p95_ms=samples[min(n - 1, int(0.95 * n))],
        iqr_ms=q3 - q1, cv=(sd / mean) if mean else 0.0,
        trials=n, samples_ms=samples,
    )


# --------------------------------------------------------------------------
# Task spec
# --------------------------------------------------------------------------


@dataclass
class TaskSpec:
    task_id: str
    title: str
    summary: str
    signature: str
    description: str
    oracle: Oracle
    make_inputs: Callable[[np.random.Generator], dict[str, np.ndarray]]
    reference: Callable[..., np.ndarray]
    naive_source: str
    vtc_relevance: str = ""
    timing: TimingConfig = field(default_factory=TimingConfig)

    def prompt_block(self) -> str:
        return (
            f"# Task `{self.task_id}` -- {self.title}\n\n"
            f"{self.description.strip()}\n\n"
            f"Required signature:\n```python\n{self.signature}\n```\n\n"
            f"Correctness oracle: {self.oracle.mode}. {self.oracle.justification}\n"
            f"Thresholds: {self.oracle.thresholds()}\n"
        )
