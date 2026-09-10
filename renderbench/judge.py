"""The judge: compile, run, check, time. FROZEN once working.

Per the project plan this module is the interface both lanes build against, and
neither author edits it again after it works. The Sakana CUDA-Engineer incident
is the cautionary tale -- a judge that stays editable while experiments run is a
judge the experiments will eventually be shaped around.

    judge(kernel_code, task_id) -> JudgeResult

JudgeResult is dict-compatible with the interface agreed in the plan
({pass, error, timing}) and carries the extra diagnostics the advisor agent needs.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import tasks

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "assets"
CACHE.mkdir(exist_ok=True)

DEFAULT_SEED = 20260909
TIMEOUT_S = 180

# Anti-reward-hacking: a candidate that reaches into the harness, the cached
# ground truth, or the reference implementation is not solving the task.
FORBIDDEN = [
    (r"\bimport\s+renderbench\b|\bfrom\s+renderbench\b", "imports the harness"),
    (r"\bfrom\s+\.\s*import|\bimport\s+_worker\b", "imports harness internals"),
    (r"truth[_-]?\w*\.npz|assets[/\\]", "reads the cached ground truth"),
    (r"\breference\s*\(", "calls the reference implementation"),
    (r"\bsubprocess\b|\bos\.system\b|\bsocket\b", "spawns processes or network"),
    (r"__import__\s*\(\s*[\"']renderbench", "dynamically imports the harness"),
]


@dataclass
class JudgeResult:
    passed: bool
    error: str
    timing: float | None            # median ms, None if it never ran
    stage: str                      # compile | run | contract | correctness | done
    task_id: str
    speedup: float | None = None    # vs. the naive baseline
    oracle: dict = field(default_factory=dict)
    timing_detail: dict = field(default_factory=dict)
    baseline_ms: float | None = None

    # -- the interface literally agreed in the plan -------------------------
    def as_interface(self) -> dict:
        return {"pass": self.passed, "error": self.error, "timing": self.timing}

    def to_dict(self) -> dict:
        return {"pass": self.passed, "error": self.error, "timing": self.timing,
                "stage": self.stage, "task_id": self.task_id,
                "speedup": self.speedup, "baseline_ms": self.baseline_ms,
                "oracle": self.oracle, "timing_detail": self.timing_detail}

    def summary(self) -> str:
        if self.passed:
            return (f"PASS  {self.timing:.2f} ms"
                    + (f"  ({self.speedup:.2f}x vs naive)" if self.speedup else ""))
        return f"FAIL[{self.stage}]  {self.error.strip().splitlines()[-1][:120]}"


def screen_source(code: str) -> str | None:
    """Static check run before the candidate is ever executed."""
    for pattern, why in FORBIDDEN:
        if re.search(pattern, code):
            return f"rejected: candidate {why} (matched /{pattern}/)"
    if "def kernel" not in code:
        return "rejected: no `def kernel` in submission"
    return None


def truth_path(task_id: str, seed: int = DEFAULT_SEED) -> Path:
    """Ground truth is computed once from the reference and cached, so every
    candidate in every iteration is scored against byte-identical targets."""
    p = CACHE / f"truth_{task_id}_{seed}.npz"
    if not p.exists():
        task = tasks.get(task_id)
        inputs = task.make_inputs(np.random.default_rng(seed))
        np.savez_compressed(p, truth=task.reference(**inputs))
    return p


def _run_worker(task_id: str, kernel_path: Path, seed: int,
                timing: bool) -> dict:
    cfg = json.dumps({"task_id": task_id, "kernel_path": str(kernel_path),
                      "seed": seed, "timing": timing,
                      "truth_path": str(truth_path(task_id, seed))})
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "renderbench" / "_worker.py"), cfg],
            capture_output=True, text=True, timeout=TIMEOUT_S, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"stage": "run", "ok": False,
                "error": f"candidate exceeded the {TIMEOUT_S}s judge timeout"}
    m = re.search(r"@@RB@@(.*?)@@RB@@", proc.stdout, re.S)
    if not m:
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        return {"stage": "run", "ok": False,
                "error": f"worker died (exit {proc.returncode}):\n{tail}"}
    return json.loads(m.group(1))


def baseline_ms(task_id: str, seed: int = DEFAULT_SEED, refresh=False) -> float:
    """Time the naive reference under the identical timing config.

    Cached, because it is slow and does not change -- but cached per task so the
    speedup denominator is always the same number every candidate is measured
    against.
    """
    p = CACHE / f"baseline_{task_id}_{seed}.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())["median_ms"]
    task = tasks.get(task_id)
    src = (f"import numpy as np\nfrom renderbench.tasks import "
           f"{task.reference.__name__} as _ref\n"
           f"def kernel(**kw):\n    return _ref(**kw)\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(src)
        tmp = Path(fh.name)
    try:
        out = _run_worker(task_id, tmp, seed, timing=True)
    finally:
        tmp.unlink(missing_ok=True)
    if not out.get("ok"):
        raise RuntimeError(f"baseline for {task_id} failed: {out.get('error')}")
    d = out["timing"]
    p.write_text(json.dumps(d, indent=2))
    return d["median_ms"]


def judge(kernel_code: str, task_id: str, seed: int = DEFAULT_SEED,
          timing: bool = True) -> JudgeResult:
    """Compile, run, check correctness, and time one candidate kernel."""
    reject = screen_source(kernel_code)
    if reject:
        return JudgeResult(False, reject, None, "screen", task_id)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(kernel_code)
        tmp = Path(fh.name)
    try:
        out = _run_worker(task_id, tmp, seed, timing)
    finally:
        tmp.unlink(missing_ok=True)

    if not out.get("ok"):
        return JudgeResult(False, out.get("error", "unknown"), None,
                           out.get("stage", "run"), task_id,
                           oracle=out.get("oracle", {}))

    t = out.get("timing", {})
    med = t.get("median_ms")
    base = baseline_ms(task_id, seed) if (timing and med) else None
    return JudgeResult(
        passed=True, error="", timing=med, stage="done", task_id=task_id,
        speedup=(base / med) if (base and med) else None,
        oracle=out.get("oracle", {}), timing_detail=t, baseline_ms=base,
    )
