"""GPU judge for Triton candidates. Runs inside rbvenv (torch + triton).

    python triton_judge.py <task_id> <candidate.py>

Prints one @@J@@<json> line: {ok, stage, error, ms}.
Same oracles as the CPU track -- the candidate has to clear exactly the bar a
numpy kernel does, including the calibrated perceptual thresholds.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from renderbench import tasks  # noqa: E402
from renderbench.judge import DEFAULT_SEED  # noqa: E402

FORBIDDEN = ("import numpy", "\\.cpu()", "\\.numpy()")


def emit(**kw):
    print("@@J@@" + json.dumps(kw))
    sys.exit(0)


def main():
    task_id, path = sys.argv[1], Path(sys.argv[2])
    src = path.read_text(encoding="utf-8")

    if "@triton.jit" not in src:
        emit(ok=False, stage="screen", error="no @triton.jit in submission -- "
             "this must be a Triton GPU kernel, not a host-side implementation")

    import torch

    task = tasks.get(task_id)
    inputs = task.make_inputs(np.random.default_rng(DEFAULT_SEED))
    truth = task.reference(**inputs)

    # ---- compile ----
    try:
        spec = importlib.util.spec_from_file_location("cand", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        emit(ok=False, stage="compile", error=traceback.format_exc(limit=4)[-900:])
    if not callable(getattr(mod, "kernel", None)):
        emit(ok=False, stage="compile", error="no callable `kernel`")

    # ---- move inputs to GPU ----
    dev = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            dev[k] = torch.from_numpy(np.ascontiguousarray(v)).cuda()
        else:
            dev[k] = int(v)

    # ---- run (this is where a Triton compile error actually surfaces) ----
    try:
        out = mod.kernel(**dev)
        torch.cuda.synchronize()
    except Exception:
        emit(ok=False, stage="run", error=traceback.format_exc(limit=6)[-900:])

    if not torch.is_tensor(out):
        emit(ok=False, stage="run",
             error=f"kernel returned {type(out).__name__}, expected torch.Tensor")

    got = out.detach().cpu().numpy()

    # ---- same oracle as the CPU track ----
    res = task.oracle.check(got, truth)
    if not res.passed:
        emit(ok=False, stage="correctness", error=res.detail,
             metrics=res.metrics)

    # ---- time it (kernel-only, data already resident) ----
    for _ in range(5):
        mod.kernel(**dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        mod.kernel(**dev)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 30 * 1000

    emit(ok=True, stage="done", error="", ms=round(ms, 4),
         oracle=res.mode, detail=res.detail)


if __name__ == "__main__":
    main()
