"""Judge worker. Runs one candidate kernel in its own process and prints JSON.

Isolated on purpose: a candidate that segfaults, hangs, or exhausts memory takes
down only this process, and the parent still records a structured failure rather
than losing the run.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from renderbench import tasks  # noqa: E402
from renderbench.core import time_callable  # noqa: E402


def emit(payload):
    sys.stdout.write("@@RB@@" + json.dumps(payload) + "@@RB@@")
    sys.stdout.flush()
    sys.exit(0)


def main():
    cfg = json.loads(sys.argv[1])
    task = tasks.get(cfg["task_id"])
    seed = cfg["seed"]
    do_timing = cfg.get("timing", True)

    rng = np.random.default_rng(seed)
    inputs = task.make_inputs(rng)
    frozen = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in inputs.items()}

    # ---- load candidate -------------------------------------------------
    ns: dict = {}
    try:
        code = Path(cfg["kernel_path"]).read_text(encoding="utf-8")
        exec(compile(code, "<candidate>", "exec"), ns)
    except Exception:
        emit({"stage": "compile", "ok": False,
              "error": traceback.format_exc(limit=6)})
    fn = ns.get("kernel")
    if not callable(fn):
        emit({"stage": "compile", "ok": False,
              "error": "module defines no callable named `kernel`"})

    # ---- run ------------------------------------------------------------
    try:
        got = fn(**inputs)
    except Exception:
        emit({"stage": "run", "ok": False,
              "error": traceback.format_exc(limit=6)})

    if not isinstance(got, np.ndarray):
        emit({"stage": "run", "ok": False,
              "error": f"kernel returned {type(got).__name__}, expected np.ndarray"})

    # ---- input mutation check (contract says inputs are read-only) ------
    mutated = [k for k, v in frozen.items()
               if isinstance(v, np.ndarray) and not np.array_equal(v, inputs[k])]
    if mutated:
        emit({"stage": "contract", "ok": False,
              "error": f"kernel mutated read-only input(s) in place: {mutated}. "
                       "Return a new array instead."})

    # ---- correctness ----------------------------------------------------
    truth = np.load(cfg["truth_path"])["truth"]
    res = task.oracle.check(got, truth)
    if not res.passed:
        emit({"stage": "correctness", "ok": False, "error": res.detail,
              "oracle": {"mode": res.mode, "detail": res.detail,
                         "metrics": res.metrics}})

    payload = {"stage": "done", "ok": True, "error": "",
               "oracle": {"mode": res.mode, "detail": res.detail,
                          "metrics": res.metrics}}

    # ---- timing ---------------------------------------------------------
    if do_timing:
        t = time_callable(lambda: fn(**inputs), task.timing)
        payload["timing"] = t.to_dict()
    emit(payload)


if __name__ == "__main__":
    main()
