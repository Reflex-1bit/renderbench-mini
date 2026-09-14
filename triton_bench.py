"""Triton track: generate real GPU kernels, compile and check them on the GPU.

Fast path. Generates with the NIM providers (system python), judges in the
rbvenv (torch+triton) via subprocess, with one round of compile-error feedback.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from renderbench.agent.llm import extract_kernel  # noqa: E402
from renderbench.agent.nim import provider  # noqa: E402

RBVENV = r"C:\Users\adity\rbvenv\Scripts\python.exe"
OUT = ROOT / "results" / "triton_track"

SYSTEM = """You write GPU kernels in Triton (@triton.jit, triton.language as tl). NOT numpy.

Return ONE fenced ```python block containing the COMPLETE file: imports, the
@triton.jit kernel, and a wrapper named `kernel` with the given signature.
No numpy. No .cpu()/.numpy()/.item() inside `kernel` -- all work happens in the
GPU kernel launch. Return a new tensor; never mutate the inputs in place.

Triton reminders that matter:
- offsets are 1-D; reshape with [:, None] before broadcasting against [None, :]
- mask every load and store
- integer math: .to(tl.int32) before multiplying uint8 values"""

TASKS = {
    "blit": dict(
        sig="def kernel(dst: torch.Tensor, src: torch.Tensor, dy: int, dx: int) -> torch.Tensor",
        spec="""Copy `src` (sh, sw, 4) uint8 into a COPY of `dst` (H, W, 4) uint8 with its
top-left corner at row dy, column dx. Pixels landing outside `dst` are dropped
(clip, do not wrap). The placement given will overhang the destination edges,
so clipping is exercised. Return a new (H, W, 4) uint8 tensor.""",
    ),
    "alpha_composite": dict(
        sig="def kernel(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor",
        spec="""Premultiplied source-over composite. dst, src are (H, W, 4) uint8.
Per channel c in 0..3:  out[c] = clip(src[c] + (dst[c] * (255 - src[3])) // 255, 0, 255)
Integer FLOOR division by 255. Widen to int32 before multiplying.""",
    ),
    "srgb_gamma": dict(
        sig="def kernel(linear: torch.Tensor) -> torch.Tensor",
        spec="""Encode linear light `linear` (H, W, 3) float32 in [0,1] to 8-bit sRGB,
returning (H, W, 3) uint8.
    s = 12.92 * v                          if v <= 0.0031308
    s = 1.055 * v**(1/2.4) - 0.055         otherwise
    out = clip(round(s * 255), 0, 255)
Round half away from zero (values are non-negative, so floor(x+0.5) is fine).""",
    ),
}


def judge_on_gpu(task_id: str, path: Path) -> dict:
    r = subprocess.run([RBVENV, str(ROOT / "triton_judge.py"), task_id, str(path)],
                       capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    for line in r.stdout.splitlines():
        if line.startswith("@@J@@"):
            return json.loads(line[5:])
    return {"ok": False, "stage": "judge-crash",
            "error": (r.stderr or r.stdout)[-500:]}


def run_one(model: str, task_id: str, rounds: int = 2) -> dict:
    t = TASKS[task_id]
    user = (f"Task `{task_id}`.\n\n{t['spec']}\n\nRequired wrapper signature:\n"
            f"```python\n{t['sig']}\n```")
    p = provider(model)
    p.deadline_s = 420
    d = OUT / model / task_id
    d.mkdir(parents=True, exist_ok=True)
    hist = []
    t0 = time.time()

    for i in range(rounds):
        try:
            c = p.complete(SYSTEM, user, max_tokens=4000)
        except Exception as e:
            hist.append({"round": i, "stage": "provider", "error": repr(e)[:300]})
            break
        code = extract_kernel(c.text)
        if not code:
            hist.append({"round": i, "stage": "parse", "error": "no kernel block"})
            break
        f = d / f"r{i}.py"
        f.write_text(code, encoding="utf-8")
        v = judge_on_gpu(task_id, f)
        v.update(round=i, tokens=c.usage.completion_tokens,
                 has_jit="@triton.jit" in code)
        hist.append(v)
        if v.get("ok"):
            break
        user = (f"Task `{task_id}`.\n\n{t['spec']}\n\nRequired signature:\n"
                f"```python\n{t['sig']}\n```\n\nYour previous attempt failed.\n\n"
                f"```python\n{code}\n```\n\nFailure ({v.get('stage')}):\n"
                f"{(v.get('error') or '')[:900]}\n\n"
                f"Return the COMPLETE corrected file.")

    ok = any(h.get("ok") for h in hist)
    return {"model": model, "task": task_id, "solved": ok,
            "wall_s": round(time.time() - t0, 1),
            "solved_round": next((h["round"] for h in hist if h.get("ok")), None),
            "ms": next((h.get("ms") for h in hist if h.get("ok")), None),
            "rounds": hist}


def main():
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["deepseek-flash", "muse-glimmer"]
    tasks = sys.argv[2].split(",") if len(sys.argv) > 2 else list(TASKS)
    jobs = [(m, t) for m in models for t in tasks]
    print(f"TRITON TRACK: {len(jobs)} jobs ({models} x {tasks})\n")

    res = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {ex.submit(run_one, m, t): (m, t) for m, t in jobs}
        for f in as_completed(futs):
            r = f.result()
            res.append(r)
            mark = "SOLVED" if r["solved"] else "fail"
            stages = ",".join(str(h.get("stage")) for h in r["rounds"])
            print(f"  [{r['wall_s']:6.0f}s] {r['model']:15} {r['task']:17} "
                  f"{mark:7} r={r['solved_round']} [{stages}]")

    wall = time.time() - t0
    print("\n" + "=" * 74)
    print(f"{'model':16}{'task':18}{'triton?':>9}{'solved':>8}{'rnd':>5}{'gpu ms':>10}")
    print("-" * 74)
    for m in models:
        for t in tasks:
            r = next((x for x in res if x["model"] == m and x["task"] == t), None)
            if not r:
                continue
            jit = any(h.get("has_jit") for h in r["rounds"])
            print(f"{m:16}{t:18}{('yes' if jit else 'NO'):>9}"
                  f"{('YES' if r['solved'] else 'no'):>8}"
                  f"{(r['solved_round'] if r['solved'] else '-'):>5}"
                  f"{(f'{r[chr(109)+chr(115)]:.3f}' if r['ms'] else '-'):>10}")
    print("-" * 74)
    n = sum(1 for r in res if r["solved"])
    print(f"solved {n}/{len(res)}   wall {wall:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "triton_track.json").write_text(json.dumps(
        {"models": models, "tasks": tasks, "wall_s": round(wall, 1),
         "n_solved": n, "n_jobs": len(res), "results": res}, indent=2),
        encoding="utf-8")
    print(f"wrote results/triton_track/triton_track.json")


if __name__ == "__main__":
    main()
