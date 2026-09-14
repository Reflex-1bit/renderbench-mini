"""Cross-model run: the same RenderBench tasks, every working free NIM model,
dispatched concurrently on one shared 40 rpm budget.

    python cross_model.py [--tasks blit glyph_atlas_blit] [--rounds 2]

No secrets in this file -- the key comes from NVIDIA_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from renderbench.agent.nim import SHARED_LIMITER, WORKING, provider  # noqa: E402
from renderbench.agent.parallel import PoolJob, run_pool  # noqa: E402
from renderbench.judge import baseline_ms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["blit", "glyph_atlas_blit"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=sorted(WORKING))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or f"cross-{stamp}"
    out_dir = ROOT / "results" / "runs" / tag

    print(f"cross-model run -> results/runs/{tag}")
    print(f"models : {args.models}")
    print(f"tasks  : {args.tasks}   rounds: {args.rounds}")
    print(f"limiter: {SHARED_LIMITER.rpm} rpm shared across all providers\n")

    print("naive baselines (cached):")
    for t in args.tasks:
        print(f"  {t:20} {baseline_ms(t):8.2f} ms")
    print()

    jobs = [
        PoolJob(label=f"{m}/{t}", task_id=t, provider=provider(m),
                max_rounds=args.rounds)
        for m in args.models for t in args.tasks
    ]
    print(f"dispatching {len(jobs)} jobs concurrently...\n")

    t0 = time.perf_counter()
    results = run_pool(jobs, log_dir=out_dir, max_workers=len(jobs))
    wall = time.perf_counter() - t0

    # ---- results table -------------------------------------------------
    print("\n" + "=" * 86)
    print(f"{'model':18}{'task':20}{'solved':>8}{'rnd':>5}{'naive ms':>11}"
          f"{'best ms':>10}{'speedup':>10}")
    print("-" * 86)
    rows = []
    for m in args.models:
        for t in args.tasks:
            r = next((x for x in results if x.label == f"{m}/{t}"), None)
            if r is None or not r.ok or r.run is None:
                print(f"{m:18}{t:20}{'ERR':>8}")
                rows.append({"model": m, "task": t, "status": "error",
                             "error": (r.error if r else "missing")})
                continue
            run = r.run
            print(f"{m:18}{t:20}"
                  f"{('yes' if run.solved else 'NO'):>8}"
                  f"{(run.rounds_to_first_pass if run.solved else '-'):>5}"
                  f"{run.baseline_ms:>11.2f}"
                  f"{(f'{run.best_ms:.2f}' if run.best_ms else '-'):>10}"
                  f"{(f'{run.best_speedup:.2f}x' if run.best_speedup else '-'):>10}")
            rows.append({
                "model": m, "task": t, "status": "ok",
                "solved": run.solved,
                "rounds_to_first_pass": run.rounds_to_first_pass,
                "baseline_ms": run.baseline_ms, "best_ms": run.best_ms,
                "best_speedup": run.best_speedup,
                "wall_s": r.wall_s,
                "rounds": [
                    {"index": rd.index, "pass": rd.judge.get("pass"),
                     "stage": rd.judge.get("stage"),
                     "error": (rd.judge.get("error") or "")[:200],
                     "completion_tokens": rd.usage.get("completion_tokens"),
                     "wall_s": rd.wall_s}
                    for rd in run.rounds],
            })
    print("-" * 86)
    solved = [r for r in rows if r.get("solved")]
    print(f"solved {len(solved)}/{len(rows)}   wall {wall:.0f}s   "
          f"limiter waits {SHARED_LIMITER.waits} "
          f"({SHARED_LIMITER.total_wait_s:.0f}s total)")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_model.json").write_text(json.dumps({
        "tag": tag, "timestamp_utc": stamp,
        "models": {m: WORKING[m][0] for m in args.models},
        "tasks": args.tasks, "rounds": args.rounds,
        "endpoint": "https://integrate.api.nvidia.com/v1",
        "rpm_limit": SHARED_LIMITER.rpm,
        "limiter_waits": SHARED_LIMITER.waits,
        "limiter_total_wait_s": round(SHARED_LIMITER.total_wait_s, 1),
        "wall_s": round(wall, 1),
        "n_solved": len(solved), "n_jobs": len(rows),
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote results/runs/{tag}/cross_model.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
