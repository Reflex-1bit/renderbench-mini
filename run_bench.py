"""RenderBench-mini runner.

    python run_bench.py                        # offline replay, all tasks
    python run_bench.py --provider live        # GLM-5.3 (needs RB_API_KEY)
    python run_bench.py --tasks glyph_atlas_blit text_page_raster
    python run_bench.py --no-advisor           # single-agent ablation (live only)

Refuses to run until the perceptual oracle's thresholds have been calibrated, so
results can never be produced against thresholds that do not separate a known-bad
implementation from a known-good one.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from renderbench import tasks                                    # noqa: E402
from renderbench.agent.llm import (PINNED_MODEL, OfflineProvider,  # noqa: E402
                                   make_provider)
from renderbench.agent.loop import run_task                      # noqa: E402
from renderbench.agent.prompts import PROMPT_VERSION             # noqa: E402
from renderbench.judge import DEFAULT_SEED, baseline_ms          # noqa: E402

CALIB = ROOT / "results" / "threshold_calibration.json"


def calibration_gate(force: bool) -> dict:
    if not CALIB.exists():
        print("No threshold calibration found. Running calibrate_thresholds.py "
              "first...\n")
        subprocess.run([sys.executable, str(ROOT / "calibrate_thresholds.py")],
                       check=False)
    if not CALIB.exists():
        sys.exit("calibration did not produce a report; refusing to run")
    c = json.loads(CALIB.read_text(encoding="utf-8"))
    if not c.get("all_probes_separated"):
        if not force:
            sys.exit("REFUSING TO RUN: the perceptual thresholds do not separate "
                     "the calibration probes. Fix the oracle, not the results. "
                     "(--force-uncalibrated to override, which invalidates the run)")
        print("WARNING: running against uncalibrated thresholds. "
              "These results are not reportable.\n")
    return c


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor(), "numpy": None,
           "torch": None, "cuda_device": None, "triton": None}
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        pass
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["offline", "live"], default="offline")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--no-advisor", action="store_true",
                    help="single-agent ablation: coder retries without critique")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force-uncalibrated", action="store_true")
    args = ap.parse_args()

    calib = calibration_gate(args.force_uncalibrated)
    task_ids = args.tasks or tasks.TASK_ORDER
    for t in task_ids:
        tasks.get(t)  # fail fast on a typo

    provider = make_provider(args.provider)
    offline = isinstance(provider, OfflineProvider)
    use_advisor = not args.no_advisor

    if offline and args.no_advisor:
        print("NOTE: --no-advisor has no effect in offline mode -- the replay "
              "pool does not branch on critique, so the ablation is not "
              "measurable without a live model. Reporting it as unsupported.\n")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or f"{args.provider}-{stamp}"
    out_dir = ROOT / "results" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"RenderBench-mini  |  provider={provider.name}  model={provider.model}")
    print(f"prompts={PROMPT_VERSION}  seed={args.seed}  max_rounds={args.rounds}  "
          f"advisor={'on' if use_advisor else 'off'}")
    print(f"run -> results/runs/{tag}\n")

    print("Measuring naive baselines (cached)...")
    for t in task_ids:
        print(f"  {t:20} {baseline_ms(t, args.seed):8.2f} ms")
    print()

    runs, t_start = [], time.perf_counter()
    for t in task_ids:
        spec = tasks.get(t)
        print(f"[{t}] {spec.title}  ({spec.oracle.mode} oracle)")
        r = run_task(t, provider, max_rounds=args.rounds,
                     use_advisor=use_advisor, log_dir=out_dir,
                     seed=args.seed if args.seed != DEFAULT_SEED else None)
        runs.append(r)
        print()

    wall = time.perf_counter() - t_start
    solved = [r for r in runs if r.solved]
    speedups = [r.best_speedup for r in solved if r.best_speedup]
    geo = (float(__import__("math").exp(
        sum(__import__("math").log(s) for s in speedups) / len(speedups)))
        if speedups else 0.0)

    summary = {
        "tag": tag, "timestamp_utc": stamp,
        "provider": provider.name, "model": provider.model,
        "pinned_model_setting": PINNED_MODEL,
        "prompt_version": PROMPT_VERSION,
        "seed": args.seed, "max_rounds": args.rounds,
        "advisor": use_advisor,
        "advisor_ablation_supported": (not offline),
        "offline_disclaimer": (
            "provider=offline replays a fixed pool of human-written kernels. "
            "All correctness verdicts and timings are real measurements of that "
            "code by the frozen judge, but the run measures the HARNESS, not any "
            "model's capability." if offline else None),
        "environment": environment(),
        "calibration": {"all_probes_separated": calib.get("all_probes_separated"),
                        "thresholds": calib.get("thresholds"),
                        "ssim_margin": calib.get("ssim_margin"),
                        "caught_only_by_caps": calib.get("caught_only_by_caps")},
        "wall_s": round(wall, 1),
        "n_tasks": len(runs), "n_solved": len(solved),
        "geomean_speedup": round(geo, 3),
        "total_cost_usd": round(sum(r.cost_usd for r in runs), 4),
        "tasks": [r.to_dict() for r in runs],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")

    # ---- table --------------------------------------------------------
    print("=" * 78)
    print(f"{'task':22}{'oracle':12}{'solved':>7}{'rnd':>5}"
          f"{'naive ms':>10}{'best ms':>10}{'speedup':>9}")
    print("-" * 78)
    for r in runs:
        spec = tasks.get(r.task_id)
        print(f"{r.task_id:22}{spec.oracle.mode:12}"
              f"{('yes' if r.solved else 'NO'):>7}"
              f"{(r.rounds_to_first_pass if r.solved else '-'):>5}"
              f"{r.baseline_ms:>10.2f}"
              f"{(f'{r.best_ms:.2f}' if r.best_ms else '-'):>10}"
              f"{(f'{r.best_speedup:.2f}x' if r.best_speedup else '-'):>9}")
    print("-" * 78)
    print(f"solved {len(solved)}/{len(runs)}   geomean speedup {geo:.2f}x   "
          f"wall {wall:.0f}s   cost ${summary['total_cost_usd']:.4f}")
    if offline:
        print("\nprovider=offline: measures the harness, not a model. "
              "Use --provider live with RB_API_KEY for capability numbers.")
    print(f"\nwrote results/runs/{tag}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
