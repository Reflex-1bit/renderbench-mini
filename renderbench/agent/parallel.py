"""Concurrent multi-provider runner.

The coder->judge->advisor loop is inherently SEQUENTIAL within one task: the
advisor needs the coder's failure before it can critique, the coder needs the
critique before it revises. There is no way to parallelize rounds within one
run -- that's not a limitation of this module, it's the shape of the loop.

What genuinely doesn't depend on anything else: different (task, provider,
seed) runs. blit on model A and srgb_gamma on model B share no state, so they
can run at the same time. That's what this module actually parallelizes --
independent runs across a pool of providers, not the loop itself.

Each job gets its own thread. Jobs sharing one provider still queue behind
that provider's real rate limit (that's physics, not something Python
concurrency changes) -- the win is that a slow/rate-limited provider no
longer blocks a *different* provider's job from making progress.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .llm import Provider
from .loop import TaskRun, run_task


@dataclass
class PoolJob:
    label: str                    # human-readable id, e.g. "glm-free/blit"
    task_id: str
    provider: Provider
    max_rounds: int = 3
    use_advisor: bool = True
    seed: int | None = None


@dataclass
class PoolResult:
    label: str
    task_id: str
    provider_name: str
    wall_s: float
    ok: bool
    error: str = ""
    run: TaskRun | None = None


def run_pool(jobs: list[PoolJob], log_dir: Path | None = None,
            max_workers: int | None = None) -> list[PoolResult]:
    """Run every job concurrently. Returns results in completion order (not
    submission order) so you can watch the fast ones land before the slow
    ones -- useful signal on its own about which provider is actually usable.
    """
    results: list[PoolResult] = []
    t0 = time.perf_counter()

    def _one(job: PoolJob) -> PoolResult:
        j0 = time.perf_counter()
        try:
            run = run_task(job.task_id, job.provider, max_rounds=job.max_rounds,
                          use_advisor=job.use_advisor,
                          log_dir=(log_dir / job.label) if log_dir else None,
                          seed=job.seed, verbose=False)
            return PoolResult(job.label, job.task_id, job.provider.name,
                             time.perf_counter() - j0, True, run=run)
        except Exception as e:  # noqa: BLE001
            return PoolResult(job.label, job.task_id, job.provider.name,
                             time.perf_counter() - j0, False, error=repr(e))

    with ThreadPoolExecutor(max_workers=max_workers or len(jobs)) as ex:
        futures = {ex.submit(_one, job): job for job in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            status = "solved" if (r.ok and r.run and r.run.solved) else (
                "ok-unsolved" if r.ok else "FAILED")
            print(f"  [{r.wall_s:6.1f}s] {r.label:24} ({r.provider_name}) {status}"
                 + (f": {r.error[:80]}" if r.error else ""))

    wall = time.perf_counter() - t0
    serial_est = sum(r.wall_s for r in results)
    print(f"\npool wall time: {wall:.1f}s  |  sum of job times: {serial_est:.1f}s  "
         f"|  speedup from concurrency: {serial_est / wall:.2f}x")
    return results
