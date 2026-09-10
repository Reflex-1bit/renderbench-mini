"""The loop controller: coder -> judge -> advisor -> coder, with a hard round cap.

Every intermediate kernel, prompt, raw model response, judge verdict and critique
is written to disk. A run that is not fully logged is not a result.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .. import tasks
from ..judge import JudgeResult, baseline_ms, judge
from . import prompts
from .llm import (BudgetExceeded, Completion, OfflineProvider, Provider, Usage,
                  extract_kernel)


@dataclass
class Round:
    index: int
    kernel: str | None
    coder_note: str
    judge: dict
    critique: str
    usage: dict
    wall_s: float
    parse_failed: bool = False


@dataclass
class TaskRun:
    task_id: str
    provider: str
    model: str
    use_advisor: bool
    max_rounds: int
    rounds: list[Round] = field(default_factory=list)
    baseline_ms: float = 0.0
    best_round: int | None = None
    best_ms: float | None = None
    best_speedup: float | None = None
    best_kernel: str | None = None
    solved: bool = False
    rounds_to_first_pass: int | None = None
    cost_usd: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["rounds"] = [asdict(r) for r in self.rounds]
        return d


# --------------------------------------------------------------------------
# Advisor
# --------------------------------------------------------------------------


def offline_critique(jr: JudgeResult, kernel: str) -> str:
    """Derive a critique from the judge's own measurements.

    Used when no model is available. This is rule-based, not a model talking --
    but every number in it is a real measurement from the run, so the loop's
    control flow is exercised for real.
    """
    o = jr.oracle.get("metrics", {}) if jr.oracle else {}
    if jr.stage == "screen":
        return f"Submission rejected before execution: {jr.error}"
    if jr.stage == "compile":
        return ("The submission did not compile. Fix the syntax and make sure "
                "exactly one top-level `kernel` function is defined.")
    if jr.stage == "run":
        last = jr.error.strip().splitlines()[-1][:200]
        return (f"The kernel raised at runtime: {last}. This is a shape or "
                "indexing contract violation, not a performance issue -- re-read "
                "the required behaviour at the array bounds before optimising.")
    if jr.stage == "contract":
        return (f"{jr.error} The judge re-checks every input array after the "
                "call; write into a fresh buffer.")
    if jr.stage == "correctness":
        mode = jr.oracle.get("mode", "?")
        if mode == "exact":
            frac = o.get("mismatch_frac", 0.0)
            return (f"Output is not bit-identical: {frac:.1%} of elements differ. "
                    f"{jr.error} A mismatch this broad points at the arithmetic "
                    "width or the rounding rule rather than an edge case.")
        if mode == "tolerance":
            return (f"Exceeds the allowed tolerance: max abs error "
                    f"{o.get('max_abs_err', 0):.4g} against a cap of "
                    f"{o.get('atol', 0):g}, over {o.get('over_tol_frac', 0):.2%} "
                    "of elements. The deviation is systematic, so check the "
                    "formula and the sampling convention, not the precision.")
        if mode == "perceptual":
            bits = []
            if o.get("ssim", 1) < 0.99:
                bits.append(f"SSIM {o['ssim']:.4f} is below the 0.990 floor")
            if o.get("max_pixel_err", 0) > 0.06:
                bits.append(f"worst pixel is off by {o['max_pixel_err']:.3f} "
                            "(cap 0.060)")
            if o.get("bad_frac", 0) > 0.02:
                bits.append(f"{o['bad_frac']:.2%} of pixels exceed the per-pixel "
                            "cap (limit 2%)")
            return ("Structurally wrong output: " + "; ".join(bits) + ". Errors "
                    "concentrated at glyph edges mean the sub-pixel offset is "
                    "being discarded rather than resampled.")
        return f"Correctness failure: {jr.error}"

    sp = jr.speedup or 1.0
    return (f"Correct, and {sp:.2f}x the naive baseline "
            f"({jr.timing:.2f} ms vs {jr.baseline_ms:.2f} ms). Remaining headroom "
            "is in memory traffic: look at the dtype width of the intermediates, "
            "how many full-size temporaries are allocated per call, and whether "
            "any per-element work can be hoisted out of a Python-level loop or "
            "precomputed at module scope.")


def live_critique(provider: Provider, task, kernel: str,
                  jr: JudgeResult) -> tuple[str, Usage]:
    timing_line = ""
    if jr.passed:
        timing_line = (f"Timing: {jr.timing:.3f} ms median vs "
                       f"{jr.baseline_ms:.3f} ms naive baseline "
                       f"({jr.speedup:.2f}x).")
    user = prompts.ADVISOR_USER.format(
        task_block=task.prompt_block(), kernel=kernel,
        verdict="PASS" if jr.passed else "FAIL",
        stage=jr.stage, detail=(jr.error or jr.oracle.get("detail", "")),
        timing_line=timing_line)
    # 150-word critique in prose, but reasoning models (GLM-5.3 included) spend
    # tokens on hidden reasoning_content before any visible content, so the
    # budget has to cover that too -- 400 was tuned for a non-reasoning model
    # and starved this call on the free endpoint.
    c = provider.complete(prompts.ADVISOR_SYSTEM, user, max_tokens=1500)
    return c.text.strip(), c.usage


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


def run_task(task_id: str, provider: Provider, max_rounds: int = 4,
             use_advisor: bool = True, log_dir: Path | None = None,
             seed: int | None = None, verbose: bool = True) -> TaskRun:
    task = tasks.get(task_id)
    base = baseline_ms(task_id)
    offline = isinstance(provider, OfflineProvider)

    run = TaskRun(task_id=task_id, provider=provider.name, model=provider.model,
                  use_advisor=use_advisor, max_rounds=max_rounds,
                  baseline_ms=base)

    critique = ""
    for i in range(max_rounds):
        t0 = time.perf_counter()
        usage = Usage()
        parse_failed = False

        # ---- coder ------------------------------------------------------
        if offline:
            if provider.exhausted(task_id) and i > 0:
                break
            kernel, note = provider.next_for(task_id)
        else:
            if i == 0:
                user = prompts.CODER_USER.format(
                    task_block=task.prompt_block(),
                    naive_source=task.naive_source, baseline_ms=base)
            else:
                user = (prompts.CODER_USER.format(
                    task_block=task.prompt_block(),
                    naive_source=task.naive_source, baseline_ms=base)
                    + "\n\n" + prompts.REVISE_USER.format(critique=critique))
            try:
                c: Completion = provider.complete(prompts.CODER_SYSTEM, user)
                usage = c.usage
                note = "model response"
                kernel = extract_kernel(c.text)
                if kernel is None:
                    parse_failed = True
            except BudgetExceeded as e:
                if verbose:
                    print(f"    budget stop: {e}")
                break
            except Exception as e:  # noqa: BLE001
                # A provider-level failure (timeout after retries, malformed
                # response, empty content with nothing usable in
                # reasoning_content) is a failed round for this task, not a
                # reason to abort every other task in the run.
                note = f"provider call failed: {e}"
                kernel = None
                parse_failed = True

        # ---- judge ------------------------------------------------------
        if kernel is None:
            reason = (note if (not offline and note.startswith("provider call failed"))
                      else "could not extract a `kernel` function from the "
                           "model response")
            jr = JudgeResult(False, reason, None, "parse", task_id)
        else:
            jr = judge(kernel, task_id, **({"seed": seed} if seed else {}))

        # ---- advisor ----------------------------------------------------
        # A live-provider hiccup here (empty content, transient HTTP error
        # after retries exhausted, etc.) should degrade this round's critique,
        # not abort the run and lose every round already completed -- hence
        # the broad catch. BudgetExceeded still gets its own branch so a hard
        # cost cap is never silently swallowed.
        if use_advisor and i < max_rounds - 1:
            if offline:
                critique = offline_critique(jr, kernel or "")
            else:
                try:
                    critique, au = live_critique(provider, task, kernel or "", jr)
                    usage = Usage(usage.prompt_tokens + au.prompt_tokens,
                                  usage.completion_tokens + au.completion_tokens)
                except BudgetExceeded:
                    critique = ""
                except Exception as e:  # noqa: BLE001
                    critique = f"[advisor call failed: {e}]"
        else:
            critique = ""

        rd = Round(index=i, kernel=kernel,
                   coder_note=note if offline else "",
                   judge=jr.to_dict(), critique=critique,
                   usage={"prompt_tokens": usage.prompt_tokens,
                          "completion_tokens": usage.completion_tokens,
                          "cost_usd": round(usage.cost_usd, 6)},
                   wall_s=round(time.perf_counter() - t0, 3),
                   parse_failed=parse_failed)
        run.rounds.append(rd)
        run.cost_usd += usage.cost_usd

        if verbose:
            print(f"    round {i}: {jr.summary()}")

        if jr.passed:
            if run.rounds_to_first_pass is None:
                run.rounds_to_first_pass = i
            run.solved = True
            if run.best_ms is None or jr.timing < run.best_ms:
                run.best_ms = jr.timing
                run.best_speedup = jr.speedup
                run.best_round = i
                run.best_kernel = kernel

        # Persist after every round, not only at the end -- a crash on round 3
        # should not cost the log of rounds 0-2 that already ran and cost money.
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{task_id}.json").write_text(
                json.dumps(run.to_dict(), indent=2), encoding="utf-8")
            if run.best_kernel:
                (log_dir / f"{task_id}.best.py").write_text(
                    run.best_kernel, encoding="utf-8")

        if offline and provider.exhausted(task_id):
            break
    return run
