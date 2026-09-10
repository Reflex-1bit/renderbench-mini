# RenderBench-mini

A working, scaled-down implementation of the RenderBench project plan: a
benchmark and multi-agent loop for **LLMs writing GPU rendering kernels**.

The full plan targets 8 tasks, Triton on datacentre GPUs, and GLM-5.3 across an
8-week schedule. This is the part that runs today, on one laptop, in about 30
seconds — built to make the *methodology* concrete and testable well before the
model spend starts.

```bash
python calibrate_thresholds.py     # gate: do the oracle thresholds actually work?
python run_bench.py                # 6 tasks, agent loop, offline replay
python gpu_track.py                # the same tasks as real Triton kernels
python make_figures.py             # figures for the writeup
```

Nothing here needs an API key. `run_bench.py --provider live` switches the same
loop onto GLM-5.3.

---

## What actually got built

| Plan item | Status |
|---|---|
| Task specs | **6 of 8** — descope ladder drops `atlas_lookup`, `tiled_multipage_pack` |
| Naive baselines | 6/6, timed and cached as the speedup denominator |
| Correctness oracles | exact / tolerance / perceptual, all three in use |
| Threshold justification | `calibrate_thresholds.py`, **run before any result**, blocks the runner if it fails |
| Judge module | subprocess-isolated, frozen, source-screened |
| Timing methodology | warmup + median-of-N + IQR + CV, declared per task |
| Coder→judge→advisor→coder loop | working, round-capped, fully logged |
| Response parsing | 3-tier fallback, parse failures recorded not swallowed |
| Full logging | every kernel, prompt, verdict, critique, token count |
| Baselines: naive, single-agent, coder+advisor | naive ✅, coder+advisor ✅ (live, both models) -- single-agent (`--no-advisor`) not yet run live |
| Triton kernels on real hardware | ✅ RTX 5060, sm_120 |
| Gate 0 (Glyph render-vs-encode profile) | **not done** — needs Glyph installed |
| Expert baseline (Slug) | **not done** |
| RQ3 Glyph pipeline swap-in | **not done** |

---

## Results

### CPU track — the agent loop, offline replay

6/6 tasks solved, geomean **3.22x** over the naive reference.

| task | oracle | first pass | naive | best | speedup |
|---|---|---|---|---|---|
| `blit` | exact | round 1 | 12.19 ms | 0.94 ms | **13.01x** |
| `alpha_composite` | exact | round 1 | 69.14 ms | 24.52 ms | **2.82x** |
| `srgb_gamma` | tolerance | round 1 | 85.07 ms | 19.20 ms | **4.43x** |
| `bilinear_resize` | tolerance | round 1 | 145.28 ms | 55.70 ms | **2.61x** |
| `glyph_atlas_blit` | perceptual | round 1 | 45.39 ms | 27.66 ms | **1.64x** |
| `text_page_raster` | perceptual | round 1 | 182.81 ms | 114.51 ms | **1.60x** |

The gradient is the interesting part, and it is the thesis in miniature: the
elementwise, exactly-checkable tasks give up 2.6–13x, while the two branch-heavy,
order-constrained glyph tasks — the ones that are actually *rendering* — give up
1.6x. Speedups fall off exactly where the task stops looking like a math kernel.

### GPU track — real Triton kernels, RTX 5060 (sm_120)

Two numbers per kernel, and the difference matters. **Kernel-only** times just
the launch with inputs already resident on the GPU and the output left there —
what a fused pipeline stage would see. **End-to-end** times host array in,
kernel, host array out, on every call — what calling this kernel from
CPU-resident code (e.g. handed a numpy page, must return a numpy page)
genuinely costs, transfer included. An earlier version of this table reported
kernel-only numbers as if they were end-to-end and overstated `alpha_composite`
by **25x** (1593x claimed vs. 63x measured honestly at the time); both
`gpu_track.py` and the numbers below now report both, always.

| kernel | oracle | naive CPU | kernel-only | end-to-end (honest) |
|---|---|---|---|---|
| `alpha_composite` | exact — bit-identical | 65.94 ms | 0.051 ms → 1283x | 1.337 ms → **49x** |
| `glyph_atlas_blit` TILE=16 | perceptual — pass | 43.83 ms | 0.424 ms → 103x | 1.205 ms → 36x |
| `glyph_atlas_blit` TILE=32 | perceptual — pass | 43.83 ms | 0.164 ms → 268x | 0.932 ms → 47x |
| `glyph_atlas_blit` TILE=64 | perceptual — pass | 43.83 ms | 0.148 ms → 296x | 0.723 ms → **61x** |

Geomean: kernel-only 320x, **end-to-end 47.5x**. For an op this cheap, PCIe
transfer dominates completely — `alpha_composite`'s transfer overhead alone
(1.29 ms) is 25x its own compute time (0.051 ms). The honest number is still a
real, substantial speedup; it's just not the eye-catching one.

Source-over blending is not associative, so glyphs must be applied in submission
order — which rules out the obvious one-thread-per-glyph scatter. The kernel is a
**tiled rasteriser**: each program owns an output tile and walks the glyph list in
order, scalar-rejecting glyphs whose box misses its tile.

---

## The methodology bits worth arguing about

### The perceptual oracle is not SSIM

Anti-aliased coverage at a fractional offset has no bit-reproducible answer, so
the glyph tasks need tolerance. But **SSIM alone does not work**, and the
calibration measured exactly how badly:

| probe | verdict | SSIM | max pixel err |
|---|---|---|---|
| `bilinear-f32` | must pass | 1.00000 | 0.004 |
| `supersample-8x` | must pass | **0.99928** | 0.082 |
| `supersample-4x` | must fail | 0.99730 | 0.176 |
| `nearest-snap` | must fail | 0.71552 | 1.000 |
| `drop-one-glyph` | must fail | **0.99930** | 0.769 |

A renderer that **silently drops a glyph** scores SSIM 0.99930 — *higher* than a
perfectly valid supersampled implementation at 0.99928. SSIM does not merely fail
to separate them; it ranks them backwards. Any SSIM-only oracle would prefer the
renderer that loses text, which for a VTC pipeline is the worst possible failure,
because the text is the entire payload.

So the oracle is a conjunction: **SSIM floor ∧ bounded worst pixel ∧ bounded bad-
pixel fraction**. The per-pixel cap is what actually catches the dropped glyph.

![probe comparison](results/figures/fig_probes.png)

Bottom row is the error map. The `drop-one-glyph` column looks identical to the
reference by eye — its single red `z` is the whole defect.

### The admissible set is a property, not an algorithm

Thresholds were not hand-tuned until things passed. They are fitted to a stated
rule — *an implementation passes if it resolves sub-pixel phase to 1/8 px or
finer and loses no coverage* — and `calibrate_thresholds.py` verifies that rule
separates all five probes. Phase resolution maps to error monotonically:

| supersample | phase | SSIM | max px |
|---|---|---|---|
| 2x | 1/2 px | 0.98953 | 0.326 |
| 4x | 1/4 px | 0.99730 | 0.176 |
| 8x | 1/8 px | 0.99928 | 0.082 |
| 16x | 1/16 px | 0.99980 | 0.039 |

`run_bench.py` **refuses to run** if the calibration does not separate the probes.
You cannot produce a number against a broken oracle without passing
`--force-uncalibrated`, which stamps the run as unreportable.

### The oracle earns its keep immediately

The first version of the Triton glyph rasteriser scored **SSIM 0.99999** and was
still wrong: it bled coverage one row above and one column left of each glyph's
box, because the four bilinear taps were not gated on the glyph box. Error: ~36
code values on boundary pixels. Caught by the per-pixel cap, invisible to the
structural floor — the same failure mode as `drop-one-glyph`, found in real code
within an hour of the oracle existing.

### Anti-reward-hacking

Per the plan's Sakana lesson, the judge is frozen and defended:

- static source screen before execution — rejects candidates that import the
  harness, read the cached ground truth, call the reference, or touch
  subprocess/sockets
- every input array is re-checked after the call; in-place mutation fails
- ground truth is computed once from the reference and cached, so every candidate
  in every round is scored against byte-identical targets
- **median**, not min — min-of-N is how GPU benchmarks overstate speedups
- candidates run in a subprocess with a timeout, so a hang or segfault is a
  structured failure rather than a lost run

---

## Honest limits

- **Offline mode measures the harness, not a model.** `--provider offline`
  replays a fixed pool of human-written kernels. The correctness verdicts,
  timings and critiques are real measurements of that code by the frozen judge,
  but the loop's *choice* of what to write next is scripted. Every artefact is
  stamped `provider=offline`. Do not report these as capability numbers.
- **The advisor ablation is not measurable offline.** The replay pool does not
  branch on critique, so single-agent and coder+advisor would be identical by
  construction. `--no-advisor` says so and refuses to pretend. Needs a live model.
- **CPU "speedups" are vectorisation**, not GPU optimisation. The GPU track is
  where the real claim lives.
- **Gate 0 has not been run.** The RQ3 cost-savings claim is still unsupported —
  it needs Glyph profiled locally for render-vs-encode-vs-decode share. The 306x
  render speedup is only interesting if render is a meaningful slice of the
  pipeline, and nothing here establishes that.
- **No Slug comparison**, so there is no expert ceiling yet. "306x over a naive
  Python reference" is a much weaker statement than "within N% of Slug".
- Single machine, single seed for reported numbers; laptop GPU with thermal
  variance (CV is reported per measurement — the 16-22% CV on the fastest kernels
  is real and worth fixing with CUDA events before publication).
- **Run-to-run variance is visible at this scale.** The table above is the `demo`
  run (geomean 3.22x). An immediate repeat from a cleared cache gave 3.11x, with
  per-task speedups moving by up to 0.23x. Nothing here is precise to two
  significant figures, and the CPU numbers should be read as "roughly this
  shape", not as measurements. Multi-seed and multi-repeat aggregation is
  required before any of this is reportable.

---

## Layout

```
renderbench/
  core.py       oracles (SSIM impl), timing methodology, TaskSpec   [frozen]
  tasks.py      6 task specs + references + input generators
  judge.py      judge(kernel_code, task_id) -> JudgeResult          [frozen]
  _worker.py    subprocess judge worker
  pool.py       offline replay candidates
  agent/
    prompts.py  coder + advisor prompts, version-stamped
    llm.py      OpenAI-compatible live provider + offline provider
    loop.py     the controller
calibrate_thresholds.py   oracle calibration gate
run_bench.py              CPU track runner
gpu_track.py              Triton kernels
make_figures.py           figures
```

The interface the plan says to agree on before either lane writes code:

```python
judge(kernel_code, task_id) -> {"pass": bool, "error": str, "timing": float}
```

`JudgeResult.as_interface()` returns exactly that. The richer object carries the
diagnostics the advisor needs, so Lane 2 can build against the stub and swap in
the real judge without a signature change.

## Live mode

Speaks either an OpenAI-compatible `/chat/completions` endpoint (GLM-5.3
direct, an aggregator) or Anthropic's native `/messages` API
(`api.anthropic.com`), auto-detected from `RB_BASE_URL`:

```bash
export RB_API_KEY=...            # or a .env file next to run_bench.py
export RB_MODEL=glm-5.3          # pinned; logged into every run summary
export RB_BASE_URL=...           # omit for GLM-5.3's default; api.anthropic.com/v1 for Claude
export RB_BUDGET_USD=5.00        # hard cap, checked before each call
python run_bench.py --provider live --rounds 5
```

Pricing defaults to the plan's $1.40/M in, $4.40/M out (GLM-5.3); override with
`RB_PRICE_IN`/`RB_PRICE_OUT` for a different model. Cost is accumulated per
round and the provider refuses the call that would cross the cap. Every call is
also capped by a wall-clock `deadline_s` (default 240s) across all retries --
`requests`' own `timeout=` only bounds the gap between bytes received, not the
call as a whole, and one early call against a flaky free endpoint measurably
ran ~200 minutes before this existed.

### What live mode actually found, run against two real models

**GLM-5.3-free** (an aggregator's free tier, not Zhipu direct): `blit` solved
end to end -- coder wrote a correct clipped blit, the advisor gave a specific,
implementation-level critique ("the double-copy reads then immediately
overwrites the intersection"), the coder's revision followed it exactly and
introduced a real bug (dropped `import numpy as np`), and the judge caught it
cleanly. 11.5x speedup, $0.05. The wider 6-task run mostly failed on this tier,
and not from model weakness: this reasoning model can spend its entire
`max_tokens` budget on hidden `reasoning_content` before writing any visible
code (one call: 6,613 completion tokens, 254 characters of actual kernel), and
a revise-round prompt that didn't say "return the complete file, not a diff"
let it drop the import on revision. Both are harness bugs, now fixed
(`PROMPT_VERSION` bumped to `rb-mini-0.4`); the free endpoint's own instability
(one calibration call ran ~200 minutes before timing out) is not fixable from
this side.

**Claude Opus 5** (`api.anthropic.com`, real paid usage): 4/6 solved, geomean
**4.27x**, $0.76 spent.

| task | solved | speedup |
|---|---|---|
| `blit` | yes | 15.95x |
| `alpha_composite` | yes | 2.10x |
| `srgb_gamma` | yes | 2.53x |
| `bilinear_resize` | yes | 3.91x |
| `glyph_atlas_blit` | no | real bug (below) |
| `text_page_raster` | no | account ran out of credit mid-run -- untested, not a capability result |

The one real failure is worth keeping: Opus wrote a genuinely sophisticated
`glyph_atlas_blit` kernel -- in-place buffer reuse, no per-glyph allocation,
correct four-tap bilinear structure -- but sized its output write as
`(gh+1, gw+1)` instead of `(gh, gw)`, bleeding coverage one row and column past
each glyph's own box. **The identical failure class as the Triton bug found
earlier in this project**, caught the same way: invisible to SSIM (which stayed
near 1.0 on the passing cases), caught by the bounded per-pixel cap. Good
evidence the oracle generalizes across who -- or what -- writes the kernel.

## Next, in order

1. **Gate 0** — profile Glyph's render vs. encode vs. decode split. This decides
   whether RQ3 survives, and it is three days of work that gates a research
   question.
2. Re-run `text_page_raster` live once there's account credit -- it never got a
   fair attempt (ran out of balance mid-suite, not a capability result).
3. `--no-advisor` live run for the single-agent baseline, now that
   coder+advisor is proven live on two models.
4. Slug comparison on `glyph_atlas_blit` for the expert ceiling.
5. CUDA events instead of wall-clock for the GPU track.
6. Restore the two descoped tasks.
