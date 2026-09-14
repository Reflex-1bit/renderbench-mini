# Cross-model failure taxonomy (cross-v3, 2026-09-14)

Four models, NVIDIA NIM free endpoints, same two tasks, same frozen oracle.
All failures below are genuine model output — every infrastructure timeout was
eliminated first (see "What had to be fixed" at the bottom).

## The solve-rate cliff

| Model | `blit` (exact oracle) | `glyph_atlas_blit` (perceptual oracle) |
|---|---|---|
| `deepseek-v4-flash` | solved r0 — **15.79x** | failed |
| `kimi-k3` | solved r0 — **17.62x** | failed |
| `muse-glimmer-30b` | solved r0 — **16.42x** | failed |
| `nemotron-3-ultra-550b` | solved r0 — **16.62x** | failed |

**4/4 on the elementwise exact-oracle task, first attempt, clustered in a tight
15.8–17.6x band. 0/4 on the order-dependent perceptual rendering task.** The
cliff is task-shaped, not model-shaped — a 550B frontier MoE and a 30B model
land on the same side of it, twice.

## The failure modes are nameable, and they match the calibration probes

Calibration probes (synthetic, written before any model ran):

| Probe | SSIM | max_px | bad_frac |
|---|---|---|---|
| `supersample-8x` (valid, 1/8 px phase) | 0.99928 | 0.0824 | 0.00000 |
| `supersample-4x` (too coarse, 1/4 px) | 0.99730 | 0.1765 | 0.00045 |
| `nearest-snap` (phase discarded) | 0.71552 | 1.0000 | 0.11023 |
| `drop-one-glyph` (silent content loss) | 0.99930 | 0.7686 | 0.00019 |

What the models actually produced:

| Model | SSIM | max_px | bad_frac | Matches probe | Diagnosis |
|---|---|---|---|---|---|
| `nemotron-ultra` r0 **and** r1 | 0.99737 | 0.1765 | 0.00035 | `supersample-4x` (0.99730 / 0.1765 / 0.00045) | **quarter-pixel phase quantisation** — too coarse by exactly the margin the oracle was calibrated to reject. Identical across both rounds: the advisor critique moved it not at all. |
| `kimi-k3` r0 | 0.74583 | 0.9725 | 0.11053 | `nearest-snap` (0.71552 / 1.0000 / 0.11023) | **sub-pixel phase discarded entirely** — `bad_frac` matches the probe to three decimals |
| `deepseek-flash` r0 | 0.99868 | 0.6627 | 0.00062 | `drop-one-glyph` shape | **localised catastrophic error** — see below |

## The headline: a real model, in the wild, beat SSIM

`deepseek-v4-flash` round 0 scored **SSIM 0.99868 against a 0.998 floor — it
PASSED the structural check** — and was rejected solely by the bounded
per-pixel cap (0.6627 against a 0.1 cap).

This is the synthetic `drop-one-glyph` result reproduced by an actual model
solving an actual task, not by a probe built to make the point. An SSIM-only
oracle — the obvious choice, and the one every general IQA pipeline reaches
for — would have **accepted this kernel**.

That is three independent code generators now producing failures the
conjunctive oracle catches and a structural metric alone does not:
a hand-written Triton kernel (box-boundary overshoot), Claude Opus 5 (the same
overshoot, `(gh+1, gw+1)` instead of `(gh, gw)`), and now deepseek-v4-flash.

## What had to be fixed to get an honest number

Two earlier runs produced "0/4 solved" that was **not a capability result** and
would have been wrong to report:

1. **240s deadline** — 7 of 8 glyph failures were this firing, not the models.
2. **Non-streaming requests** — the real root cause. With `stream=False` the
   server sends zero bytes until generation completes, and `requests`' read
   timeout measures the gap *between* bytes, so it tripped every time no matter
   how high it was set. Measured on the same endpoint, same prompt, same
   minute: non-streaming returned **0 tokens after 597s**; streaming delivered
   its first chunk in **1.0s**. This also retroactively explains the ~200-minute
   hang recorded against a different free endpoint earlier in this project.
3. **Hidden reasoning consuming the entire budget** — with `thinking` enabled
   these models streamed reasoning for 240s and emitted **zero** characters of
   code. With `chat_template_kwargs: {"thinking": false}`: **37s, 2,192
   characters, first character at 4s.**

Only after all three were fixed did the glyph failures become `correctness` /
`compile` / `run` — genuine model output — rather than transport errors.

---

# Triton track (2026-09-14, same day, later)

## Everything above this line was numpy

Every result in this document up to here — the cliff, the taxonomy, the
cross-model table — was produced by models writing **numpy**, because
`CODER_SYSTEM` says "You may use numpy" and never once mentions Triton, GPU, or
CUDA. No model in this project had ever been asked for a GPU kernel.

## Asking for Triton instead, same models, same tasks

`triton_judge.py` compiles the candidate, runs it on the RTX 5060, and scores it
with the **same frozen oracles** as the CPU track. 4/6 solved.

| Task | naive | agent numpy | agent Triton (kernel-only) | Triton vs numpy |
|---|---|---|---|---|
| `blit` | 11.32 ms | 0.72 ms (15.7x) | **0.050 ms — 225x** | 14x |
| `alpha_composite` | 65.94 ms | 29.98 ms (2.2x) | **0.036 ms — 1852x** | 842x |
| `srgb_gamma` | 87.28 ms | 18.71 ms (4.7x) | **0.047 ms — 1861x** | 399x |

**End-to-end** (host→device→host, the honest number) for `alpha_composite`:
**0.969 ms = 69.7x** vs naive, **30.9x** vs the best numpy kernel. Transfer is
0.91 ms of that 0.969 ms — the compute is essentially free at this size, so
kernel-only figures above should never be quoted alone.

## The finding the numpy track was blind to

| Model | numpy | Triton |
|---|---|---|
| `deepseek-v4-flash` | solved `blit` r0 | **3/3** |
| `muse-glimmer-30b` | solved `blit` r0 | **1/3** |

On numpy these two are indistinguishable — both solve `blit` on the first
attempt, within 1 ms of each other. Point them at Triton and they separate
hard. Model choice barely matters for host-side vectorisation and matters
enormously for GPU kernel generation; a CPU-only benchmark cannot see this axis
at all.

`muse-glimmer` failures were genuine: `blit` never produced a usable code block
(parse), `srgb_gamma` failed at `run` twice with real Triton compiler errors fed
back as feedback.

## Compile-error feedback works

`deepseek-flash` on `alpha_composite`, hand-run earlier: round 0 produced
structurally correct Triton (`@triton.jit`, correct grid/launch, masked
load/store, right algorithm) with one real broadcasting bug — `offsets` not
reshaped to `[:, None]` before combining with a `[1,4]` channel index, while
correctly doing exactly that for `mask` one line below. Given the verbatim
`CompilationError`, it fixed the precise line in 30 s / 695 tokens and
recompiled to a **bit-exact** result.

`srgb_gamma` (deepseek) and `alpha_composite` (muse-glimmer) both also solved on
round 1 after a failed round 0 — so 3 of the 4 solves needed exactly one round
of real compiler feedback.
