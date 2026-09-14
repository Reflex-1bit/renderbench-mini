# Preregistration

Per Part IX of the build guide: if a decision is not here, it is not decided. Record each
before 25 September. This file is frozen alongside Freeze 1.

| # | Decision | Status | Notes |
|---|---|---|---|
| 1 | Who writes the AI pack | **Open — needs you** | Guide recommends GLM-5.3 writes it, per §5.5; otherwise arm A1 is empty and the paper loses its central comparison |
| 2 | MLSys 2027 deadline, style file | **Deadline resolved** | Oct 30 '26, 12:00 PM PDT — confirmed live 2026-09-14, see citations.md. Style file/page limit/anonymization rules still not posted; re-check closer to Oct 10 |
| 3 | Loop inside DeepSeek Harness or plain Python | **Open — needs you** | Guide recommends Harness for the full reasoning log, one pinned version, minimal tool surface |
| 4 | University GPU time | **Open — needs you, highest lead time** | Email this week. Decides whether RQ3b is a demonstration or a ceiling |
| 5 | Profiling frequency | **Recommended: final round only** | Per guide — Nsight Compute profiling dominates CudaForge's runtime (10-12 of ~26.5 min); don't pay that cost every round |
| 6 | Endpoint | **Recommended: Z.ai direct, not tokenrouter free tier** | renderbench-mini's live GLM tests ran on tokenrouter's `z-ai/glm-5.3-free`, an unverified third-party free proxy, explicitly disclaimed as unstable. That endpoint also produced a ~200-minute hang before a wall-clock deadline fix existed. Any reported GLM number must come from Z.ai direct |
| 7 | reasoning_effort | **Open — needs you** | Guide recommends max for main runs + a high-effort cost ablation. Note: renderbench-mini's own reasoning-token-starvation bug (6,613 completion tokens -> 254 chars of kernel) was found on the free endpoint; unclear if it reproduces on Z.ai direct at any effort level — worth checking early |
| 8 | Oracle thresholds | **Recommended: reuse renderbench-mini's calibrated approach** | Set from calibration probes before results, per calibrate_thresholds.py's existing design — already validated (5/5 probes separated) in the mini version |

## What's frozen so far
Nothing yet. Freeze 1 (judge, oracles, thresholds, timing, task specs, human pack, prompts)
is scheduled for 25 September per the guide's Phase 2.
