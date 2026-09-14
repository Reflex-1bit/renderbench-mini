# citations.md

One line per claim entering the paper: the claim, the source, section/page, who checked, the date.
Nothing enters the paper without a line here. Per the build guide's rule: every number has a
source someone on this team opened, or it is gone.

| Claim | Source | Section/page | Checked by | Date |
|---|---|---|---|---|
| MLSys 2027 paper submission deadline is Oct 30 '26, 12:00 PM PDT (distinct field, labeled "Paper Submission Deadline", not the Oct 10 open date) | mlsys.org/Conferences/2027/Dates | "Paper Submission Deadline" row | Claude (fetched live) | 2026-09-14 |
| MLSys 2027 conference program dates (sessions, location specifics beyond "Bellevue, WA") not yet set | mlsys.org/Conferences/2027/Dates | "program dates have not been set yet" | Claude (fetched live) | 2026-09-14 |
| MLSys 2027 style file / page limit / anonymization rules not yet posted | mlsys.org/Conferences/2027/CallForPapers | "Details ... have not been announced yet" | Claude (fetched live) | 2026-09-14 |
| Live-mode "GLM-5.3" tests in renderbench-mini used `z-ai/glm-5.3-free` via tokenrouter.com (a third-party aggregator's free tier), NOT Z.ai's direct API. Model identity/quantization/pricing behavior of that endpoint is unconfirmed against the official $1.40/$4.40 spec. | renderbench-mini/.env (RB_MODEL, RB_BASE_URL), tokenrouter.com's own "capacity limited, stability not guaranteed" disclaimer | live-mode config | Claude + user (live session) | 2026-09-10 |
| renderbench-mini: all agent-written `.best.py` files import only numpy, never triton; `environment.triton` recorded as None in those run summaries | renderbench-mini repo, `results/runs/*/*.best.py` and `results/runs/*/summary.json` | — | Claude (built the repo) | 2026-09-09/10 |
| renderbench-mini calibration: supersample-8x (valid) scores SSIM 0.99928; drop-one-glyph (invalid, missing a letter) scores 0.99930 — ranks backwards | renderbench-mini/results/threshold_calibration.json, calibrate_thresholds.py | — | Claude (ran it) | 2026-09-09 |
| renderbench-mini GPU track: geomean kernel-only 320x, end-to-end (honest) 47.5x; earlier version overstated alpha_composite by 25x (1593x vs 63x) by reporting kernel-only as end-to-end | renderbench-mini repo, README.md, results/gpu_track.json | GPU track section | Claude (ran it, then caught and fixed the error) | 2026-09-09 |
| renderbench-mini: one live API call (GLM-5.3-free, during `live-v2-test`, a 3-round coder call on `srgb_gamma`) ran ~200 minutes (11,837s) before ReadTimeout, prior to the wall-clock deadline fix. NOTE: README's wording calls this "one calibration call" — that phrasing is imprecise; it was a live agent-loop coder round, not `calibrate_thresholds.py`. Needs tightening before it enters the paper. | renderbench-mini session log (this conversation), results/runs/live-v2-test/ | — | Claude + user | 2026-09-10 |

## Still needed (from the build guide's own unverified list, §1.5)
Not yet checked — do not cite:
- DeepSeek-OCR compression/accuracy figures
- `triton.testing.do_bench` as the correct benchmarking helper for our installed Triton version
- Nsight Compute + WSL2 counter access requirements
- DeepSeek Harness plugin API shape
- vLLM serving Glyph on one H100
- AgentKernelArena, AGAR, VTC-R1, "Correctness Illusion" (titles only, content unread)
- arXiv:2603.29010 (DSL + Speed-of-Light guidance for kernel agents)
- arXiv:2608.01848 (faithful VTC evaluation framework)
- "Visual Text Compression as Measure Transport", arXiv:2605.06708
- GLM-5.3 official spec (pricing, context window, reasoning_effort behavior) — build guide's own §1.3 claims are dated 12 Sept 2026; not independently re-verified by Claude this session, and Claude's training data predates GLM-5.3's existence entirely (knowledge cutoff January 2026), so these need re-checking against Z.ai's own docs, not taken on the guide's word or mine
- Four rendering-pack primary sources (GPU Gems ch.24, Porter & Duff, Roofline paper, Lengyel/Slug JCGT paper) — not yet fetched
