# Lab Notebook

Append-only. Four lines per entry: what I did, what I found, what broke, what is next.
This file becomes the methods section in October — write it like someone else will read it.

---

## 2026-09-14

**Did:** Read through the v3.0 build guide in full. Checked MLSys 2027's actual pages
(mlsys.org/Conferences/2027/Dates and /CallForPapers) instead of trusting the guide's
Sept-12 snapshot or any third-party aggregator. Set up the paper repo scaffolding
(citations.md, LAB_NOTEBOOK.md, HANDOFF.md, PREREGISTRATION.md, packs/, bench/, agent/,
runs/, analysis/, paper/) at `C:\Users\adity\renderbench-paper`.

**Found:** The MLSys 2027 deadline is now live and explicit: "Paper Submission Deadline:
Oct 30 '26, 12:00 PM PDT" — a distinct labeled field, not the Oct 10 open date the guide
warned about conflating. This appears to have been posted between the guide's writing
(Sept 12) and today (Sept 14). Style file / page limit / anonymization rules still not
posted. Also confirmed: renderbench-mini's live "GLM-5.3" tests actually ran against
`z-ai/glm-5.3-free` on tokenrouter.com, a third-party free tier explicitly disclaimed as
unstable/capacity-limited — not Z.ai's direct, paid, spec-verified API. Any GLM claim in
the paper needs a re-run against Z.ai direct before it's reportable, per the guide's own
Decision #6.

**Broke:** Nothing broke today — this was reading/verification/scaffolding, not code.

**Next:** Open Decision #2 (MLSys deadline) is resolved, record it in PREREGISTRATION.md.
Still open and user-blocked: university GPU access email, Z.ai account/credits, the §5.5
AI-pack-authorship decision. Still Claude-executable and not yet started: reading the four
rendering-pack primary sources into citations.md, Gate 0 Tier 1 (Glyph render/rasterize/crop
timing, no GPU needed), and confirming whether Triton needs WSL2 at all — renderbench-mini
already proved Triton runs natively on Windows via `triton-windows` (no WSL2), which may
let Phase 1's GPU setup skip a step the guide assumes is required. Worth re-checking whether
Glyph's own dependencies (reportlab, pdf2image, poppler) also run natively on Windows before
committing to the WSL2 requirement wholesale.

## 2026-09-14 (cont'd)

**Did:** Built and live-tested a concurrent multi-provider runner
(`renderbench/agent/parallel.py` in renderbench-mini) -- dispatches independent
(task, provider) loop runs across a thread pool instead of serializing
everything through one queue.

**Found:** Real proof of concept works. 3 concurrent jobs (1 live GLM-5.3-free
call taking 152s, 2 offline jobs taking 3-4s each) completed with wall time
152.4s, not the serial sum of 159s -- the slow provider did not block the fast
ones. Architecturally this is the right shape for the planned 120-run study
(5 seeds x 6 tasks x 4 arms): dispatch independent runs across the ~6
candidate free NVIDIA NIM models found earlier, each in its own thread, so a
slow/rate-limited model never blocks progress on the others. The real
throughput win will show up once job durations are comparable across
providers (this smoke test's 152s-vs-4s gap was too lopsided to show much
speedup) -- worth re-measuring once more free-tier keys are in place.

**Broke:** Nothing. One GLM-5.3-free call came back "ok-unsolved" within the
pool test; a same-config, same-task retry solved cleanly on the very next
attempt. Consistent with the free tier's already-documented instability
(README's ~200-minute hang from earlier), not a new bug.

**Next:** Get NVIDIA_API_KEY set up (user-blocked) to actually spread the 120
planned runs across multiple distinct free models rather than one key's rate
limit. Concurrency architecture itself is done and pushed.

## 2026-09-14 (evening) — the Triton gap, found and closed

**Did:** Extended the agent loop from the CPU track to the Triton track. Up to
this point every run (offline demo, GLM live runs, the Opus 5 run, the 4-model
cross-model study) exercised the CPU track, which asks for NumPy by design —
that track is the scaffolding that runs with no GPU and no API key, used to get
the oracles, judge, calibration and loop working and reproducible first. Triton
was always the stated target language and the GPU track has had hand-written
Triton since the start; what had not been done was pointing the loop at it.
Ran a smoke test to confirm models can produce working Triton at all, then built
`triton_judge.py` (compiles a candidate, runs it on the RTX 5060, scores it with
the SAME frozen oracles as the CPU track, times it) and `triton_bench.py`
(generate → GPU judge → one round of verbatim compiler-error feedback,
concurrent).

**Found:**
- Models *can* write working Triton. deepseek-v4-flash produced structurally
  correct Triton on round 0 with one real broadcasting bug, fixed it from the
  verbatim CompilationError in 30s, and the result was bit-exact.
- 4/6 solved across 2 models × 3 tasks. Kernel-only 225x–1861x vs naive;
  end-to-end (host→device→host) 69.7x on alpha_composite, of which 0.91ms of
  0.969ms is PCIe transfer. Quote end-to-end, never kernel-only alone.
- Triton beats the best agent-written NumPy kernel by 14x–842x kernel-only,
  30.9x end-to-end on alpha_composite.
- **New axis the CPU track was blind to:** deepseek-flash 3/3 vs muse-glimmer
  1/3 on Triton, while those two models are indistinguishable on NumPy (both
  solve blit round 0, within 1ms). Model choice barely matters for host-side
  vectorisation and matters enormously for GPU kernel generation.
- 3 of the 4 solves needed exactly one round of compiler feedback — direct
  support for the judge→advisor loop design on the GPU track.

**Broke:** Nothing new. Earlier in the day, two invalid "0/4 solved" results had
to be thrown out and re-run: a 240s call deadline, and (the real root cause)
non-streaming requests, where the server sends zero bytes until generation
finishes and `requests`' between-bytes read timeout therefore fires regardless
of its value. Streaming: first chunk at 1.0s vs 0 tokens after 597s. Also
`chat_template_kwargs {"thinking": false}` — with thinking on, these models
streamed reasoning for 240s and emitted zero characters of code.

**Next:** Run `glyph_atlas_blit` on the Triton track. All four models failed it
in NumPy, and it is where GPU advantage should compress hardest (order-dependent
source-over blending forbids one-thread-per-glyph). That is the task that tests
the actual thesis rather than the easy elementwise ones. Then: the human pack
(§5), which is still the critical path and still does not exist.
