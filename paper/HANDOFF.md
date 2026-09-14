# Handoff

Paste this at the start of any new AI session. Update it at the end of every working session.
Keep under 400 words.

## Thesis
Human-authored rendering guidance (compositing, colour, sampling, performance physics),
formatted for an LLM to read, lets GLM-5.3 write GPU rendering kernels that are more often
correct and faster than the same model working from its own knowledge or from guidance it
wrote for itself — and those kernels can replace the slow drawing step inside a
visual-text-compression (VTC) pipeline (Glyph).

## The four arms
A0 no pack · A1 AI-written pack · A2 human pack · A3 human pack + advisor agent.
Spend-matched to A3's median $/task, capped at 8 rounds (A0-A2) or 3 rounds (A3).

## What's frozen
Nothing yet. Freeze 1 (judge/oracles/thresholds/timing/tasks/human pack/prompts) is
scheduled for 25 September.

## What's broken / unverified
- **The single biggest open risk**: no one has ever confirmed GLM-5.3 (or any model) can
  write a *working Triton kernel* through this project's agent loop. Every live run so far
  (in the scaled-down renderbench-mini prototype) produced pure numpy — the loop was never
  pointed at Triton. This is the 24 September smoke test and it decides whether this is a
  "kernel generation" paper or a "where generation breaks down" paper.
- renderbench-mini's live "GLM-5.3" results actually ran on tokenrouter's
  `z-ai/glm-5.3-free`, an unverified third-party free tier, not Z.ai's direct API. Not
  representative of official GLM-5.3 behavior/pricing until re-run on Z.ai direct.
- MLSys 2027 deadline is now confirmed: **Oct 30 '26, 12:00 PM PDT** (checked 2026-09-14).
  Style file and page-limit rules still not posted.

## Next step
Phase 1 setup: WSL2 + Triton on both machines (note: Triton itself already proven to run
natively on Windows without WSL2, via `triton-windows` — worth checking if Glyph's deps
(reportlab/pdf2image/poppler) also run natively before committing to WSL2 for everything).
Then: Gate 0 Tier 1 (Glyph render/rasterize/crop timing split, no GPU needed), read the four
rendering-pack primary sources into citations.md, and get Z.ai direct access sorted before
any GLM number is trusted.

## Top 3 open decisions (need the humans, not the AI)
1. Who writes the AI pack — you, or GLM-5.3 per §5.5's recommendation (if you write it, arm
   A1 is empty and the central human-vs-machine-guidance comparison disappears)
2. Harness (DeepSeek Harness) vs plain Python loop for the agent controller
3. University GPU access — email sent? This has the longest lead time of anything in the plan
