"""Calibrate and justify the perceptual oracle's thresholds.

Run this BEFORE any experiment. It measures the oracle's metrics on five probe
implementations whose verdicts we know a priori, and reports whether the declared
thresholds separate them:

  must PASS  bilinear-f32     same filter, float32 accumulation
  must PASS  supersample-8x   different algorithm, 1/8 px phase -- admissible
  must FAIL  supersample-4x   same algorithm, 1/4 px phase -- too coarse
  must FAIL  nearest-snap     sub-pixel phase discarded entirely
  must FAIL  drop-one-glyph   pixel-perfect except one glyph is missing

The last probe is the reason the oracle is not SSIM alone, and the measured
result is stronger than expected: dropping a glyph scores SSIM 0.99930, while the
*valid* supersample-8x scores 0.99928. SSIM does not merely fail to separate
them, it ranks them backwards. A structural floor on its own would therefore
prefer a renderer that silently loses text -- the worst possible failure for a
VTC pipeline, where the token content is exactly what the encoder must read. The
bounded per-pixel cap is what carries the decision.

Writes results/threshold_calibration.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from renderbench import tasks                                   # noqa: E402
from renderbench.core import PerceptualMatch, ssim              # noqa: E402
from renderbench.judge import DEFAULT_SEED                      # noqa: E402

TASK_ID = "glyph_atlas_blit"
# The admissible set is defined by a PROPERTY, not by an algorithm: an
# implementation passes if it resolves sub-pixel phase to 1/8 px or finer and
# loses no coverage. These five probes bracket that rule from both sides.
PROBES_EXPECTED = {
    "bilinear-f32":    True,   # same filter, float32 accumulation
    "supersample-8x":  True,   # different algorithm, 1/8 px phase -- admissible
    "supersample-4x":  False,  # same algorithm, 1/4 px phase -- too coarse
    "nearest-snap":    False,  # phase discarded entirely
    "drop-one-glyph":  False,  # pixel-perfect except one glyph is missing
}


def _blend(out, cov, y0, x0, gh, gw, colour):
    tile = out[y0:y0 + gh, x0:x0 + gw]
    a = cov[..., None]
    tile *= (1.0 - a)
    tile += a * colour


def probe_bilinear_f32(page, atlas, gid, gy, gx, rgb):
    """Same algorithm as the reference, accumulated in float32."""
    out = page.astype(np.float32).copy()
    _, gh, gw = atlas.shape
    for i in range(len(gid)):
        y0, x0 = int(np.floor(gy[i])), int(np.floor(gx[i]))
        fy, fx = np.float32(gy[i] - y0), np.float32(gx[i] - x0)
        pad = np.zeros((gh + 1, gw + 1), dtype=np.float32)
        pad[:gh, :gw] = atlas[gid[i]]
        cov = (pad[:gh, :gw] * (1 - fy) * (1 - fx) + pad[1:, :gw] * fy * (1 - fx)
               + pad[:gh, 1:] * (1 - fy) * fx + pad[1:, 1:] * fy * fx) / 255.0
        _blend(out, cov, y0, x0, gh, gw, rgb[i].astype(np.float32))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _probe_supersample(page, atlas, gid, gy, gx, rgb, S):
    """A genuinely different AA strategy: replicate coverage S times, shift on
    the fine grid, then box-downsample. S sets the phase resolution to 1/S px,
    which is the quantity the oracle is really gating on."""
    out = page.astype(np.float64).copy()
    _, gh, gw = atlas.shape
    for i in range(len(gid)):
        y0, x0 = int(np.floor(gy[i])), int(np.floor(gx[i]))
        fy, fx = float(gy[i] - y0), float(gx[i] - x0)
        hi = np.repeat(np.repeat(atlas[gid[i]].astype(np.float64), S, 0), S, 1)
        pad = np.zeros((S * gh + S, S * gw + S))
        pad[:S * gh, :S * gw] = hi
        sy, sx = int(round(fy * S)), int(round(fx * S))
        shifted = pad[sy:sy + S * gh, sx:sx + S * gw]
        cov = shifted.reshape(gh, S, gw, S).mean(axis=(1, 3)) / 255.0
        _blend(out, cov, y0, x0, gh, gw, rgb[i].astype(np.float64))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def probe_supersample_8x(page, atlas, gid, gy, gx, rgb):
    return _probe_supersample(page, atlas, gid, gy, gx, rgb, 8)


def probe_supersample_4x(page, atlas, gid, gy, gx, rgb):
    return _probe_supersample(page, atlas, gid, gy, gx, rgb, 4)


def probe_nearest(page, atlas, gid, gy, gx, rgb):
    """Snaps to the nearest integer pixel, discarding the sub-pixel phase."""
    out = page.astype(np.float64).copy()
    _, gh, gw = atlas.shape
    for i in range(len(gid)):
        y0, x0 = int(round(float(gy[i]))), int(round(float(gx[i])))
        cov = atlas[gid[i]].astype(np.float64) / 255.0
        _blend(out, cov, y0, x0, gh, gw, rgb[i].astype(np.float64))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def probe_drop_one(page, atlas, gid, gy, gx, rgb):
    """Exactly the reference, minus a single glyph."""
    keep = np.ones(len(gid), dtype=bool)
    keep[len(gid) // 2] = False
    return tasks.glyph_reference(page, atlas, gid[keep], gy[keep], gx[keep],
                                 rgb[keep])


PROBES = {
    "bilinear-f32": probe_bilinear_f32,
    "supersample-8x": probe_supersample_8x,
    "supersample-4x": probe_supersample_4x,
    "nearest-snap": probe_nearest,
    "drop-one-glyph": probe_drop_one,
}


def main():
    task = tasks.get(TASK_ID)
    oracle: PerceptualMatch = task.oracle
    inputs = task.make_inputs(np.random.default_rng(DEFAULT_SEED))
    truth = task.reference(**inputs)

    print(f"Calibrating `{TASK_ID}` perceptual oracle")
    print(f"  declared: ssim >= {oracle.min_ssim}, max_pixel_err <= "
          f"{oracle.max_pixel_err}, bad_frac <= {oracle.max_bad_frac}\n")

    rows, all_ok = [], True
    for name, fn in PROBES.items():
        got = fn(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in inputs.items()})
        r = oracle.check(got, truth)
        expected = PROBES_EXPECTED[name]
        ok = (r.passed == expected)
        all_ok &= ok
        m = r.metrics
        rows.append({"probe": name, "expected_pass": expected,
                     "actual_pass": r.passed, "separates": ok, **m})
        print(f"  {name:16} expect={'PASS' if expected else 'FAIL'} "
              f"got={'PASS' if r.passed else 'FAIL'} "
              f"{'ok' if ok else '<<< THRESHOLDS DO NOT SEPARATE'}")
        print(f"      ssim={m['ssim']:.5f}  max_px={m['max_pixel_err']:.4f}  "
              f"bad_frac={m['bad_frac']:.5f}")

    # How much margin is there between the tightest pass and loosest fail?
    passes = [r for r in rows if r["expected_pass"]]
    fails = [r for r in rows if not r["expected_pass"]]
    margin = min(r["ssim"] for r in passes) - max(r["ssim"] for r in fails)
    print(f"\n  SSIM margin (worst valid - best invalid): {margin:+.5f}")
    ssim_only = [r["probe"] for r in fails if r["ssim"] >= oracle.min_ssim]
    if ssim_only:
        print(f"  Caught ONLY by the non-SSIM caps: {ssim_only}")
        print("  -> this is the empirical case for not using SSIM alone.")

    out = {"task": TASK_ID, "seed": DEFAULT_SEED,
           "thresholds": oracle.thresholds(),
           "justification": oracle.justification,
           "all_probes_separated": bool(all_ok),
           "ssim_margin": margin,
           "caught_only_by_caps": ssim_only,
           "probes": rows}
    p = Path(__file__).parent / "results" / "threshold_calibration.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {p.relative_to(Path(__file__).parent)}")
    print("  VERDICT:", "thresholds separate all probes" if all_ok
          else "THRESHOLDS FAIL TO SEPARATE -- do not run experiments")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
