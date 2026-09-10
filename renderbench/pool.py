"""Replay pool for offline mode.

No LLM API key is required to demo the loop. In offline mode the coder agent
draws from this pool instead of calling a model. The kernels here are REAL --
they are compiled, executed, checked and timed by the same frozen judge, so the
correctness verdicts, speedups and advisor critiques in an offline run are all
genuine measurements. The only thing being replayed is the model's choice of
what to write next.

Each task's list is ordered the way a competent iteration actually goes: a first
attempt with a characteristic bug, a correct-but-slow fix, then optimisation.
Across the suite the first attempts cover every failure stage the judge can
report -- run error, exact mismatch, tolerance violation, perceptual violation --
so the harness's error paths are all exercised by a real run.

Offline results are labelled `provider=offline` everywhere and must never be
reported as a measurement of model capability. They measure the harness.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PoolEntry:
    label: str
    note: str          # what this attempt is trying, in the coder's voice
    source: str


# ==========================================================================
# blit
# ==========================================================================

_BLIT = [
    PoolEntry(
        "unclipped-slice",
        "Replace the per-pixel loop with a single slice assignment.",
        """
import numpy as np

def kernel(dst, src, dy, dx):
    out = dst.copy()
    dy, dx = int(dy), int(dx)
    sh, sw = src.shape[:2]
    out[dy:dy + sh, dx:dx + sw] = src
    return out
""",
    ),
    PoolEntry(
        "clipped-slice",
        "Compute the clipped overlap rectangle first, then slice both sides.",
        """
import numpy as np

def kernel(dst, src, dy, dx):
    out = dst.copy()
    H, W = out.shape[:2]
    sh, sw = src.shape[:2]
    dy, dx = int(dy), int(dx)

    y0, x0 = max(0, dy), max(0, dx)
    y1, x1 = min(H, dy + sh), min(W, dx + sw)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out
""",
    ),
]

# ==========================================================================
# alpha_composite
# ==========================================================================

_COMPOSITE = [
    PoolEntry(
        "uint8-overflow",
        "Vectorise the whole page at once with plain array arithmetic.",
        """
import numpy as np

def kernel(dst, src):
    inv = 255 - src[:, :, 3:4]
    return np.clip(src + (dst * inv) // 255, 0, 255).astype(np.uint8)
""",
    ),
    PoolEntry(
        "float64-widen",
        "Widen to float64 so the multiply cannot wrap, then floor and clip.",
        """
import numpy as np

def kernel(dst, src):
    s = src.astype(np.float64)
    d = dst.astype(np.float64)
    inv = 255.0 - s[:, :, 3:4]
    out = s + np.floor(d * inv / 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)
""",
    ),
    PoolEntry(
        "uint16-int",
        "float64 triples the memory traffic. uint16 is wide enough for "
        "255*255 and keeps the op integer end to end.",
        """
import numpy as np

def kernel(dst, src):
    s = src.astype(np.uint16)
    d = dst.astype(np.uint16)
    inv = 255 - s[:, :, 3:4]
    out = s + (d * inv) // 255
    return np.minimum(out, 255).astype(np.uint8)
""",
    ),
    PoolEntry(
        "uint16-preallocated",
        "This op is memory bound, so cut allocations: one scratch buffer, "
        "in-place multiply, in-place floor-divide, single output write.",
        """
import numpy as np

def kernel(dst, src):
    s16 = src.astype(np.uint16)
    scratch = dst.astype(np.uint16)
    inv = 255 - s16[:, :, 3:4]

    scratch *= inv
    scratch //= 255
    scratch += s16
    np.minimum(scratch, 255, out=scratch)
    return scratch.astype(np.uint8)
""",
    ),
]

# ==========================================================================
# srgb_gamma
# ==========================================================================

_GAMMA = [
    PoolEntry(
        "pure-power",
        "Vectorise the transfer function as a single power law.",
        """
import numpy as np

def kernel(linear):
    s = 1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)
""",
    ),
    PoolEntry(
        "piecewise-vectorised",
        "Restore the linear segment below the 0.0031308 knee, still vectorised.",
        """
import numpy as np

def kernel(linear):
    v = linear.astype(np.float64)
    s = np.where(v <= 0.0031308,
                 v * 12.92,
                 1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)
""",
    ),
    PoolEntry(
        "lut-4096",
        "pow() on 2.4M pixels dominates. The oracle allows one code value of "
        "error, so a 4096-entry LUT indexed by the quantised input is in "
        "budget and turns the transcendental into a gather.",
        """
import numpy as np

_N = 4096
_x = (np.arange(_N) + 0.5) / _N
_s = np.where(_x <= 0.0031308, _x * 12.92,
              1.055 * np.power(_x, 1.0 / 2.4) - 0.055)
_LUT = np.clip(np.rint(_s * 255.0), 0, 255).astype(np.uint8)

def kernel(linear):
    idx = np.clip((linear * _N).astype(np.int32), 0, _N - 1)
    return _LUT[idx]
""",
    ),
]

# ==========================================================================
# bilinear_resize
# ==========================================================================

_RESIZE = [
    PoolEntry(
        "align-corners",
        "Vectorise with a coordinate grid over the full source extent.",
        """
import numpy as np

def kernel(src, out_h, out_w):
    out_h, out_w = int(out_h), int(out_w)
    sh, sw = src.shape[:2]
    sy = np.arange(out_h) * (sh - 1) / (out_h - 1)
    sx = np.arange(out_w) * (sw - 1) / (out_w - 1)
    y0 = np.floor(sy).astype(np.int64); y1 = np.minimum(y0 + 1, sh - 1)
    x0 = np.floor(sx).astype(np.int64); x1 = np.minimum(x0 + 1, sw - 1)
    wy = (sy - y0)[:, None, None]; wx = (sx - x0)[None, :, None]
    s = src.astype(np.float64)
    top = s[y0][:, x0] * (1 - wx) + s[y0][:, x1] * wx
    bot = s[y1][:, x0] * (1 - wx) + s[y1][:, x1] * wx
    return np.clip(np.rint(top * (1 - wy) + bot * wy), 0, 255).astype(np.uint8)
""",
    ),
    PoolEntry(
        "half-pixel",
        "Switch to half-pixel centres and clamp before flooring, per the spec.",
        """
import numpy as np

def kernel(src, out_h, out_w):
    out_h, out_w = int(out_h), int(out_w)
    sh, sw = src.shape[:2]
    sy = np.clip((np.arange(out_h) + 0.5) * sh / out_h - 0.5, 0, sh - 1)
    sx = np.clip((np.arange(out_w) + 0.5) * sw / out_w - 0.5, 0, sw - 1)
    y0 = np.floor(sy).astype(np.int64); y1 = np.minimum(y0 + 1, sh - 1)
    x0 = np.floor(sx).astype(np.int64); x1 = np.minimum(x0 + 1, sw - 1)
    wy = (sy - y0)[:, None, None]; wx = (sx - x0)[None, :, None]
    s = src.astype(np.float64)
    top = s[y0][:, x0] * (1 - wx) + s[y0][:, x1] * wx
    bot = s[y1][:, x0] * (1 - wx) + s[y1][:, x1] * wx
    return np.clip(np.rint(top * (1 - wy) + bot * wy), 0, 255).astype(np.uint8)
""",
    ),
    PoolEntry(
        "separable-float32",
        "The 4-tap gather materialises four full float64 pages. Do it "
        "separably instead: interpolate rows into an (out_h, sw) intermediate, "
        "then columns. float32 stays inside the 1-code-value budget.",
        """
import numpy as np

def kernel(src, out_h, out_w):
    out_h, out_w = int(out_h), int(out_w)
    sh, sw = src.shape[:2]
    s = src.astype(np.float32)

    sy = np.clip((np.arange(out_h) + 0.5) * sh / out_h - 0.5, 0, sh - 1)
    y0 = np.floor(sy).astype(np.int64); y1 = np.minimum(y0 + 1, sh - 1)
    wy = (sy - y0).astype(np.float32)[:, None, None]
    rows = s[y0] * (1 - wy) + s[y1] * wy

    sx = np.clip((np.arange(out_w) + 0.5) * sw / out_w - 0.5, 0, sw - 1)
    x0 = np.floor(sx).astype(np.int64); x1 = np.minimum(x0 + 1, sw - 1)
    wx = (sx - x0).astype(np.float32)[None, :, None]
    out = rows[:, x0] * (1 - wx) + rows[:, x1] * wx

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
""",
    ),
]

# ==========================================================================
# glyph_atlas_blit / text_page_raster  (identical contract)
# ==========================================================================

_GLYPH_NEAREST = PoolEntry(
    "nearest-snap",
    "Snap each glyph to the nearest integer pixel so the blit is a plain "
    "slice -- the sub-pixel offset should be visually negligible.",
    """
import numpy as np

def kernel(page, atlas, gid, gy, gx, rgb):
    out = page.astype(np.float64).copy()
    gh, gw = atlas.shape[1], atlas.shape[2]
    for i in range(len(gid)):
        y0 = int(round(float(gy[i]))); x0 = int(round(float(gx[i])))
        cov = atlas[gid[i]].astype(np.float64) / 255.0
        tile = out[y0:y0 + gh, x0:x0 + gw]
        a = cov[..., None]
        tile *= (1.0 - a)
        tile += a * rgb[i].astype(np.float64)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
""",
)

_GLYPH_BILINEAR = PoolEntry(
    "per-glyph-bilinear",
    "Restore the sub-pixel resample: pad the coverage and blend the four "
    "shifted taps per glyph before compositing.",
    """
import numpy as np

def kernel(page, atlas, gid, gy, gx, rgb):
    out = page.astype(np.float64).copy()
    n, gh, gw = atlas.shape
    for i in range(len(gid)):
        y0 = int(np.floor(gy[i])); x0 = int(np.floor(gx[i]))
        fy = float(gy[i]) - y0; fx = float(gx[i]) - x0

        pad = np.zeros((gh + 1, gw + 1), dtype=np.float64)
        pad[:gh, :gw] = atlas[gid[i]]
        cov = (pad[:gh, :gw] * (1 - fy) * (1 - fx)
               + pad[1:, :gw] * fy * (1 - fx)
               + pad[:gh, 1:] * (1 - fy) * fx
               + pad[1:, 1:] * fy * fx) / 255.0

        tile = out[y0:y0 + gh, x0:x0 + gw]
        a = cov[..., None]
        tile *= (1.0 - a)
        tile += a * rgb[i].astype(np.float64)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
""",
)

_GLYPH_BATCHED = PoolEntry(
    "batched-coverage",
    "The resample is per-glyph but independent, so lift it out of the loop: "
    "gather every glyph into one (n, gh+1, gw+1) padded stack and compute all "
    "coverages in four vectorised terms. Only the order-dependent blend stays "
    "in Python.",
    """
import numpy as np

def kernel(page, atlas, gid, gy, gx, rgb):
    out = page.astype(np.float64).copy()
    n, gh, gw = atlas.shape
    m = len(gid)

    y0 = np.floor(gy).astype(np.int64)
    x0 = np.floor(gx).astype(np.int64)
    fy = (gy - y0).astype(np.float64)[:, None, None]
    fx = (gx - x0).astype(np.float64)[:, None, None]

    pad = np.zeros((m, gh + 1, gw + 1), dtype=np.float64)
    pad[:, :gh, :gw] = atlas[gid]

    cov = (pad[:, :gh, :gw] * (1 - fy) * (1 - fx)
           + pad[:, 1:, :gw] * fy * (1 - fx)
           + pad[:, :gh, 1:] * (1 - fy) * fx
           + pad[:, 1:, 1:] * fy * fx)
    cov *= (1.0 / 255.0)

    colours = rgb.astype(np.float64)
    for i in range(m):
        a = cov[i][..., None]
        tile = out[y0[i]:y0[i] + gh, x0[i]:x0[i] + gw]
        tile *= (1.0 - a)
        tile += a * colours[i]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
""",
)

_GLYPH_FIXEDPOINT = PoolEntry(
    "batched-premultiplied",
    "Fold the colour into the coverage once (premultiply) so the inner blend "
    "is a single fused multiply-add on a (gh, gw, 3) tile, and keep the "
    "coverage stack in float32 to halve the bandwidth of the batched resample.",
    """
import numpy as np

def kernel(page, atlas, gid, gy, gx, rgb):
    out = page.astype(np.float32).copy()
    n, gh, gw = atlas.shape
    m = len(gid)

    y0 = np.floor(gy).astype(np.int64)
    x0 = np.floor(gx).astype(np.int64)
    fy = (gy - y0).astype(np.float32)[:, None, None]
    fx = (gx - x0).astype(np.float32)[:, None, None]

    pad = np.zeros((m, gh + 1, gw + 1), dtype=np.float32)
    pad[:, :gh, :gw] = atlas[gid]

    cov = (pad[:, :gh, :gw] * ((1 - fy) * (1 - fx))
           + pad[:, 1:, :gw] * (fy * (1 - fx))
           + pad[:, :gh, 1:] * ((1 - fy) * fx)
           + pad[:, 1:, 1:] * (fy * fx))
    cov *= np.float32(1.0 / 255.0)

    inv = 1.0 - cov
    pre = cov[..., None] * rgb.astype(np.float32)[:, None, None, :]

    for i in range(m):
        tile = out[y0[i]:y0[i] + gh, x0[i]:x0[i] + gw]
        tile *= inv[i][..., None]
        tile += pre[i]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
""",
)

_GLYPH = [_GLYPH_NEAREST, _GLYPH_BILINEAR, _GLYPH_BATCHED, _GLYPH_FIXEDPOINT]

POOL: dict[str, list[PoolEntry]] = {
    "blit": _BLIT,
    "alpha_composite": _COMPOSITE,
    "srgb_gamma": _GAMMA,
    "bilinear_resize": _RESIZE,
    "glyph_atlas_blit": _GLYPH,
    "text_page_raster": _GLYPH,
}
