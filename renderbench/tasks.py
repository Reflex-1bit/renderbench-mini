"""The RenderBench-mini task suite.

Six rendering primitives, scaled down from the eight in the full plan (the
descope ladder cuts `atlas_lookup` and `tiled_multipage_pack` first). Each task
carries its own input generator, an obviously-correct reference implementation
that doubles as the naive baseline, and the oracle appropriate to its numerics.

The tasks are deliberately ordered from "a single defensible answer" to "no
bit-reproducible answer exists", because that gradient is the methodological
point: kernel-agent benchmarks so far have only lived at the left end.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np

from .core import (ExactMatch, PerceptualMatch, TaskSpec, TimingConfig,
                   ToleranceMatch)

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"

# --------------------------------------------------------------------------
# Shared synthetic assets
# --------------------------------------------------------------------------

GLYPH_H, GLYPH_W = 24, 16
N_GLYPHS = 62
ALPHABET = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789")

_WIN_FONTS = ["segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf"]


def _font_path() -> str | None:
    root = os.environ.get("WINDIR", "C:/Windows")
    for name in _WIN_FONTS:
        p = Path(root) / "Fonts" / name
        if p.exists():
            return str(p)
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if Path(p).exists():
            return p
    return None


@functools.lru_cache(maxsize=1)
def glyph_atlas() -> np.ndarray:
    """(N_GLYPHS, GLYPH_H, GLYPH_W) uint8 anti-aliased coverage.

    Cached to disk so every run -- and every agent iteration -- sees byte-identical
    input. Renders from a real system font when one is available; otherwise falls
    back to procedural strokes so the benchmark stays runnable anywhere.
    """
    ASSET_DIR.mkdir(exist_ok=True)
    cache = ASSET_DIR / "glyph_atlas.npy"
    if cache.exists():
        return np.load(cache)

    atlas = np.zeros((N_GLYPHS, GLYPH_H, GLYPH_W), dtype=np.uint8)
    fp = _font_path()
    made = False
    if fp:
        try:
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.truetype(fp, GLYPH_H - 6)
            for i, ch in enumerate(ALPHABET):
                im = Image.new("L", (GLYPH_W, GLYPH_H), 0)
                ImageDraw.Draw(im).text((1, 1), ch, fill=255, font=font)
                atlas[i] = np.asarray(im, dtype=np.uint8)
            made = True
        except Exception:
            made = False
    if not made:
        yy, xx = np.mgrid[0:GLYPH_H, 0:GLYPH_W].astype(np.float64)
        for i in range(N_GLYPHS):
            f = 0.4 + 0.09 * (i % 11)
            ph = 0.5 * (i % 7)
            d = np.abs(np.sin(f * xx + ph) * (GLYPH_H * 0.32)
                       + GLYPH_H * 0.5 - yy)
            atlas[i] = np.clip(255.0 * (1.6 - d), 0, 255).astype(np.uint8)
    np.save(cache, atlas)
    return atlas


def _placements(rng: np.random.Generator, n: int, H: int, W: int):
    """Fractional glyph placements -- the reason the glyph tasks cannot use an
    exact oracle. Sub-pixel y/x means coverage must be resampled, and several
    resampling choices are all legitimate."""
    gid = rng.integers(0, N_GLYPHS, size=n).astype(np.int32)
    y = rng.uniform(0, H - GLYPH_H - 1, size=n).astype(np.float32)
    x = rng.uniform(0, W - GLYPH_W - 1, size=n).astype(np.float32)
    rgb = rng.integers(0, 256, size=(n, 3)).astype(np.uint8)
    return gid, y, x, rgb


# --------------------------------------------------------------------------
# T1  blit
# --------------------------------------------------------------------------

def blit_inputs(rng):
    return {
        "dst": rng.integers(0, 256, (1024, 768, 4), dtype=np.uint8),
        "src": rng.integers(0, 256, (256, 192, 4), dtype=np.uint8),
        "dy": np.int32(880), "dx": np.int32(660),
    }


def blit_reference(dst, src, dy, dx):
    """Copy src into dst at (dy,dx), clipping at the destination edges."""
    out = dst.copy()
    H, W = out.shape[:2]
    sh, sw = src.shape[:2]
    for r in range(sh):
        oy = int(dy) + r
        if oy < 0 or oy >= H:
            continue
        for c in range(sw):
            ox = int(dx) + c
            if 0 <= ox < W:
                out[oy, ox] = src[r, c]
    return out


BLIT = TaskSpec(
    task_id="blit",
    title="Clipped RGBA blit",
    summary="Copy a sub-image into a page buffer, clipping at the edges.",
    signature="def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray",
    description="""
Copy every pixel of `src` (sh, sw, 4) uint8 into a **copy** of `dst`
(H, W, 4) uint8 with its top-left corner at row `dy`, column `dx`.

Pixels that land outside `dst` are dropped -- the destination is never grown and
never wraps. `dst` must not be mutated in place; return a new array.
The placement is chosen so that clipping is actually exercised.
""",
    oracle=ExactMatch(),
    make_inputs=blit_inputs,
    reference=blit_reference,
    naive_source="reference (per-pixel Python loop with a bounds test per pixel)",
    vtc_relevance="Page assembly: pasting a rendered text block into the image "
                  "handed to the vision encoder.",
    timing=TimingConfig(warmup=3, trials=15),
)

# --------------------------------------------------------------------------
# T2  alpha_composite
# --------------------------------------------------------------------------

def composite_inputs(rng):
    return {
        "dst": rng.integers(0, 256, (1024, 768, 4), dtype=np.uint8),
        "src": rng.integers(0, 256, (1024, 768, 4), dtype=np.uint8),
    }


def composite_reference(dst, src):
    """Premultiplied source-over: out = src + dst * (255 - src_a) // 255.

    Integer floor division, not rounding -- fixed by this spec so the result is
    uniquely defined and can be checked exactly.
    """
    out = np.empty_like(dst)
    H = dst.shape[0]
    for r in range(H):
        sa = src[r, :, 3].astype(np.uint16)
        inv = (255 - sa)
        for c in range(4):
            out[r, :, c] = (src[r, :, c].astype(np.uint16)
                            + (dst[r, :, c].astype(np.uint16) * inv) // 255
                            ).clip(0, 255).astype(np.uint8)
    return out


COMPOSITE = TaskSpec(
    task_id="alpha_composite",
    title="Premultiplied source-over composite",
    summary="Blend a full-page RGBA layer over another with integer alpha math.",
    signature="def kernel(dst: np.ndarray, src: np.ndarray) -> np.ndarray",
    description="""
Composite premultiplied-alpha `src` over `dst`, both (H, W, 4) uint8, and return
a new array.

Per channel c (including alpha):
    out[c] = clip(src[c] + (dst[c] * (255 - src[3])) // 255, 0, 255)

Integer **floor** division by 255, applied to all four channels. Do the
arithmetic in a width that cannot overflow: dst[c] * 255 exceeds uint8 and also
exceeds int16 when intermediates are chained.
""",
    oracle=ExactMatch(),
    make_inputs=composite_inputs,
    reference=composite_reference,
    naive_source="reference (row loop, per-channel widening to uint16)",
    vtc_relevance="Layering rendered text over a background before encode.",
    timing=TimingConfig(warmup=3, trials=15),
)

# --------------------------------------------------------------------------
# T3  srgb_gamma
# --------------------------------------------------------------------------

def gamma_inputs(rng):
    return {"linear": rng.random((1024, 768, 3), dtype=np.float32)}


def gamma_reference(linear):
    """Piecewise sRGB opto-electronic transfer function, evaluated in float64."""
    out = np.empty(linear.shape, dtype=np.uint8)
    for r in range(linear.shape[0]):
        v = linear[r].astype(np.float64)
        lo = v * 12.92
        hi = 1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055
        s = np.where(v <= 0.0031308, lo, hi)
        out[r] = np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)
    return out


GAMMA = TaskSpec(
    task_id="srgb_gamma",
    title="Linear to sRGB encode",
    summary="Piecewise transfer function over a full page, float32 in, uint8 out.",
    signature="def kernel(linear: np.ndarray) -> np.ndarray",
    description="""
Encode linear light `linear` (H, W, 3) float32 in [0, 1] to 8-bit sRGB.

    s = 12.92 * v                              if v <= 0.0031308
    s = 1.055 * v ** (1 / 2.4) - 0.055         otherwise
    out = clip(round(s * 255), 0, 255)  as uint8

A 256- or 4096-entry lookup table is an acceptable implementation strategy: the
oracle allows one 8-bit code value of deviation precisely so that LUT-based and
direct-`pow` implementations can both pass.
""",
    oracle=ToleranceMatch(
        atol=1.0,
        justification=(
            "A quantised LUT and a direct pow() evaluation legitimately differ by "
            "up to one 8-bit code value; requiring bit-equality here would reject "
            "the standard production implementation."),
    ),
    make_inputs=gamma_inputs,
    reference=gamma_reference,
    naive_source="reference (row loop, float64 pow on every pixel)",
    vtc_relevance="Colour-space conversion on the render path before the encoder.",
    timing=TimingConfig(warmup=3, trials=15),
)

# --------------------------------------------------------------------------
# T4  bilinear_resize
# --------------------------------------------------------------------------

def resize_inputs(rng):
    return {
        "src": rng.integers(0, 256, (512, 384, 4), dtype=np.uint8),
        "out_h": np.int32(1024), "out_w": np.int32(768),
    }


def resize_reference(src, out_h, out_w):
    """Half-pixel-centre bilinear resample (align_corners=False)."""
    out_h, out_w = int(out_h), int(out_w)
    sh, sw = src.shape[:2]
    out = np.empty((out_h, out_w, src.shape[2]), dtype=np.uint8)
    sy_all = (np.arange(out_h) + 0.5) * sh / out_h - 0.5
    sx_all = (np.arange(out_w) + 0.5) * sw / out_w - 0.5
    sx = np.clip(sx_all, 0, sw - 1)
    x0 = np.floor(sx).astype(np.int64)
    x1 = np.minimum(x0 + 1, sw - 1)
    wx = (sx - x0)[None, :, None]
    for r in range(out_h):
        y = min(max(sy_all[r], 0.0), sh - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, sh - 1)
        wy = y - y0
        top = src[y0][x0].astype(np.float64) * (1 - wx[0]) + src[y0][x1].astype(np.float64) * wx[0]
        bot = src[y1][x0].astype(np.float64) * (1 - wx[0]) + src[y1][x1].astype(np.float64) * wx[0]
        out[r] = np.clip(np.rint(top * (1 - wy) + bot * wy), 0, 255).astype(np.uint8)
    return out


RESIZE = TaskSpec(
    task_id="bilinear_resize",
    title="Bilinear RGBA resample",
    summary="Upscale a page with half-pixel-centre bilinear filtering.",
    signature="def kernel(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray",
    description="""
Resample `src` (sh, sw, 4) uint8 to (out_h, out_w, 4) uint8 with bilinear
filtering and **half-pixel centres** (`align_corners=False`):

    sy = (r + 0.5) * sh / out_h - 0.5, clamped to [0, sh - 1]
    sx = (c + 0.5) * sw / out_w - 0.5, clamped to [0, sw - 1]

Bottom/right taps clamp to the last row/column. Round the final weighted sum to
nearest and clip to [0, 255].
""",
    oracle=ToleranceMatch(
        atol=1.0,
        justification=(
            "Accumulating the four taps in a different order, or in float32 rather "
            "than float64, shifts the rounded result by at most one code value; "
            "that is an implementation freedom, not an error."),
    ),
    make_inputs=resize_inputs,
    reference=resize_reference,
    naive_source="reference (row loop, float64 gather and lerp)",
    vtc_relevance="Rescaling a rendered page to the vision encoder's input grid.",
    timing=TimingConfig(warmup=3, trials=10),
)

# --------------------------------------------------------------------------
# T5  glyph_atlas_blit
# --------------------------------------------------------------------------

PAGE_H, PAGE_W = 512, 512
N_SPRITES = 600


def glyph_inputs(rng):
    gid, y, x, rgb = _placements(rng, N_SPRITES, PAGE_H, PAGE_W)
    return {
        "page": np.full((PAGE_H, PAGE_W, 3), 255, dtype=np.uint8),
        "atlas": glyph_atlas(),
        "gid": gid, "gy": y, "gx": x, "rgb": rgb,
    }


def _sample_coverage(g, fy, fx):
    """Bilinear resample of one glyph's coverage at a fractional offset."""
    gh, gw = g.shape
    pad = np.zeros((gh + 1, gw + 1), dtype=np.float64)
    pad[:gh, :gw] = g
    a = pad[:gh, :gw] * (1 - fy) * (1 - fx)
    b = pad[1:, :gw] * fy * (1 - fx)
    c = pad[:gh, 1:] * (1 - fy) * fx
    d = pad[1:, 1:] * fy * fx
    return a + b + c + d


def glyph_reference(page, atlas, gid, gy, gx, rgb):
    """Alpha-blend each glyph's anti-aliased coverage onto the page in colour."""
    out = page.astype(np.float64).copy()
    gh, gw = atlas.shape[1], atlas.shape[2]
    for i in range(len(gid)):
        y0, x0 = int(np.floor(gy[i])), int(np.floor(gx[i]))
        cov = _sample_coverage(atlas[gid[i]], float(gy[i]) - y0, float(gx[i]) - x0) / 255.0
        tile = out[y0:y0 + gh, x0:x0 + gw]
        col = rgb[i].astype(np.float64)
        tile *= (1.0 - cov[..., None])
        tile += cov[..., None] * col
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


_GLYPH_JUSTIFICATION = (
    "Anti-aliased coverage placed at a fractional offset has no bit-reproducible "
    "answer: direct bilinear resampling and area-averaged supersampling are both "
    "defensible implementations. The admissible set is therefore defined by a "
    "property rather than by an algorithm -- an implementation passes if it "
    "resolves sub-pixel phase to 1/8 px or finer and loses no coverage. The three "
    "thresholds were fitted to that rule by calibrate_thresholds.py, which "
    "confirms they admit exact bilinear and 1/8-px supersampling while rejecting "
    "1/4-px phase quantisation, nearest-neighbour snapping, and a dropped glyph. "
    "Calibrated before any experiment was run; frozen since."
)

# min_ssim rejects 1/4-px phase quantisation (0.9973) while admitting 1/8-px
# (0.99928). max_pixel_err is the load-bearing one: a single dropped glyph scores
# SSIM 0.9993 -- above any usable structural floor -- and is caught only here.
_GLYPH_ORACLE = dict(min_ssim=0.998, max_pixel_err=0.100, max_bad_frac=0.001,
                     justification=_GLYPH_JUSTIFICATION)

GLYPH = TaskSpec(
    task_id="glyph_atlas_blit",
    title="Anti-aliased glyph atlas blit",
    summary="Composite 600 sub-pixel-positioned glyphs from an atlas onto a page.",
    signature=("def kernel(page, atlas, gid, gy, gx, rgb) -> np.ndarray"),
    description="""
Composite `len(gid)` glyphs onto a copy of `page` (H, W, 3) uint8.

`atlas` is (n_glyphs, gh, gw) uint8 coverage. For glyph i, coverage comes from
`atlas[gid[i]]` placed at **fractional** position (`gy[i]`, `gx[i]`) -- take the
integer part as the top-left corner and resample the coverage by the fractional
remainder. Coverage outside the glyph box is zero.

Then blend in colour `rgb[i]`:

    a = coverage / 255
    page = page * (1 - a) + rgb[i] * a

Glyphs are drawn in array order and may overlap, so the blend is order
dependent. Round to nearest at the end. Placements never cross the page edge.
""",
    oracle=PerceptualMatch(**_GLYPH_ORACLE),
    make_inputs=glyph_inputs,
    reference=glyph_reference,
    naive_source="reference (Python loop over glyphs, float64 tile blend)",
    vtc_relevance="The core of the VTC render step -- this is the loop Glyph "
                  "currently pays for on the CPU.",
    timing=TimingConfig(warmup=2, trials=8),
)

# --------------------------------------------------------------------------
# T6  text_page_raster
# --------------------------------------------------------------------------

LINES, COLS = 40, 78
LEADING, ADVANCE = 12.0, 6.4


def page_inputs(rng):
    n = LINES * COLS
    gid = rng.integers(0, N_GLYPHS, size=n).astype(np.int32)
    row = np.repeat(np.arange(LINES), COLS)
    col = np.tile(np.arange(COLS), LINES)
    gy = (8.0 + row * LEADING).astype(np.float32)
    gx = (6.0 + col * ADVANCE).astype(np.float32)
    return {
        "page": np.full((int(8 + LINES * LEADING) + GLYPH_H + 4,
                         int(6 + COLS * ADVANCE) + GLYPH_W + 4, 3),
                        255, dtype=np.uint8),
        "atlas": glyph_atlas(),
        "gid": gid, "gy": gy, "gx": gx,
        "rgb": np.zeros((n, 3), dtype=np.uint8),
    }


PAGE = TaskSpec(
    task_id="text_page_raster",
    title="Full-page text raster",
    summary="Rasterise a 40x78 page of text -- the VTC workload end to end.",
    signature="def kernel(page, atlas, gid, gy, gx, rgb) -> np.ndarray",
    description="""
Identical contract to `glyph_atlas_blit`, but at full-page scale: ~3100 glyphs
laid out on a text grid with a fractional horizontal advance, all in black on
white.

This is the workload a VTC pipeline actually runs per page, so its wall-clock
time is the one that converts directly into a cost-per-million-tokens number.
Because the advance is fractional, most glyphs sit at a different sub-pixel
phase, and per-glyph coverage cannot be cached naively across the row.
""",
    oracle=PerceptualMatch(**_GLYPH_ORACLE),
    make_inputs=page_inputs,
    reference=glyph_reference,
    naive_source="reference (Python loop over ~3100 glyphs)",
    vtc_relevance="Directly the step Glyph's own repo flags as having "
                  "'significant room for acceleration'.",
    timing=TimingConfig(warmup=1, trials=5),
)

# --------------------------------------------------------------------------

REGISTRY: dict[str, TaskSpec] = {
    t.task_id: t for t in (BLIT, COMPOSITE, GAMMA, RESIZE, GLYPH, PAGE)
}

TASK_ORDER = list(REGISTRY)


def get(task_id: str) -> TaskSpec:
    if task_id not in REGISTRY:
        raise KeyError(f"unknown task {task_id!r}; have {TASK_ORDER}")
    return REGISTRY[task_id]
