"""Render the figures used in the dashboard and the writeup.

  fig_page.png       the reference full-page raster (what a VTC encoder eats)
  fig_probes.png     a magnified crop of each calibration probe next to its
                     error map, which is the visual argument for the oracle

Everything is rendered from the same seeded inputs the benchmark uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import calibrate_thresholds as calib          # noqa: E402
from renderbench import tasks                 # noqa: E402
from renderbench.judge import DEFAULT_SEED    # noqa: E402

FIG = ROOT / "results" / "figures"
# Centred on the glyph that `drop-one-glyph` removes (a lowercase z at
# y=389.5, x=131.7), so that probe's column shows the missing letter rather than
# an unremarkable patch. Every probe is cropped identically.
CROP = (72, 352, 212, 442)     # x0, y0, x1, y1 in the 512x512 glyph page
ZOOM = 4


def _crop(img, box=CROP, zoom=ZOOM):
    x0, y0, x1, y1 = box
    c = Image.fromarray(img[y0:y1, x0:x1])
    return c.resize((c.width * zoom, c.height * zoom), Image.NEAREST)


def _errmap(got, truth, box=CROP, zoom=ZOOM):
    """Absolute error, gamma-boosted so sub-threshold error is still visible."""
    d = np.abs(got.astype(np.float64) - truth.astype(np.float64)).max(axis=2)
    d = (np.clip(d / 255.0, 0, 1) ** 0.45) * 255.0
    x0, y0, x1, y1 = box
    heat = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    v = d[y0:y1, x0:x1]
    heat[..., 0] = v.astype(np.uint8)                      # red   = error
    heat[..., 2] = (255 - v).astype(np.uint8) // 4         # blue  = clean
    im = Image.fromarray(heat)
    return im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)


def label_strip(width, text, h=22):
    from PIL import ImageDraw
    im = Image.new("RGB", (width, h), (17, 17, 20))
    ImageDraw.Draw(im).text((6, 5), text, fill=(225, 225, 235))
    return im


def stack_v(images):
    w = max(i.width for i in images)
    out = Image.new("RGB", (w, sum(i.height for i in images)), (17, 17, 20))
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height
    return out


def stack_h(images, gap=10):
    h = max(i.height for i in images)
    out = Image.new("RGB", (sum(i.width for i in images) + gap * (len(images) - 1), h),
                    (17, 17, 20))
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    # ---- full page ----------------------------------------------------
    page_task = tasks.get("text_page_raster")
    pins = page_task.make_inputs(np.random.default_rng(DEFAULT_SEED))
    page = page_task.reference(**pins)
    Image.fromarray(page).save(FIG / "fig_page.png")
    print(f"  fig_page.png       {page.shape[1]}x{page.shape[0]}  "
          f"{len(pins['gid'])} glyphs")

    # ---- probe comparison ---------------------------------------------
    gtask = tasks.get("glyph_atlas_blit")
    gins = gtask.make_inputs(np.random.default_rng(DEFAULT_SEED))
    truth = gtask.reference(**gins)

    cols = [stack_v([label_strip((CROP[2] - CROP[0]) * ZOOM,
                                 "reference  (ground truth)"),
                     _crop(truth),
                     label_strip((CROP[2] - CROP[0]) * ZOOM, "error: none")])]

    for name, fn in calib.PROBES.items():
        got = fn(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in gins.items()})
        r = gtask.oracle.check(got, truth)
        verdict = "PASS" if r.passed else "FAIL"
        m = r.metrics
        cols.append(stack_v([
            label_strip((CROP[2] - CROP[0]) * ZOOM, f"{name}   [{verdict}]"),
            _crop(got),
            label_strip((CROP[2] - CROP[0]) * ZOOM,
                        f"ssim {m['ssim']:.5f}  maxpx {m['max_pixel_err']:.3f}"),
            _errmap(got, truth),
        ]))
        print(f"  probe {name:16} {verdict}")

    out = stack_h(cols)
    out.save(FIG / "fig_probes.png")
    print(f"  fig_probes.png     {out.width}x{out.height}")
    print(f"\nwrote {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
