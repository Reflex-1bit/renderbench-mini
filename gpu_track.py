"""GPU track: the same two tasks, written as real Triton kernels.

The CPU track exists so the harness runs anywhere. This is the track the actual
research targets: Triton kernels on real hardware, scored by the *same frozen
oracles* as the CPU track, so a GPU kernel has to clear exactly the bar a CPU one
does -- including the calibrated perceptual thresholds.

Two kernels, chosen to sit at opposite ends of the difficulty gradient:

  alpha_composite    embarrassingly parallel, elementwise, integer -- the shape
                     of kernel that existing kernel-agent work already handles.

  glyph_atlas_blit   a tiled rasteriser. Source-over blending is not associative,
                     so glyphs must be applied in submission order; that forbids
                     the obvious one-thread-per-glyph scatter. Instead each
                     program owns one output tile and walks the glyph list in
                     order, skipping glyphs whose box misses the tile. This is
                     the branch-heavy, memory-bound, order-constrained structure
                     that makes rendering a genuine generalisation test rather
                     than another elementwise op.

    python gpu_track.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from renderbench import tasks                                   # noqa: E402
from renderbench.core import TimingConfig, time_callable        # noqa: E402
from renderbench.judge import DEFAULT_SEED, baseline_ms         # noqa: E402

try:
    import torch
    import triton
    import triton.language as tl
except Exception as e:  # noqa: BLE001
    sys.exit(f"GPU track needs torch + triton: {e}")

if not torch.cuda.is_available():
    sys.exit("GPU track needs a CUDA device")

DEV = "cuda"


# ==========================================================================
# alpha_composite
# ==========================================================================

@triton.jit
def _composite_kernel(dst_ptr, src_ptr, out_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n

    d = tl.load(dst_ptr + off, mask=m, other=0).to(tl.int32)
    s = tl.load(src_ptr + off, mask=m, other=0).to(tl.int32)
    # Alpha is channel 3 of each RGBA group; broadcast it across the group.
    a = tl.load(src_ptr + (off // 4) * 4 + 3, mask=m, other=0).to(tl.int32)

    out = s + (d * (255 - a)) // 255
    out = tl.minimum(out, 255)
    tl.store(out_ptr + off, out.to(tl.uint8), mask=m)


def composite_gpu(dst_t, src_t):
    out = torch.empty_like(dst_t)
    n = dst_t.numel()
    _composite_kernel[(triton.cdiv(n, 1024),)](dst_t, src_t, out, n, BLOCK=1024)
    return out


# ==========================================================================
# glyph_atlas_blit -- tiled rasteriser
# ==========================================================================

@triton.jit
def _glyph_tile_kernel(page_ptr, out_ptr, atlas_ptr,
                       gid_ptr, gy_ptr, gx_ptr, rgb_ptr,
                       n_glyphs, H, W,
                       GH: tl.constexpr, GW: tl.constexpr,
                       TILE: tl.constexpr):
    tid_y = tl.program_id(0)
    tid_x = tl.program_id(1)
    y0t = tid_y * TILE
    x0t = tid_x * TILE

    ys = y0t + tl.arange(0, TILE)[:, None]
    xs = x0t + tl.arange(0, TILE)[None, :]
    inb = (ys < H) & (xs < W)
    base = (ys * W + xs) * 3

    r = tl.load(page_ptr + base + 0, mask=inb, other=0).to(tl.float32)
    g = tl.load(page_ptr + base + 1, mask=inb, other=0).to(tl.float32)
    b = tl.load(page_ptr + base + 2, mask=inb, other=0).to(tl.float32)

    for i in range(n_glyphs):
        gy = tl.load(gy_ptr + i)
        gx = tl.load(gx_ptr + i)
        iy = tl.floor(gy).to(tl.int32)
        ix = tl.floor(gx).to(tl.int32)

        # Scalar reject: does this glyph's box touch this tile at all?
        if (iy < y0t + TILE) and (iy + GH > y0t) and \
           (ix < x0t + TILE) and (ix + GW > x0t):
            fy = gy - tl.floor(gy)
            fx = gx - tl.floor(gx)
            gid = tl.load(gid_ptr + i)
            abase = gid * GH * GW

            ly = ys - iy
            lx = xs - ix

            # Four taps of the padded coverage. Every tap is gated on the pixel
            # lying inside the glyph's own (GH, GW) box: the reference writes
            # coverage only to page[y0:y0+GH, x0:x0+GW], so a tap that would
            # bleed one row above or one column left of that box must contribute
            # nothing. Dropping this gate leaves SSIM at 0.99999 and still puts
            # ~36 code values of error on the boundary pixels -- caught by the
            # oracle's per-pixel cap, not by its structural floor.
            inbox = (ly >= 0) & (ly < GH) & (lx >= 0) & (lx < GW)
            m00 = inbox
            m10 = inbox & (ly + 1 < GH)
            m01 = inbox & (lx + 1 < GW)
            m11 = m10 & (lx + 1 < GW)

            o00 = abase + ly * GW + lx
            c00 = tl.load(atlas_ptr + o00, mask=m00, other=0).to(tl.float32)
            c10 = tl.load(atlas_ptr + o00 + GW, mask=m10, other=0).to(tl.float32)
            c01 = tl.load(atlas_ptr + o00 + 1, mask=m01, other=0).to(tl.float32)
            c11 = tl.load(atlas_ptr + o00 + GW + 1, mask=m11, other=0).to(tl.float32)

            cov = (c00 * (1 - fy) * (1 - fx) + c10 * fy * (1 - fx)
                   + c01 * (1 - fy) * fx + c11 * fy * fx) * (1.0 / 255.0)

            cr = tl.load(rgb_ptr + i * 3 + 0).to(tl.float32)
            cg = tl.load(rgb_ptr + i * 3 + 1).to(tl.float32)
            cb = tl.load(rgb_ptr + i * 3 + 2).to(tl.float32)

            inv = 1.0 - cov
            r = r * inv + cov * cr
            g = g * inv + cov * cg
            b = b * inv + cov * cb

    # Round-half-away-from-zero to match numpy's rint on non-ties closely enough
    # for the calibrated tolerance; values here are non-negative.
    tl.store(out_ptr + base + 0,
             tl.minimum(tl.floor(r + 0.5), 255.0).to(tl.uint8), mask=inb)
    tl.store(out_ptr + base + 1,
             tl.minimum(tl.floor(g + 0.5), 255.0).to(tl.uint8), mask=inb)
    tl.store(out_ptr + base + 2,
             tl.minimum(tl.floor(b + 0.5), 255.0).to(tl.uint8), mask=inb)


def glyph_gpu(page_t, atlas_t, gid_t, gy_t, gx_t, rgb_t, tile=32):
    H, W = page_t.shape[0], page_t.shape[1]
    GH, GW = atlas_t.shape[1], atlas_t.shape[2]
    out = torch.empty_like(page_t)
    grid = (triton.cdiv(H, tile), triton.cdiv(W, tile))
    _glyph_tile_kernel[grid](page_t, out, atlas_t, gid_t, gy_t, gx_t, rgb_t,
                             gid_t.numel(), H, W, GH=GH, GW=GW, TILE=tile)
    return out


# ==========================================================================

def to_dev(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


def run_case(task_id, build, build_e2e, tile_note=""):
    """Times two honestly different things and reports both:

    kernel-only  -- input arrays already resident on the GPU, output left on
                    the GPU. This is what a fused pipeline stage would see if
                    it consumed the previous stage's tensor directly.
    end-to-end   -- host array in, kernel, host array out, every call. This is
                    what calling the kernel from a CPU-resident pipeline (e.g.
                    handed a numpy page, must return a numpy page) actually
                    costs, including the PCIe traffic. For small ops this
                    number is dramatically worse than kernel-only, and only
                    the second one is a fair comparison to a naive CPU
                    reference measured the same way -- reporting kernel-only
                    alone previously overstated this project's own numbers
                    by 25x on alpha_composite (1593x claimed vs 63x honest).
    """
    task = tasks.get(task_id)
    inputs = task.make_inputs(np.random.default_rng(DEFAULT_SEED))
    truth = np.load(ROOT / "assets" / f"truth_{task_id}_{DEFAULT_SEED}.npz")["truth"]

    fn = build(inputs)
    got = fn().cpu().numpy()
    res = task.oracle.check(got, truth)

    cfg = TimingConfig(warmup=10, trials=30)
    t_kernel = time_callable(fn, cfg, sync=torch.cuda.synchronize)

    fn_e2e = build_e2e(inputs)
    t_e2e = time_callable(fn_e2e, cfg, sync=torch.cuda.synchronize)

    base = baseline_ms(task_id)

    print(f"[{task_id}] {task.title}{tile_note}")
    print(f"    oracle       : {'PASS' if res.passed else 'FAIL'}  ({res.mode})  {res.detail}")
    print(f"    naive CPU    : {base:9.3f} ms")
    print(f"    triton kernel-only : {t_kernel.median_ms:9.3f} ms  "
          f"(data pre-staged on device) -> {base / t_kernel.median_ms:8.1f}x")
    print(f"    triton end-to-end  : {t_e2e.median_ms:9.3f} ms  "
          f"(host in, host out, every call) -> {base / t_e2e.median_ms:8.1f}x")
    print(f"    transfer overhead  : {t_e2e.median_ms - t_kernel.median_ms:9.3f} ms\n")

    return {"task": task_id, "passed": res.passed, "oracle_mode": res.mode,
            "oracle_detail": res.detail, "oracle_metrics": res.metrics,
            "naive_cpu_ms": base,
            "triton_kernel_only_ms": t_kernel.median_ms,
            "triton_e2e_ms": t_e2e.median_ms,
            "speedup_kernel_only": base / t_kernel.median_ms,
            "speedup_e2e": base / t_e2e.median_ms,
            "timing_kernel_only": t_kernel.to_dict(),
            "timing_e2e": t_e2e.to_dict()}


def main():
    print(f"GPU track  |  {torch.cuda.get_device_name(0)}  "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"torch {torch.__version__}  triton {triton.__version__}\n")

    results = []

    def build_composite(inp):
        d, s = to_dev(inp["dst"]), to_dev(inp["src"])
        return lambda: composite_gpu(d, s)

    def build_composite_e2e(inp):
        def run():
            d = torch.from_numpy(inp["dst"]).to(DEV)
            s = torch.from_numpy(inp["src"]).to(DEV)
            return composite_gpu(d, s).cpu().numpy()
        return run

    results.append(run_case("alpha_composite", build_composite, build_composite_e2e))

    for tile in (16, 32, 64):
        def build_glyph(inp, tile=tile):
            p = to_dev(inp["page"])
            a = to_dev(inp["atlas"])
            gid = to_dev(inp["gid"].astype(np.int32))
            gy = to_dev(inp["gy"].astype(np.float32))
            gx = to_dev(inp["gx"].astype(np.float32))
            rgb = to_dev(inp["rgb"])
            return lambda: glyph_gpu(p, a, gid, gy, gx, rgb, tile=tile)

        def build_glyph_e2e(inp, tile=tile):
            def run():
                p = torch.from_numpy(inp["page"]).to(DEV)
                a = torch.from_numpy(inp["atlas"]).to(DEV)
                gid = torch.from_numpy(inp["gid"].astype(np.int32)).to(DEV)
                gy = torch.from_numpy(inp["gy"].astype(np.float32)).to(DEV)
                gx = torch.from_numpy(inp["gx"].astype(np.float32)).to(DEV)
                rgb = torch.from_numpy(inp["rgb"]).to(DEV)
                return glyph_gpu(p, a, gid, gy, gx, rgb, tile=tile).cpu().numpy()
            return run

        results.append(run_case("glyph_atlas_blit", build_glyph, build_glyph_e2e,
                                f"  [TILE={tile}]"))
        results[-1]["tile"] = tile

    ok = [r for r in results if r["passed"]]
    geo_kernel = (math.exp(sum(math.log(r["speedup_kernel_only"]) for r in ok) / len(ok))
                 if ok else 0)
    geo_e2e = (math.exp(sum(math.log(r["speedup_e2e"]) for r in ok) / len(ok))
              if ok else 0)
    print("=" * 70)
    print(f"passed {len(ok)}/{len(results)}   geomean speedup vs CPU naive: "
          f"kernel-only {geo_kernel:.1f}x  |  end-to-end (honest) {geo_e2e:.1f}x")

    out = {"device": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0)),
           "torch": torch.__version__, "triton": triton.__version__,
           "seed": DEFAULT_SEED,
           "note": "Scored by the same frozen oracles as the CPU track, "
                   "including the calibrated perceptual thresholds. "
                   "speedup_e2e (host array in, host array out, every call) "
                   "is the honest number; speedup_kernel_only times only the "
                   "launch with data pre-staged on the GPU and is what a "
                   "fused pipeline stage would see, not what calling this "
                   "kernel from CPU-resident code costs.",
           "geomean_speedup_kernel_only": geo_kernel,
           "geomean_speedup_e2e": geo_e2e, "results": results}
    p = ROOT / "results" / "gpu_track.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote results/gpu_track.json")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
