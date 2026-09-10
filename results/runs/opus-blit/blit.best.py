import numpy as np


def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H = dst.shape[0]
    W = dst.shape[1]
    sh = src.shape[0]
    sw = src.shape[1]
    dy = int(dy)
    dx = int(dx)

    # clipped window in destination space
    y0 = dy if dy > 0 else 0
    y1 = dy + sh
    if y1 > H:
        y1 = H
    x0 = dx if dx > 0 else 0
    x1 = dx + sw
    if x1 > W:
        x1 = W

    # nothing visible -> plain copy
    if y1 <= y0 or x1 <= x0 or sh == 0 or sw == 0:
        return dst.copy()

    out = np.empty(dst.shape, dtype=dst.dtype)

    # widen to uint32 when layout permits: same bytes, 1/4 the item count
    use32 = (
        dst.ndim == 3
        and src.ndim == 3
        and dst.dtype == np.uint8
        and src.dtype == np.uint8
        and dst.shape[2] == 4
        and src.shape[2] == 4
        and dst.flags.c_contiguous
        and src.flags.c_contiguous
    )
    if use32:
        d = dst.view(np.uint32)
        s = src.view(np.uint32)
        o = out.view(np.uint32)
    else:
        d = dst
        s = src
        o = out

    # top / bottom bands (full width)
    if y0:
        o[:y0] = d[:y0]
    if y1 < H:
        o[y1:] = d[y1:]

    mo = o[y0:y1]
    md = d[y0:y1]

    # left / right bands inside the window rows
    if x0:
        mo[:, :x0] = md[:, :x0]
    if x1 < W:
        mo[:, x1:] = md[:, x1:]

    # the hole gets the clipped source region
    mo[:, x0:x1] = s[y0 - dy:y1 - dy, x0 - dx:x1 - dx]

    return out