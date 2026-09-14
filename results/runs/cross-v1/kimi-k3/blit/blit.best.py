import numpy as np


def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = dst.copy()
    H, W = dst.shape[0], dst.shape[1]
    sh, sw = src.shape[0], src.shape[1]

    y0 = dy if dy > 0 else 0
    x0 = dx if dx > 0 else 0
    y1 = dy + sh
    if y1 > H:
        y1 = H
    x1 = dx + sw
    if x1 > W:
        x1 = W

    if y0 < y1 and x0 < x1:
        sy0 = y0 - dy
        sx0 = x0 - dx
        region = src[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        if not region.flags.c_contiguous:
            region = np.ascontiguousarray(region)
        out[y0:y1, x0:x1] = region
    return out