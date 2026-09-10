import numpy as np


def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    # Fresh destination copy (never mutate inputs).
    out = dst.copy()

    H = dst.shape[0]
    W = dst.shape[1]
    sh = src.shape[0]
    sw = src.shape[1]

    # Clip the source rectangle against the destination bounds.
    # Source rows/cols that would land outside [0,H) x [0,W) are dropped.
    sy0 = -dy if dy < 0 else 0
    sx0 = -dx if dx < 0 else 0
    over_y = dy + sh - H
    over_x = dx + sw - W
    sy1 = sh - over_y if over_y > 0 else sh
    sx1 = sw - over_x if over_x > 0 else sw

    if sy1 > sy0 and sx1 > sx0:
        # Single strided block assignment in C: no per-pixel Python work.
        out[dy + sy0:dy + sy1, dx + sx0:dx + sx1] = src[sy0:sy1, sx0:sx1]

    return out