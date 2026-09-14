import numpy as np


def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    # Full copy of the destination (single contiguous memcpy when possible).
    out = dst.copy()

    H = dst.shape[0]
    W = dst.shape[1]
    sh = src.shape[0]
    sw = src.shape[1]

    # Intersection of the placed src rect [dy, dy+sh) x [dx, dx+sw)
    # with the destination bounds [0, H) x [0, W).
    y0 = dy if dy > 0 else 0
    y1 = dy + sh
    if y1 > H:
        y1 = H
    x0 = dx if dx > 0 else 0
    x1 = dx + sw
    if x1 > W:
        x1 = W

    if y0 < y1 and x0 < x1:
        # Vectorized strided copy of the overlapping region only.
        out[y0:y1, x0:x1] = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]

    return out