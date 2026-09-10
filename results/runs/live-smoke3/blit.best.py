import numpy as np


def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H = dst.shape[0]
    W = dst.shape[1]
    sh = src.shape[0]
    sw = src.shape[1]

    # Vertical clip of the source rectangle against destination bounds.
    y0 = dy if dy > 0 else 0
    y1 = dy + sh
    if y1 > H:
        y1 = H
    if y0 >= y1:
        return dst.copy()

    # Horizontal clip.
    x0 = dx if dx > 0 else 0
    x1 = dx + sw
    if x1 > W:
        x1 = W
    if x0 >= x1:
        return dst.copy()

    sy0 = y0 - dy
    sy1 = y1 - dy
    sx0 = x0 - dx
    sx1 = x1 - dx

    # Fast path: the blit covers the entire destination -> result is just src.
    if y0 == 0 and x0 == 0 and y1 == H and x1 == W:
        return src[sy0:sy1, sx0:sx1].copy()

    # General path: one contiguous memcpy of dst, then overwrite the
    # (already clipped) intersection with the source pixels.
    out = dst.copy()
    out[y0:y1, x0:x1] = src[sy0:sy1, sx0:sx1]
    return out