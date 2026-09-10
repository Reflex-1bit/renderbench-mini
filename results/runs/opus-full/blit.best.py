import numpy as np


def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H = dst.shape[0]
    W = dst.shape[1]
    sh = src.shape[0]
    sw = src.shape[1]

    dy = int(dy)
    dx = int(dx)

    # Destination-space clipped window
    y0 = dy if dy > 0 else 0
    x0 = dx if dx > 0 else 0
    y1 = dy + sh
    x1 = dx + sw
    if y1 > H:
        y1 = H
    if x1 > W:
        x1 = W

    if y1 <= y0 or x1 <= x0:
        return dst.copy()

    # Full coverage: the window is the whole destination -> pure src copy
    if y0 == 0 and x0 == 0 and y1 == H and x1 == W:
        out = np.empty_like(dst)
        out[...] = src[-dy:H - dy, -dx:W - dx]
        return out

    out = np.empty_like(dst)

    # Top band
    if y0 > 0:
        out[:y0] = dst[:y0]
    # Bottom band
    if y1 < H:
        out[y1:] = dst[y1:]
    # Left strip
    if x0 > 0:
        out[y0:y1, :x0] = dst[y0:y1, :x0]
    # Right strip
    if x1 < W:
        out[y0:y1, x1:] = dst[y0:y1, x1:]

    # Window from src
    out[y0:y1, x0:x1] = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out