import numpy as np

def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = dst.copy()
    sh, sw = src.shape[0], src.shape[1]
    H, W = dst.shape[0], dst.shape[1]

    y0 = dy if dy > 0 else 0
    y1 = dy + sh
    if y1 > H:
        y1 = H
    if y0 >= y1:
        return out

    x0 = dx if dx > 0 else 0
    x1 = dx + sw
    if x1 > W:
        x1 = W
    if x0 >= x1:
        return out

    src_y0 = y0 - dy
    src_x0 = x0 - dx

    out[y0:y1, x0:x1] = src[src_y0:src_y0 + (y1 - y0), src_x0:src_x0 + (x1 - x0)]
    return out