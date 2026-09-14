import numpy as np

def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W, _ = dst.shape
    sh, sw, _ = src.shape

    y0_dst = dy if dy > 0 else 0
    if y0_dst < 0:
        y0_dst = 0
    y1_dst = dy + sh
    if y1_dst > H:
        y1_dst = H
    x0_dst = dx if dx > 0 else 0
    if x0_dst < 0:
        x0_dst = 0
    x1_dst = dx + sw
    if x1_dst > W:
        x1_dst = W

    # use max/min for clarity and correctness with large values
    y0_dst = max(dy, 0)
    y1_dst = min(dy + sh, H)
    x0_dst = max(dx, 0)
    x1_dst = min(dx + sw, W)

    if y0_dst >= y1_dst or x0_dst >= x1_dst:
        return dst.copy()

    src_y0 = y0_dst - dy
    src_x0 = x0_dst - dx
    src_y1 = src_y0 + (y1_dst - y0_dst)
    src_x1 = src_x0 + (x1_dst - x0_dst)

    out = dst.copy()
    out[y0_dst:y1_dst, x0_dst:x1_dst] = src[src_y0:src_y1, src_x0:src_x1]
    return out