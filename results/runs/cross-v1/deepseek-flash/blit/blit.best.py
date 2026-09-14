import numpy as np

def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W = dst.shape[:2]
    sh, sw = src.shape[:2]

    r0 = max(0, dy)
    r1 = min(H, dy + sh)
    c0 = max(0, dx)
    c1 = min(W, dx + sw)

    if r0 >= r1 or c0 >= c1:
        return dst.copy()

    sr0 = r0 - dy
    sr1 = r1 - dy
    sc0 = c0 - dx
    sc1 = c1 - dx

    out = dst.copy()
    out[r0:r1, c0:c1, :] = src[sr0:sr1, sc0:sc1, :]
    return out