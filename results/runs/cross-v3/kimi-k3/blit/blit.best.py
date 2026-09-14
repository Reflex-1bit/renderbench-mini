import numpy as np


def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = dst.copy()
    H, W = dst.shape[0], dst.shape[1]
    sh, sw = src.shape[0], src.shape[1]

    # Source region that lands inside dst
    sy0 = 0 if dy >= 0 else -dy
    sx0 = 0 if dx >= 0 else -dx
    sy1 = sh if dy + sh <= H else H - dy
    sx1 = sw if dx + sw <= W else W - dx

    if sy1 > sy0 and sx1 > sx0:
        out[dy + sy0:dy + sy1, dx + sx0:dx + sx1] = src[sy0:sy1, sx0:sx1]
    return out