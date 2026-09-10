import numpy as np

def kernel(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    d = dst.astype(np.uint16)
    s = src.astype(np.uint16)
    a = 255 - s[..., 3:4]  # (H, W, 1) uint16
    prod = d * a  # uint16, max 65025
    out = s + prod // 255
    out = np.minimum(out, 255)
    return out.astype(np.uint8)