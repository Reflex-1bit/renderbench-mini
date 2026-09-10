import numpy as np

def kernel(dst, src, dy, dx):
    out = dst.copy()
    H, W = out.shape[:2]
    sh, sw = src.shape[:2]
    dy, dx = int(dy), int(dx)

    y0, x0 = max(0, dy), max(0, dx)
    y1, x1 = min(H, dy + sh), min(W, dx + sw)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out