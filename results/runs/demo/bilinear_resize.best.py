import numpy as np

def kernel(src, out_h, out_w):
    out_h, out_w = int(out_h), int(out_w)
    sh, sw = src.shape[:2]
    s = src.astype(np.float32)

    sy = np.clip((np.arange(out_h) + 0.5) * sh / out_h - 0.5, 0, sh - 1)
    y0 = np.floor(sy).astype(np.int64); y1 = np.minimum(y0 + 1, sh - 1)
    wy = (sy - y0).astype(np.float32)[:, None, None]
    rows = s[y0] * (1 - wy) + s[y1] * wy

    sx = np.clip((np.arange(out_w) + 0.5) * sw / out_w - 0.5, 0, sw - 1)
    x0 = np.floor(sx).astype(np.int64); x1 = np.minimum(x0 + 1, sw - 1)
    wx = (sx - x0).astype(np.float32)[None, :, None]
    out = rows[:, x0] * (1 - wx) + rows[:, x1] * wx

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)