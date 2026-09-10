import numpy as np


def kernel(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src = np.asarray(src)
    sh = src.shape[0]
    sw = src.shape[1]
    ch = src.shape[2] if src.ndim == 3 else 1

    if out_h == sh and out_w == sw:
        return src.copy()

    # ---- horizontal mapping -------------------------------------------------
    sx = (np.arange(out_w, dtype=np.float64) + 0.5) * (float(sw) / out_w) - 0.5
    np.clip(sx, 0.0, sw - 1.0, out=sx)
    x0 = np.floor(sx).astype(np.intp)
    wx = (sx - x0).astype(np.float32)
    x1 = x0 + 1
    np.minimum(x1, sw - 1, out=x1)

    # ---- vertical mapping ---------------------------------------------------
    sy = (np.arange(out_h, dtype=np.float64) + 0.5) * (float(sh) / out_h) - 0.5
    np.clip(sy, 0.0, sh - 1.0, out=sy)
    y0 = np.floor(sy).astype(np.intp)
    wy = (sy - y0).astype(np.float32)
    y1 = y0 + 1
    np.minimum(y1, sh - 1, out=y1)

    # ---- restrict source rows when downscaling ------------------------------
    if 2 * out_h < sh:
        rows = np.unique(np.concatenate((y0, y1)))
        sub = src[rows]
        y0i = np.searchsorted(rows, y0)
        y1i = np.searchsorted(rows, y1)
    else:
        sub = src
        y0i = y0
        y1i = y1

    n = sub.shape[0]
    flat = np.ascontiguousarray(sub).reshape(n, sw * ch)

    off = np.arange(ch, dtype=np.intp)
    i0 = (x0[:, None] * ch + off).ravel()
    i1 = (x1[:, None] * ch + off).ravel()

    # ---- pass 1: horizontal blend over the (few) needed source rows ---------
    a = np.take(flat, i0, axis=1).astype(np.float32)
    b = np.take(flat, i1, axis=1).astype(np.float32)
    np.subtract(b, a, out=b)
    b *= np.repeat(wx, ch)
    a += b
    del b, flat

    # ---- pass 2: vertical blend into the output ----------------------------
    t = np.take(a, y0i, axis=0)
    u = np.take(a, y1i, axis=0)
    del a
    np.subtract(u, t, out=u)
    u *= wy[:, None]
    t += u
    del u
    t += 0.5

    out = np.empty((out_h, out_w * ch), dtype=np.uint8)
    out[:] = t
    return out.reshape(out_h, out_w, ch)