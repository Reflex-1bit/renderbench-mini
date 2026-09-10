import numpy as np

def kernel(page, atlas, gid, gy, gx, rgb):
    out = page.astype(np.float32).copy()
    n, gh, gw = atlas.shape
    m = len(gid)

    y0 = np.floor(gy).astype(np.int64)
    x0 = np.floor(gx).astype(np.int64)
    fy = (gy - y0).astype(np.float32)[:, None, None]
    fx = (gx - x0).astype(np.float32)[:, None, None]

    pad = np.zeros((m, gh + 1, gw + 1), dtype=np.float32)
    pad[:, :gh, :gw] = atlas[gid]

    cov = (pad[:, :gh, :gw] * ((1 - fy) * (1 - fx))
           + pad[:, 1:, :gw] * (fy * (1 - fx))
           + pad[:, :gh, 1:] * ((1 - fy) * fx)
           + pad[:, 1:, 1:] * (fy * fx))
    cov *= np.float32(1.0 / 255.0)

    inv = 1.0 - cov
    pre = cov[..., None] * rgb.astype(np.float32)[:, None, None, :]

    for i in range(m):
        tile = out[y0[i]:y0[i] + gh, x0[i]:x0[i] + gw]
        tile *= inv[i][..., None]
        tile += pre[i]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)