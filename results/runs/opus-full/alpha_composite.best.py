import numpy as np


def kernel(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    H = dst.shape[0]
    W = dst.shape[1]
    out = np.empty((H, W, 4), np.uint8)
    if H == 0 or W == 0:
        return out

    # inv = 255 - src_alpha, as uint16, shaped (H, W, 1) for broadcasting
    inv = (255 - src[..., 3].astype(np.uint16)).reshape(H, W, 1)

    # Row block sized so the uint16 scratch buffers stay in L2 cache.
    rows = 131072 // (W * 8)
    if rows < 1:
        rows = 1
    if rows > H:
        rows = H

    buf = np.empty((rows, W, 4), np.uint16)
    tmp = np.empty((rows, W, 4), np.uint16)

    for i in range(0, H, rows):
        j = i + rows
        if j > H:
            j = H
        n = j - i
        b = buf[:n]
        t = tmp[:n]

        # x = dst * inv        (max 255*255 = 65025, fits uint16)
        np.multiply(dst[i:j], inv[i:j], out=b)
        # floor(x / 255) == (x + (x >> 8) + 1) >> 8   for x <= 65025 (exact)
        np.right_shift(b, 8, out=t)
        np.add(b, t, out=b)
        np.add(b, 1, out=b)
        np.right_shift(b, 8, out=b)
        # + src, then clip to 255
        np.add(b, src[i:j], out=b)
        np.minimum(b, 255, out=b)
        np.copyto(out[i:j], b, casting='unsafe')

    return out