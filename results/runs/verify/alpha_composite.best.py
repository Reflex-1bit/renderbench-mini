import numpy as np

def kernel(dst, src):
    s16 = src.astype(np.uint16)
    scratch = dst.astype(np.uint16)
    inv = 255 - s16[:, :, 3:4]

    scratch *= inv
    scratch //= 255
    scratch += s16
    np.minimum(scratch, 255, out=scratch)
    return scratch.astype(np.uint8)