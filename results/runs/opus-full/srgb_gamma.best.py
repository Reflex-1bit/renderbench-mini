import numpy as np

# 4096-entry LUT built once at import time.
_v = np.linspace(0.0, 1.0, 4096, dtype=np.float64)
_s = np.where(_v <= 0.0031308, 12.92 * _v, 1.055 * np.power(_v, 1.0 / 2.4) - 0.055)
_LUT = np.clip(np.round(_s * 255.0), 0.0, 255.0).astype(np.uint8)


def kernel(linear: np.ndarray) -> np.ndarray:
    a = np.asarray(linear, dtype=np.float32)
    t = np.multiply(a, np.float32(4095.0), dtype=np.float32)
    t += np.float32(0.5)
    np.clip(t, 0.0, 4095.0, out=t)
    idx = t.astype(np.uint16, copy=False)
    return np.take(_LUT, idx)