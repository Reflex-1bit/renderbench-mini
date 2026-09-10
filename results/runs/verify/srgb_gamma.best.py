import numpy as np

_N = 4096
_x = (np.arange(_N) + 0.5) / _N
_s = np.where(_x <= 0.0031308, _x * 12.92,
              1.055 * np.power(_x, 1.0 / 2.4) - 0.055)
_LUT = np.clip(np.rint(_s * 255.0), 0, 255).astype(np.uint8)

def kernel(linear):
    idx = np.clip((linear * _N).astype(np.int32), 0, _N - 1)
    return _LUT[idx]