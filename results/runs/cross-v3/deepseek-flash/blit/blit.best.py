import numpy as np

def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W, _ = dst.shape
    sh, sw, _ = src.shape
    
    # Compute the valid source region that fits in dst
    src_y0 = max(0, -dy)
    src_y1 = min(sh, H - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(sw, W - dx)
    
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        # Nothing overlaps; return a copy of dst
        return dst.copy()
    
    # Destination region corresponding to the valid source region
    dst_y0 = dy + src_y0
    dst_x0 = dx + src_x0
    dst_y1 = dy + src_y1
    dst_x1 = dx + src_x1
    
    # Copy the valid source block into the destination copy
    out = dst.copy()
    out[dst_y0:dst_y1, dst_x0:dst_x1, :] = src[src_y0:src_y1, src_x0:src_x1, :]
    return out