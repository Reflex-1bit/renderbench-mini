import numpy as np

def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W = dst.shape[:2]
    sh, sw = src.shape[:2]
    
    # Calculate source region that maps to valid destination
    src_y_start = max(0, -dy)
    src_y_end = min(sh, H - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(sw, W - dx)
    
    # Check if there's any overlap
    if src_y_start >= src_y_end or src_x_start >= src_x_end:
        return dst.copy()
    
    # Calculate destination region
    dst_y_start = dy + src_y_start
    dst_y_end = dy + src_y_end
    dst_x_start = dx + src_x_start
    dst_x_end = dx + src_x_end
    
    # Create output and copy
    out = dst.copy()
    out[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = src[src_y_start:src_y_end, src_x_start:src_x_end]
    return out