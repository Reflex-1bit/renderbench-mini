import numpy as np

def kernel(dst: np.ndarray, src: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W = dst.shape[0], dst.shape[1]
    sh, sw = src.shape[0], src.shape[1]

    # Destination region that overlaps with source
    dst_y_start = max(dy, 0)
    dst_y_end = min(dy + sh, H)
    dst_x_start = max(dx, 0)
    dst_x_end = min(dx + sw, W)

    # No overlap
    if dst_y_start >= dst_y_end or dst_x_start >= dst_x_end:
        return dst.copy()

    # Corresponding source region
    src_y_start = dst_y_start - dy
    src_y_end = src_y_start + (dst_y_end - dst_y_start)
    src_x_start = dst_x_start - dx
    src_x_end = src_x_start + (dst_x_end - dst_x_start)

    out = dst.copy()
    out[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = src[src_y_start:src_y_end, src_x_start:src_x_end]
    return out