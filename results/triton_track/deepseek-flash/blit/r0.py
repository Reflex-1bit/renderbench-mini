import torch
import triton
import triton.language as tl


@triton.jit
def _blit_kernel(
    dst_ptr, src_ptr,
    H, W, sh, sw,
    dy, dx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total number of pixels in dst
    total_pixels = H * W
    mask = offsets < total_pixels
    
    # Convert flat index to (row, col)
    row = offsets // W
    col = offsets % W
    
    # Source coordinates
    src_row = row - dy
    src_col = col - dx
    
    # Valid source mask
    src_mask = (src_row >= 0) & (src_row < sh) & (src_col >= 0) & (src_col < sw)
    valid_mask = mask & src_mask
    
    # Load from source (4 channels)
    src_offsets = (src_row * sw + src_col) * 4
    src_offsets = src_offsets[:, None] + tl.arange(0, 4)[None, :]
    src_vals = tl.load(src_ptr + src_offsets, mask=valid_mask[:, None], other=0)
    
    # Store to destination
    dst_offsets = offsets[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(dst_ptr + dst_offsets, src_vals, mask=valid_mask[:, None])


def kernel(dst: torch.Tensor, src: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    H, W, _ = dst.shape
    sh, sw, _ = src.shape
    
    # Create a copy of dst
    out = dst.clone()
    
    # Flatten for kernel
    total_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_pixels, BLOCK_SIZE),)
    
    _blit_kernel[grid](
        out, src,
        H, W, sh, sw,
        dy, dx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out