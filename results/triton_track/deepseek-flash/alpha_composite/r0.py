import torch
import triton
import triton.language as tl


@triton.jit
def _alpha_composite_kernel(
    dst_ptr, src_ptr, out_ptr,
    H, W, total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute pixel indices and channel indices
    pixel_idx = offsets // 4
    channel_idx = offsets % 4
    
    # Load src values
    src_offsets = pixel_idx * 4 + channel_idx
    src_vals = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    
    # Load dst values
    dst_vals = tl.load(dst_ptr + src_offsets, mask=mask, other=0)
    
    # Load src alpha (channel 3) for each pixel
    src_alpha_offsets = pixel_idx * 4 + 3
    src_alpha = tl.load(src_ptr + src_alpha_offsets, mask=mask, other=0)
    
    # Compute: out = src + (dst * (255 - src_alpha)) // 255
    src_vals_i32 = src_vals.to(tl.int32)
    dst_vals_i32 = dst_vals.to(tl.int32)
    src_alpha_i32 = src_alpha.to(tl.int32)
    
    factor = (255 - src_alpha_i32).to(tl.int32)
    product = dst_vals_i32 * factor
    quotient = product // 255
    
    result = src_vals_i32 + quotient
    
    # Clip to [0, 255]
    result = tl.minimum(tl.maximum(result, 0), 255)
    result = result.to(tl.uint8)
    
    tl.store(out_ptr + src_offsets, result, mask=mask)


def kernel(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    assert dst.shape == src.shape, "dst and src must have the same shape"
    assert dst.dtype == torch.uint8 and src.dtype == torch.uint8
    
    H, W, C = dst.shape
    assert C == 4, "Expected 4 channels"
    
    out = torch.empty_like(dst)
    total_elements = H * W * C
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    _alpha_composite_kernel[grid](
        dst, src, out,
        H, W, total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out