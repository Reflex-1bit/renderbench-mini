import torch
import triton
import triton.language as tl


@triton.jit
def srgb_gamma_kernel(
    linear_ptr,
    out_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load linear values
    v = tl.load(linear_ptr + offsets, mask=mask)
    
    # Apply sRGB gamma encoding
    # s = 12.92 * v if v <= 0.0031308
    # s = 1.055 * v**(1/2.4) - 0.055 otherwise
    threshold = 0.0031308
    is_low = v <= threshold
    
    # Compute both branches
    low_result = 12.92 * v
    high_result = 1.055 * tl.math.pow(v, 1.0 / 2.4) - 0.055
    
    # Select based on threshold
    s = tl.where(is_low, low_result, high_result)
    
    # Scale to [0, 255] and round half away from zero (floor(x + 0.5) for non-negative)
    scaled = s * 255.0
    rounded = tl.floor(scaled + 0.5)
    
    # Clip to [0, 255]
    clipped = tl.minimum(tl.maximum(rounded, 0.0), 255.0)
    
    # Convert to uint8
    result = clipped.to(tl.uint8)
    
    tl.store(out_ptr + offsets, result, mask=mask)


def kernel(linear: torch.Tensor) -> torch.Tensor:
    # Ensure input is contiguous float32
    linear = linear.contiguous()
    if linear.dtype != torch.float32:
        linear = linear.float()
    
    # Flatten to 1D for processing
    num_elements = linear.numel()
    linear_flat = linear.view(-1)
    
    # Allocate output tensor
    out = torch.empty_like(linear_flat, dtype=torch.uint8)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    srgb_gamma_kernel[grid](
        linear_flat,
        out,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to original shape
    return out.view(linear.shape)