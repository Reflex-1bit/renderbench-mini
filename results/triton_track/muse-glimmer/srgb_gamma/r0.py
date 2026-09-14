import torch
import triton
import triton.language as tl

@triton.jit
def _srgb_gamma_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    v = tl.load(in_ptr + offs, mask=mask)
    threshold = 0.0031308

    is_low = v <= threshold
    s_low = 12.92 * v
    s_high = 1.055 * tl.math.pow(v, 1.0 / 2.4) - 0.055
    s = tl.where(is_low, s_low, s_high)

    scaled = s * 255.0
    rounded = tl.math.floor(scaled + 0.5)
    rounded = tl.maximum(0.0, tl.minimum(255.0, rounded))
    out_val = rounded.to(tl.uint8)

    tl.store(out_ptr + offs, out_val, mask=mask)


def kernel(linear: torch.Tensor) -> torch.Tensor:
    if not linear.is_contiguous():
        linear = linear.contiguous()
    out = torch.empty(linear.shape, dtype=torch.uint8, device=linear.device)
    n = linear.numel()
    BLOCK_SIZE = 1024
    grid = ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _srgb_gamma_kernel[grid](linear, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out