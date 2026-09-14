import torch
import triton
import triton.language as tl


@triton.jit
def _composite_kernel(dst_ptr, src_ptr, out_ptr, H, W, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load src (4 channels interleaved)
    src_offsets = offsets * 4 + tl.arange(0, 4)[None, :]
    src_mask = mask[:, None] & (tl.arange(0, 4)[None, :] < 4)
    src = tl.load(src_ptr + src_offsets, mask=src_mask, other=0).to(tl.int32)

    # Load dst
    dst = tl.load(dst_ptr + src_offsets, mask=src_mask, other=0).to(tl.int32)

    # Extract alpha from src (channel 3)
    src_alpha = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 3, src, 0), axis=1)  # shape [BLOCK]

    # Compute per-channel composite
    # out[c] = clip(src[c] + (dst[c] * (255 - src_alpha)) // 255, 0, 255)
    factor = 255 - src_alpha  # shape [BLOCK]
    factor = factor[:, None]  # [BLOCK, 1]

    # Compute dst * factor // 255 in int32 (safe, max 255*255=65025)
    dst_scaled = (dst * factor) // 255  # [BLOCK, 4]

    out = src + dst_scaled
    out = tl.minimum(tl.maximum(out, 0), 255)

    # Store as uint8
    out = out.to(tl.uint8)
    tl.store(out_ptr + src_offsets, out, mask=src_mask)


def kernel(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """
    Premultiplied-alpha "source over" composite.
    dst, src: (H, W, 4) uint8 CUDA tensors.
    Returns new (H, W, 4) uint8 tensor.
    """
    assert dst.is_cuda and src.is_cuda
    assert dst.shape == src.shape and dst.dtype == torch.uint8 and src.dtype == torch.uint8
    H, W, _ = dst.shape
    total = H * W
    out = torch.empty_like(dst)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    _composite_kernel[grid](
        dst, src, out, H, W, total, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
