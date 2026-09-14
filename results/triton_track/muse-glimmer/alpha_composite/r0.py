import torch
import triton
import triton.language as tl

@triton.jit
def _alpha_composite_kernel(dst_ptr, src_ptr, out_ptr, H, W, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pixels = H * W
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pixels

    pixel_offsets = offsets * 4
    c = tl.arange(0, 4)

    dst_vals = tl.load(
        dst_ptr + pixel_offsets[:, None] + c[None, :],
        mask=mask[:, None],
        other=0
    )
    src_vals = tl.load(
        src_ptr + pixel_offsets[:, None] + c[None, :],
        mask=mask[:, None],
        other=0
    )

    dst_i = dst_vals.to(tl.int32)
    src_i = src_vals.to(tl.int32)

    alpha = src_i[:, 3]
    factor = 255 - alpha
    factor_b = factor[:, None]

    tmp = dst_i * factor_b
    tmp = tmp // 255
    out_i = src_i + tmp

    out_i = tl.where(out_i < 0, 0, out_i)
    out_i = tl.where(out_i > 255, 255, out_i)

    tl.store(
        out_ptr + pixel_offsets[:, None] + c[None, :],
        out_i.to(tl.uint8),
        mask=mask[:, None]
    )


def kernel(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    assert dst.shape == src.shape and dst.shape[-1] == 4
    H, W = dst.shape[0], dst.shape[1]

    dst_c = dst.contiguous()
    src_c = src.contiguous()
    out = torch.empty_like(dst_c)

    BLOCK_SIZE = 1024
    num_pixels = H * W
    grid = ( (num_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _alpha_composite_kernel[grid](
        dst_c.view(-1),
        src_c.view(-1),
        out.view(-1),
        H,
        W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out