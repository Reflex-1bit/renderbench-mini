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

    alpha = tl.load(src_ptr + offsets * 4 + 3, mask=mask).to(tl.int32)
    factor = 255 - alpha

    # channel 0
    dst_val = tl.load(dst_ptr + offsets * 4 + 0, mask=mask).to(tl.int32)
    src_val = tl.load(src_ptr + offsets * 4 + 0, mask=mask).to(tl.int32)
    out_val = src_val + (dst_val * factor) // 255
    out_val = tl.where(out_val < 0, 0, out_val)
    out_val = tl.where(out_val > 255, 255, out_val)
    tl.store(out_ptr + offsets * 4 + 0, out_val.to(tl.uint8), mask=mask)

    # channel 1
    dst_val = tl.load(dst_ptr + offsets * 4 + 1, mask=mask).to(tl.int32)
    src_val = tl.load(src_ptr + offsets * 4 + 1, mask=mask).to(tl.int32)
    out_val = src_val + (dst_val * factor) // 255
    out_val = tl.where(out_val < 0, 0, out_val)
    out_val = tl.where(out_val > 255, 255, out_val)
    tl.store(out_ptr + offsets * 4 + 1, out_val.to(tl.uint8), mask=mask)

    # channel 2
    dst_val = tl.load(dst_ptr + offsets * 4 + 2, mask=mask).to(tl.int32)
    src_val = tl.load(src_ptr + offsets * 4 + 2, mask=mask).to(tl.int32)
    out_val = src_val + (dst_val * factor) // 255
    out_val = tl.where(out_val < 0, 0, out_val)
    out_val = tl.where(out_val > 255, 255, out_val)
    tl.store(out_ptr + offsets * 4 + 2, out_val.to(tl.uint8), mask=mask)

    # channel 3
    dst_val = tl.load(dst_ptr + offsets * 4 + 3, mask=mask).to(tl.int32)
    src_val = tl.load(src_ptr + offsets * 4 + 3, mask=mask).to(tl.int32)
    out_val = src_val + (dst_val * factor) // 255
    out_val = tl.where(out_val < 0, 0, out_val)
    out_val = tl.where(out_val > 255, 255, out_val)
    tl.store(out_ptr + offsets * 4 + 3, out_val.to(tl.uint8), mask=mask)


def kernel(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    assert dst.shape == src.shape and dst.shape[-1] == 4
    H, W = dst.shape[0], dst.shape[1]

    dst_c = dst.contiguous()
    src_c = src.contiguous()
    out = torch.empty_like(dst_c)

    BLOCK_SIZE = 1024
    num_pixels = H * W
    grid = ((num_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _alpha_composite_kernel[grid](
        dst_c.view(-1),
        src_c.view(-1),
        out.view(-1),
        H,
        W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out