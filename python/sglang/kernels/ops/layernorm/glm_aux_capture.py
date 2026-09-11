"""BF16 hc4 DFlash capture into an existing packed destination (ROCm only)."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _add_f32_rn(a, b):
    # Keep both FP32 rounding and the +0 identity operations in Reduce.cuh.
    # An explicit instruction prevents reassociation/identity elimination.
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        constraints="=v,v,v",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _capture_hc4(
    Hidden,
    Residual,
    Destination,
    HIDDEN_ROW: tl.constexpr,
    RESIDUAL_ROW: tl.constexpr,
    DESTINATION_ROW: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # This is deliberately not tl.sum. PyTorch 2.11 ReduceMomentKernel uses
    # vt0=4; Reduce.cuh does not split a four-element, non-inner reduction
    # across threads. Its four +0 accumulators combine left-to-right.
    total = tl.full((BLOCK,), 0, tl.float32)
    for stream in tl.static_range(4):
        offset = stream * 4096 + col
        value = tl.load(Hidden + row * HIDDEN_ROW + offset).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(Residual + row * RESIDUAL_ROW + offset).to(tl.float32)
            # torch.add materializes BF16 BEFORE hc_contract's FP32 mean.
            value = _add_f32_rn(value, residual).to(tl.bfloat16).to(tl.float32)
        value = _add_f32_rn(tl.full((BLOCK,), 0, tl.float32), value)
        if stream == 0:
            total = value
        else:
            total = _add_f32_rn(total, value)
    tl.store(Destination + row * DESTINATION_ROW + col, (total * 0.25).to(tl.bfloat16))


def can_capture_hc4(hidden: torch.Tensor, residual: Optional[torch.Tensor]) -> bool:
    return (
        torch.version.hip is not None
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and hidden.ndim == 2
        and hidden.shape[1] == 16384
        and hidden.stride(1) == 1
        and (
            residual is None
            or (
                residual.shape == hidden.shape
                and residual.dtype == hidden.dtype
                and residual.device == hidden.device
                and residual.stride(1) == 1
            )
        )
    )


def capture_hc4_into(
    hidden: torch.Tensor, residual: Optional[torch.Tensor], destination: torch.Tensor
) -> None:
    """Write exactly one [tokens,4096] capture; never allocate or fall back.

    Inputs may have padded row strides. Destination is normally reserve_next's
    [tokens,4096] view into [tokens,K*4096]. Empty captures do not launch.
    """
    if not can_capture_hc4(hidden, residual):
        raise ValueError("direct hc4 capture requires ROCm BF16 [tokens,16384] inputs")
    if (
        destination.shape != (hidden.shape[0], 4096)
        or destination.dtype != hidden.dtype
        or destination.device != hidden.device
        or destination.stride(1) != 1
        or destination.stride(0) < 4096
    ):
        raise ValueError("invalid direct hc4 capture destination")
    if hidden.shape[0] == 0:
        return
    _capture_hc4[(hidden.shape[0], 16)](
        hidden,
        residual if residual is not None else hidden,
        destination,
        hidden.stride(0),
        residual.stride(0) if residual is not None else 0,
        destination.stride(0),
        residual is not None,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
