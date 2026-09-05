# SPDX-License-Identifier: Apache-2.0
"""gfx942 MXFP4 MoE compatibility route adapted from SGLang PR #35525.

Uses BF16 activations and unshuffled packed MXFP4 weights. The grouped
Triton dot_scaled kernels are retained locally because current AITER removed
moe_op_mxfp4 and its old config API. On CDNA3 dot_scaled emulates MXFP4 with
BF16 MFMA, bypassing the unsupported FlyDSL/native FP4 path entirely.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch


@functools.lru_cache(maxsize=1)
def _arch() -> str:
    if not torch.version.hip or not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.split(":")[0]


@functools.lru_cache(maxsize=1)
def use_triton_mxfp4_moe() -> bool:
    """Auto-enable on gfx942; =0 disables, =1 still requires gfx942.

    Import failures on gfx942 are fatal rather than silently falling back to
    the native FP4 path known not to work on this architecture.
    """
    forced = os.environ.get("SGLANG_AITER_MXFP4_TRITON")
    if forced not in (None, "0", "1"):
        raise ValueError("SGLANG_AITER_MXFP4_TRITON must be 0 or 1")
    if forced == "0" or _arch() != "gfx942":
        return False
    from sglang.srt.layers.moe.moe_runner import mxfp4_situ_fused  # noqa: F401

    return True


# CDNA3-safe tiles, with measured K3 TP8 refinements that preserve the
# K reduction and BF16 rounding. Other model shapes keep the compatibility tiles.
def _moe_config(num_tokens: int, *, is_k3: bool = False, down: bool = False) -> dict:
    small = num_tokens < 256
    return {
        "BLOCK_SIZE_M": 16 if small else 64,
        "BLOCK_SIZE_N": 128
        if (not small or (is_k3 and down and num_tokens >= 64))
        else 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 8 if is_k3 and not small and not down else 4,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }


@functools.lru_cache(maxsize=8)
def _global_to_local_table(num_global: int, num_local: int, ep_rank: int, device: str):
    # Extra last element preserves the -1 sentinel under tensor indexing.
    table = torch.full((num_global + 1,), -1, dtype=torch.int32, device=device)
    lo = ep_rank * num_local
    hi = min(lo + num_local, num_global)
    if hi > lo:
        table[lo:hi] = torch.arange(hi - lo, dtype=torch.int32, device=device)
    return table


@functools.lru_cache(maxsize=1)
def _topk_reduce_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        down_ptr,
        ids_ptr,
        out_ptr,
        H: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            local_id = tl.load(ids_ptr + token * TOPK + k)
            # Unowned rows were not written: mask the load, not the value,
            # because 0 * an uninitialized NaN still poisons the reduction.
            value = tl.load(
                down_ptr + (token * TOPK + k) * H + offs,
                mask=(offs < H) & (local_id >= 0),
                other=0.0,
            )
            acc += value.to(tl.float32)
        tl.store(
            out_ptr + token * H + offs, acc.to(out_ptr.dtype.element_ty), mask=offs < H
        )

    return _kernel


def fused_moe_mxfp4_triton(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    num_global_experts: Optional[int] = None,
    ep_rank: int = 0,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Compute MoE from separated [gate; up] weights, retaining layer padding.

    w13 [E, 2I, H/2], w2 [E, H, I/2], E8M0 scales in groups of 32.
    Input/output use the allocated padded H; zero-padded gate/up rows and
    down columns are consumed in place, without moving the gate/up boundary.
    Expert IDs are global (StandardDispatcher's AITER contract).
    """
    import triton

    from sglang.srt.layers.moe.moe_runner.mxfp4_situ_fused import fused_moe_mxfp4_act
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    if activation not in ("silu", "situ"):
        raise NotImplementedError(f"Unsupported gfx942 MXFP4 activation: {activation}")
    if hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError("gfx942 MXFP4 Triton MoE requires BF16 activations")
    w13, w2 = w13_weight.view(torch.uint8), w2_weight.view(torch.uint8)
    experts, n13, _ = w13.shape
    hidden, inter = w2.shape[1], n13 // 2
    tokens, topk = topk_ids.shape
    assert hidden_states.shape == (tokens, hidden)
    assert w2.shape == (experts, hidden, inter // 2)
    assert topk_weights.shape == topk_ids.shape
    if tokens == 0:
        return hidden_states.new_empty((0, hidden))
    if num_global_experts is not None and num_global_experts != experts:
        if (
            num_global_experts % experts
            or not 0 <= ep_rank < num_global_experts // experts
        ):
            raise NotImplementedError(
                "gfx942 MXFP4 requires evenly partitioned routed experts"
            )
        table = _global_to_local_table(
            num_global_experts, experts, ep_rank, str(hidden_states.device)
        )
        local_ids = table[topk_ids.to(torch.long)]
    else:
        local_ids = topk_ids.to(torch.int32)
    local_ids = local_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    is_k3 = (experts, hidden, inter, topk) == (896, 3584, 384, 16)
    config = _moe_config(tokens, is_k3=is_k3)
    if tokens == 1:
        sorted_ids = expert_ids = num_padded = None
    else:
        sorted_ids, expert_ids, num_padded = moe_align_block_size(
            local_ids, config["BLOCK_SIZE_M"], experts, ignore_invalid_expert=True
        )
    intermediate = hidden_states.new_empty((tokens * topk, inter))
    fused_moe_mxfp4_act(
        hidden_states,
        w13,
        intermediate,
        w13_scale,
        topk_weights,
        local_ids,
        sorted_ids,
        expert_ids,
        num_padded,
        apply_router_weight_on_input,
        topk,
        config,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    down = hidden_states.new_empty((tokens * topk, hidden))
    fused_moe_mxfp4_act(
        intermediate,
        w2,
        down,
        w2_scale,
        topk_weights,
        local_ids,
        sorted_ids,
        expert_ids,
        num_padded,
        not apply_router_weight_on_input,
        1,
        _moe_config(tokens, is_k3=is_k3, down=True),
        activation="none",
    )
    out = hidden_states.new_empty((tokens, hidden))
    _topk_reduce_kernel()[(tokens, triton.cdiv(hidden, 512))](
        down, local_ids, out, hidden, TOPK=topk, BLOCK_H=512, num_warps=4
    )
    return out
