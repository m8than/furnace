# SPDX-License-Identifier: Apache-2.0
"""gfx942 MXFP4 MoE compatibility route adapted from SGLang PR #35525.

Uses BF16 activations and unshuffled packed MXFP4 weights. The grouped
Triton dot_scaled kernels are retained locally because current AITER removed
moe_op_mxfp4 and its old config API. On CDNA3 dot_scaled emulates MXFP4 with
BF16 MFMA. Eligible long K3 prefills stage exact BF16 weights once per call
and use BF16 dot, with different accumulation order but the same rounding
boundaries. Neither path uses native FP4 or activation quantization.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import zero_copy_context


# Keep routed activation workspaces independent of the full prefill length.
_MAX_WORKSPACE_BYTES = 2 * 1024**3
_BF16_PREFILL_WORKSPACE_BYTES = 8 * 1024**3


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


# Packed-weight tiles retain the measured compatibility accumulation order.
# Staged BF16 tiles change accumulation order, not precision/rounding boundaries.
def _moe_config(
    num_tokens: int,
    *,
    is_k3: bool = False,
    down: bool = False,
    bf16_weights: bool = False,
) -> dict:
    if bf16_weights:
        return {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 1,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        }
    # K3's 896 experts leave verification batches sparse well beyond 256
    # total tokens. Avoid padding every routed expert to 64 rows.
    small = num_tokens < 256 or (is_k3 and num_tokens <= 512)
    medium = is_k3 and 512 < num_tokens <= 1024
    return {
        "BLOCK_SIZE_M": 16 if small else 32 if medium else 64,
        "BLOCK_SIZE_N": 128
        if (not small or (is_k3 and down and num_tokens >= 64))
        else 64,
        "BLOCK_SIZE_K": 512 if is_k3 and num_tokens == 1 and not down else 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 8 if is_k3 and not (small or medium) and not down else 4,
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
    Large prefills reuse bounded intermediate buffers across token chunks.
    Every chunk retains the full batch's GEMM configuration and top-k order.
    With BF16 prefill staging enabled on gfx942, contiguous TP8/EP1 K3 SiTU
    weights expand once for at least 112 Ki actual input rows. Staging is
    call-local (7,398,752,256 bytes), plus at most 8 GiB of activation scratch;
    other calls retain the packed route and 2 GiB scratch limit. BF16 MFMA
    accumulation order changes, so staged outputs need not be bitwise equal.
    """
    import triton

    from sglang.srt.layers.moe.moe_runner.mxfp4_situ_fused import (
        expand_mxfp4_bf16,
        fused_moe_mxfp4_act,
    )
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
    stage_weights = (
        tokens >= 112 * 1024
        and envs.SGLANG_K3_PREFILL_BF16_MOE.get()
        and is_k3
        and activation == "situ"
        and w13.shape == (experts, 2 * inter, hidden // 2)
        and w13_scale.shape == (experts, 2 * inter, hidden // 32)
        and w2_scale.shape == (experts, hidden, inter // 32)
        and w13_scale.dtype == w2_scale.dtype == torch.uint8
        and w13.is_contiguous()
        and w2.is_contiguous()
        and w13_scale.is_contiguous()
        and w2_scale.is_contiguous()
        and _arch() == "gfx942"
    )
    config = _moe_config(tokens, is_k3=is_k3, bf16_weights=stage_weights)
    down_config = _moe_config(
        tokens, is_k3=is_k3, down=True, bf16_weights=stage_weights
    )
    workspace_bytes = _MAX_WORKSPACE_BYTES
    if stage_weights:
        # Full weight expansion is outside the chunk loop and is not cached.
        # Contiguous eligibility avoids copying either packed weights or scales.
        w13 = expand_mxfp4_bf16(w13, w13_scale)
        w2 = expand_mxfp4_bf16(w2, w2_scale)
        workspace_bytes = _BF16_PREFILL_WORKSPACE_BYTES
    workspace_per_token = topk * (inter + hidden) * hidden_states.element_size()
    chunk_size = min(tokens, max(1, workspace_bytes // workspace_per_token))
    intermediate = hidden_states.new_empty((chunk_size * topk, inter))
    down = hidden_states.new_empty((chunk_size * topk, hidden))
    out = zero_copy_context.get_moe_output(hidden_states)
    if out is None:
        out = hidden_states.new_empty((tokens, hidden))
    reduction_block = 4096 if stage_weights else 512
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        chunk_tokens = end - start
        chunk_ids = local_ids[start:end]
        chunk_weights = topk_weights[start:end]
        if tokens == 1:
            sorted_ids = expert_ids = num_padded = None
        else:
            sorted_ids, expert_ids, num_padded = moe_align_block_size(
                chunk_ids, config["BLOCK_SIZE_M"], experts, ignore_invalid_expert=True
            )
        chunk_intermediate = intermediate[: chunk_tokens * topk]
        fused_moe_mxfp4_act(
            hidden_states[start:end],
            w13,
            chunk_intermediate,
            w13_scale,
            chunk_weights,
            chunk_ids,
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
        chunk_down = down[: chunk_tokens * topk]
        fused_moe_mxfp4_act(
            chunk_intermediate,
            w2,
            chunk_down,
            w2_scale,
            chunk_weights,
            chunk_ids,
            sorted_ids,
            expert_ids,
            num_padded,
            not apply_router_weight_on_input,
            1,
            down_config,
            activation="none",
        )
        _topk_reduce_kernel()[(chunk_tokens, triton.cdiv(hidden, reduction_block))](
            chunk_down,
            chunk_ids,
            out[start:end],
            hidden,
            TOPK=topk,
            BLOCK_H=reduction_block,
            num_warps=4,
        )
    return out
