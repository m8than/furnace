# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Derived from AITER 9127c94a's moe_op_mxfp4_silu_fused kernels and
# moonmath-ai/sglang 8747f6bf9c15eda2ba8abfb3940ed5673b48fba6 (PR #35525).
"""CDNA3 A16W4 grouped GEMM, with optional SwiGLU/SiTUv2 epilogue.

Current AITER no longer ships the PR's moe_op_mxfp4 entry points. Keep their
unshuffled MXFP4 dot_scaled implementation here, restricted to BF16 inputs
and unswizzled weights. Long K3 prefills may instead stage the weights as
exact BF16 values and use BF16 dot; both lower to BF16 MFMA on gfx942.
No native FP4 instructions or activation quantization are needed.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd


@triton.jit
def _expand_mxfp4_bf16_kernel(
    src, scales, dst, COUNT: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offset < COUNT * 2
    byte = tl.load(src + offset // 2, mask, other=0).to(tl.uint32)
    scale = tl.load(scales + offset // 32, mask, other=0).to(tl.int32)
    code = (byte >> ((offset % 2) * 4)) & 15
    magnitude = (code & 7).to(tl.int32)
    exponent = scale + magnitude // 2 - 1
    mantissa = tl.where(magnitude == 1, 0, (magnitude & 1) << 6)
    normal = (exponent << 7) | mantissa
    subnormal = (128 + mantissa) >> tl.maximum(1 - exponent, 0)
    bits = tl.where(exponent < 1, subnormal, normal)
    bits = tl.where(exponent >= 255, 0x7F80, bits)
    bits = tl.where(magnitude == 0, 0, bits)
    bits = tl.where(scale == 255, 0x7FC0, bits)
    bits = bits | ((code & 8) << 12)
    value = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(dst + offset, value, mask)


def expand_mxfp4_bf16(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Expand all E2M1/E8M0 codes, including signed zero, subnormals, Inf/NaN.

    Finite BF16 bits and nonfinite categories match direct BF16 conversion.
    This is lossless weight staging, not bitwise-equivalent GEMM accumulation.
    The caller owns the temporary allocation; no expanded weights are cached.
    """
    assert packed.dtype == scales.dtype == torch.uint8
    assert packed.is_contiguous() and scales.is_contiguous()
    assert packed.shape[-1] % 16 == 0
    assert tuple(scales.shape) == (*packed.shape[:-1], packed.shape[-1] // 16)
    expanded = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        device=packed.device,
        dtype=torch.bfloat16,
    )
    _expand_mxfp4_bf16_kernel[(triton.cdiv(expanded.numel(), 2048),)](
        packed, scales, expanded, packed.numel(), BLOCK=2048
    )
    return expanded


@triton.jit
def _sigmoid_exp2(x):
    return 1.0 / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _situ_and_mul(gate, up, BETA: tl.constexpr, LINEAR_BETA: tl.constexpr):
    tanh_g = 2.0 * _sigmoid_exp2(gate * (2.0 / BETA)) - 1.0
    tanh_u = 2.0 * _sigmoid_exp2(up * (2.0 / LINEAR_BETA)) - 1.0
    return (BETA * tanh_g) * _sigmoid_exp2(gate) * (LINEAR_BETA * tanh_u)


@triton.jit
def _fused_moe_kernel_mxfp4_act(
    a_ptr,
    b_ptr,
    c_ptr,
    b_mx_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bmxe: tl.constexpr,
    stride_bmxk: tl.constexpr,
    stride_bmxn: tl.constexpr,
    A_DTYPE_FORMAT: tl.constexpr,
    BF16_WEIGHTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    DIRECT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    SITU_BETA: tl.constexpr,
    SITU_LINEAR_BETA: tl.constexpr,
):
    pid = tl.program_id(0)
    if DIRECT:
        num_pid_m = num_valid_tokens
    else:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_mn = num_pid_m * num_pid_n
    if pid >= grid_mn:
        return
    pid = remap_xcd(pid, grid_mn, 8)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    if DIRECT:
        # A single token has one independent row per routed expert. Preserve
        # the GEMM tile and arithmetic, but synthesize its padding in registers
        # instead of sorting experts and materializing token-index buffers.
        offs_token = tl.where(
            tl.arange(0, BLOCK_SIZE_M) == 0, pid_m, num_valid_tokens
        ).to(tl.int64)
    else:
        offs_token = tl.load(
            sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        ).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    FUSED_GATE: tl.constexpr = ACTIVATION != "none"
    OUTPUT_BLOCK_N: tl.constexpr = BLOCK_SIZE_N // 2 if FUSED_GATE else BLOCK_SIZE_N
    OUTPUT_N: tl.constexpr = N // 2 if FUSED_GATE else N
    offs_cn = pid_n * OUTPUT_BLOCK_N + tl.arange(0, OUTPUT_BLOCK_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < OUTPUT_N)
    if expert == -1:
        # Fused gate/up has N/2 output columns, including on this EP path.
        tl.store(c_ptrs, 0.0, mask=c_mask)
        return

    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    if FUSED_GATE:
        # Adjacent accumulator lanes are a gate/up pair from separated halves.
        offs_bn = (pid_n * OUTPUT_BLOCK_N + i // 2) % (N // 2) + (i % 2) * (N // 2)
    else:
        offs_bn = (pid_n * BLOCK_SIZE_N + i) % N
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    WEIGHT_PACK: tl.constexpr = 1 if BF16_WEIGHTS else 2
    offs_bk = tl.arange(0, BLOCK_SIZE_K // WEIGHT_PACK)
    a_ptrs = (
        a_ptr
        + (offs_token[:, None] // top_k) * stride_am
        + offs_ak[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert * stride_be
        + offs_bk[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    if not BF16_WEIGHTS:
        offs_sk = tl.arange(0, BLOCK_SIZE_K // 32)
        s_ptrs = (
            b_mx_scale_ptr
            + expert * stride_bmxe
            + offs_bn[:, None] * stride_bmxn
            + offs_sk[None, :] * stride_bmxk
        )
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_ak[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_bk[:, None]
            < K // WEIGHT_PACK - k * (BLOCK_SIZE_K // WEIGHT_PACK),
            other=0,
        )
        if BF16_WEIGHTS:
            acc = tl.dot(a, b, acc=acc)
        else:
            scales = tl.load(
                s_ptrs,
                mask=offs_sk[None, :] < K // 32 - k * (BLOCK_SIZE_K // 32),
                other=0,
            )
            acc = tl.dot_scaled(
                a, None, A_DTYPE_FORMAT, b, scales, "e2m1", acc=acc, fast_math=True
            )
            s_ptrs += (BLOCK_SIZE_K // 32) * stride_bmxk
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // WEIGHT_PACK) * stride_bk

    # Multiply the FP32 down accumulator by the router weight BEFORE BF16
    # rounding. A BF16 down-output staging step here would change the contract.
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc *= weight[:, None]
    # Preserve the BF16 GEMM1 rounding before the activation (as in AITER).
    acc = acc.to(c_ptr.dtype.element_ty)
    if FUSED_GATE:
        gate, up = acc.to(tl.float32).reshape(BLOCK_SIZE_M, OUTPUT_BLOCK_N, 2).split()
        if ACTIVATION == "situ":
            acc = _situ_and_mul(gate, up, SITU_BETA, SITU_LINEAR_BETA)
        else:
            acc = gate * _sigmoid_exp2(gate) * up
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def fused_moe_mxfp4_act(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_mx_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor | None,
    num_tokens_post_padded: torch.Tensor | None,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> None:
    """Run packed or staged-BF16 gate/up + activation, or plain down GEMM."""
    if activation not in ("none", "silu", "situ"):
        raise NotImplementedError(f"Unsupported gfx942 MXFP4 activation: {activation}")
    if A.dtype != torch.bfloat16 or C.dtype != torch.bfloat16:
        raise NotImplementedError("gfx942 MXFP4 Triton MoE requires BF16 activations")
    bf16_weights = B.dtype == torch.bfloat16
    assert (bf16_weights or B.dtype == torch.uint8) and B_mx_scale.dtype == torch.uint8
    assert B.shape[2] * (1 if bf16_weights else 2) == A.shape[1]
    assert A.shape[1] % 32 == 0
    assert tuple(B_mx_scale.shape) == (*B.shape[:2], A.shape[1] // 32)
    direct = sorted_token_ids is None
    assert topk_weights.is_contiguous()
    if direct:
        assert topk_ids.shape[0] == 1 and topk_ids.is_contiguous()
        assert expert_ids is None and num_tokens_post_padded is None
    else:
        assert sorted_token_ids.stride(0) == 1
    assert C.ndim == 2
    assert C.shape[1] == B.shape[1] // (2 if activation != "none" else 1)
    if activation == "situ" and (situ_beta <= 0 or situ_linear_beta <= 0):
        raise ValueError("SiTU beta and linear_beta must be positive")

    if direct:
        em = topk_ids.numel() * config["BLOCK_SIZE_M"]
        sorted_token_ids = num_tokens_post_padded = topk_ids
        expert_ids = topk_ids
    else:
        em = sorted_token_ids.numel()
        if A.shape[0] < config["BLOCK_SIZE_M"]:
            em = min(em, A.shape[0] * top_k * config["BLOCK_SIZE_M"])
    grid = (
        triton.cdiv(em, config["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], config["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4_act[grid](
        A,
        B,
        C,
        B_mx_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        B_mx_scale.stride(0),
        B_mx_scale.stride(2),
        B_mx_scale.stride(1),
        A_DTYPE_FORMAT="bf16",
        BF16_WEIGHTS=bf16_weights,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        DIRECT=direct,
        ACTIVATION=activation,
        SITU_BETA=situ_beta,
        SITU_LINEAR_BETA=situ_linear_beta,
        **config,
    )
