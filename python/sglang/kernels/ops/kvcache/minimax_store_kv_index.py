from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(head_bytes: int) -> Module:
    # Build marker is (head_bytes, kUsePDL); the index dtype (int32/int64) is a
    # runtime dispatch inside the C++ launcher.
    args = make_cpp_args(head_bytes, is_arch_support_pdl())
    return load_jit(
        "minimax_store_kv_index",
        *args,
        cuda_files=["minimax/fused_store_kv_index.cuh"],
        cuda_wrappers=[("store_kv_index", f"store_kv_index<{args}>")],
    )


def store_kv_index(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: Optional[torch.Tensor],
    idx_v_cache: Optional[torch.Tensor],
    indices: torch.Tensor,
    *,
    num_kv_heads: int,
    head_bytes: int,
) -> None:
    """Fused store of the MiniMax-M3 sparse caches in one launch.

    Writes the main ``k``/``v`` (``num_kv_heads`` heads each), the single index
    ``idx_k`` head, and optionally the single ``idx_v`` head into their caches
    at the per-token rows given by ``indices`` (out_cache_loc). In-place on the
    four cache tensors.

    All tensors must share the same (store) dtype and a head_dim whose byte size
    equals ``head_bytes`` (a multiple of 16). ``k``/``idx_k`` are 2D rows
    ``[T, num_kv_heads*head_dim]`` / ``[T, head_dim]``; caches are the matching
    ``[num_pages, ...]`` buffers. When ``idx_v`` is None there is no index value
    head (the layer is a pure block selector).
    """
    has_v = idx_v is not None
    if not has_v:
        # Pass idx_k as a dummy for the unused index-V slot; heads_per_token is
        # set so the kernel never reaches the index-V branch.
        idx_v = idx_k
        idx_v_cache = idx_k_cache
    heads_per_token = 2 * num_kv_heads + 1 + (1 if has_v else 0)

    module = _jit_module(head_bytes)
    module.store_kv_index(
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        idx_v,
        idx_v_cache,
        indices,
        num_kv_heads,
        heads_per_token,
    )


@triton.jit
def _store_dcp_cache_row(
    Src,
    Cache,
    Scale,
    token,
    row,
    owner,
    SRC_STRIDES: tl.constexpr,
    CACHE_STRIDES: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    head = offsets // HEAD_DIM
    dim = offsets % HEAD_DIM
    mask = owner & (offsets < HEADS * HEAD_DIM)
    values = tl.load(
        Src + token * SRC_STRIDES[0] + head * SRC_STRIDES[1] + dim * SRC_STRIDES[2],
        mask=mask,
        # Integer zero cannot be converted to FNUZ by Triton 3.6.
        other=0.0,
    )
    if Scale is not None:
        if Cache.dtype.element_ty.is_fp8() and not Src.dtype.element_ty.is_fp8():
            if isinstance(Scale, tl.tensor) and Scale.dtype.is_ptr():
                scale = tl.load(Scale, mask=owner, other=1.0)
            else:
                scale = Scale
            # Match the pool's in-place division: round to the source dtype
            # before converting to FP8, including for non-power-of-two scales.
            values = (values.to(tl.float32) / scale).to(Src.dtype.element_ty)
    tl.store(
        Cache
        + row * CACHE_STRIDES[0]
        + head * CACHE_STRIDES[1]
        + dim * CACHE_STRIDES[2],
        values.to(Cache.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _store_kv_index_dcp_kernel(
    K,
    V,
    KCache,
    VCache,
    IdxK,
    IdxKCache,
    IdxV,
    IdxVCache,
    Indices,
    KScale,
    VScale,
    IdxKScale,
    IdxVScale,
    DcpMask,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    KC_STRIDES: tl.constexpr,
    VC_STRIDES: tl.constexpr,
    IK_STRIDES: tl.constexpr,
    IKC_STRIDES: tl.constexpr,
    IV_STRIDES: tl.constexpr,
    IVC_STRIDES: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    BLOCK_MAIN: tl.constexpr,
    BLOCK_INDEX: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    location = tl.load(Indices + token * INDEX_STRIDE).to(tl.int64)
    if DcpMask is not None:
        # Explicit masks accompany physical locations; never divide these again.
        owner = (location >= 0) & tl.load(DcpMask + token * MASK_STRIDE)
        row = location
    else:
        owner = (location >= 0) & (location % DCP_SIZE == DCP_RANK)
        row = location // DCP_SIZE

    _store_dcp_cache_row(
        K,
        KCache,
        KScale,
        token,
        row,
        owner,
        K_STRIDES,
        KC_STRIDES,
        HEADS,
        HEAD_DIM,
        BLOCK_MAIN,
    )
    _store_dcp_cache_row(
        V,
        VCache,
        VScale,
        token,
        row,
        owner,
        V_STRIDES,
        VC_STRIDES,
        HEADS,
        HEAD_DIM,
        BLOCK_MAIN,
    )
    _store_dcp_cache_row(
        IdxK,
        IdxKCache,
        IdxKScale,
        token,
        row,
        owner,
        IK_STRIDES,
        IKC_STRIDES,
        1,
        INDEX_HEAD_DIM,
        BLOCK_INDEX,
    )
    if IdxV is not None:
        _store_dcp_cache_row(
            IdxV,
            IdxVCache,
            IdxVScale,
            token,
            row,
            owner,
            IV_STRIDES,
            IVC_STRIDES,
            1,
            INDEX_HEAD_DIM,
            BLOCK_INDEX,
        )


def store_kv_index_dcp(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: Optional[torch.Tensor],
    idx_v_cache: Optional[torch.Tensor],
    indices: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    k_scale: Optional[float | torch.Tensor] = None,
    v_scale: Optional[float | torch.Tensor] = None,
    idx_k_scale: Optional[float | torch.Tensor] = None,
    idx_v_scale: Optional[float | torch.Tensor] = None,
    dcp_kv_mask: Optional[torch.Tensor] = None,
) -> None:
    """Store mixed-dtype main/index caches in one owner-masked Triton launch.

    Sources are typed ``[tokens, heads, head_dim]`` tensors and caches are typed
    ``[physical_rows, heads, head_dim]`` tensors, with arbitrary element strides.
    Main K/V shapes match; index K/V have one head and their own head dimension.
    FP16, BF16, and FP8 (E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ) may be mixed across
    sources and destinations. Do not pass uint8 storage views.

    Without ``dcp_kv_mask``, ``indices`` contains virtual locations: rank
    ``indices % dcp_size`` writes physical row ``indices // dcp_size``. With an
    explicit boolean mask, locations are already physical and are not divided.
    The same ownership decision gates every cache write. Negative locations
    are ignored; owned nonnegative locations must fit every supplied cache.

    Scales are Python scalars or one-element tensors on the sources' device.
    Division is applied only when converting a non-FP8 source into an FP8 cache;
    already-FP8 sources and non-FP8 destinations ignore scales. No input is
    mutated, compacted, or converted outside the kernel. K-only index layers
    pass both ``idx_v`` and ``idx_v_cache`` as None and allocate no index values.
    """
    assert dcp_size > 0 and 0 <= dcp_rank < dcp_size
    assert (idx_v is None) == (idx_v_cache is None)
    assert k.ndim == 3 and v.shape == k.shape
    assert k_cache.ndim == 3 and k_cache.shape[1:] == k.shape[1:]
    assert v_cache.ndim == 3 and v_cache.shape[1:] == v.shape[1:]
    assert idx_k.ndim == 3 and idx_k.shape[:2] == (k.shape[0], 1)
    assert idx_k_cache.ndim == 3 and idx_k_cache.shape[1:] == idx_k.shape[1:]
    if idx_v is not None:
        assert idx_v.shape == idx_k.shape
        assert idx_v_cache.ndim == 3
        assert idx_v_cache.shape[1:] == idx_v.shape[1:]
    assert indices.ndim == 1 and indices.shape[0] == k.shape[0]
    assert indices.dtype in (torch.int32, torch.int64)
    if dcp_kv_mask is not None:
        assert dcp_kv_mask.shape == indices.shape
        assert dcp_kv_mask.dtype == torch.bool

    tokens, heads, head_dim = k.shape
    index_head_dim = idx_k.shape[2]
    assert heads > 0 and head_dim > 0 and index_head_dim > 0
    if tokens == 0:
        return

    _store_kv_index_dcp_kernel[(tokens,)](
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        idx_v,
        idx_v_cache,
        indices,
        k_scale,
        v_scale,
        idx_k_scale,
        idx_v_scale,
        dcp_kv_mask,
        k.stride(),
        v.stride(),
        k_cache.stride(),
        v_cache.stride(),
        idx_k.stride(),
        idx_k_cache.stride(),
        idx_v.stride() if idx_v is not None else (0, 0, 0),
        idx_v_cache.stride() if idx_v_cache is not None else (0, 0, 0),
        indices.stride(0),
        dcp_kv_mask.stride(0) if dcp_kv_mask is not None else 0,
        heads,
        head_dim,
        index_head_dim,
        dcp_size,
        dcp_rank,
        triton.next_power_of_2(heads * head_dim),
        triton.next_power_of_2(index_head_dim),
        num_warps=4,
    )
