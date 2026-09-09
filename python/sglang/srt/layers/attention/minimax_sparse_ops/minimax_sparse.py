# Copyright 2025 XunhaoLai. All rights reserved.

import logging
from typing import Callable, List, Optional, Tuple

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.dcp.comm import (
    cp_lse_ag_out_rs_mha,
    dcp_a2a_lse_reduce,
)
from sglang.kernels.ops.attention.minimax_sparse.common.index import topk_index_reduce
from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)

logger = logging.getLogger(__name__)
_msa_fallback_warned = False


def _warn_msa_fallback(err: Exception) -> None:
    global _msa_fallback_warned
    if _msa_fallback_warned:
        return
    logger.warning(
        "MiniMax MSA backend is unavailable (%s); falling back to Triton sparse attention.",
        err,
    )
    _msa_fallback_warned = True


def _all_reduce_block_scores(scores, op, group: GroupCoordinator) -> None:
    # Match GroupCoordinator's eager/graph transport choice, without a
    # quantized custom all-reduce: block selection requires exact MAX/SUM.
    communicator = group.pynccl_comm
    if communicator is not None and not communicator.disabled:
        communicator.all_reduce(scores, op=op)
    else:
        torch.distributed.all_reduce(scores, op=op, group=group.device_group)


def reduce_dcp_block_scores(
    scores: torch.Tensor, score_type: str, *, group: GroupCoordinator
) -> torch.Tensor:
    """Combine owner-local block scores before top-k, in base-2 units."""
    if score_type == "max":
        _all_reduce_block_scores(scores, torch.distributed.ReduceOp.MAX, group)
        return scores
    maximum = scores.clone()
    _all_reduce_block_scores(maximum, torch.distributed.ReduceOp.MAX, group)
    scores.sub_(maximum).masked_fill_(maximum == -float("inf"), -float("inf"))
    scores.exp2_()
    _all_reduce_block_scores(scores, torch.distributed.ReduceOp.SUM, group)
    return scores.log2_().add_(maximum)


def _merge_dcp_output(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: GroupCoordinator,
    comm_backend: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if comm_backend in ("a2a", "fi_a2a"):
        output = dcp_a2a_lse_reduce(
            output, lse, group, is_lse_base_on_e=True, comm_backend=comm_backend
        )
    else:
        output = cp_lse_ag_out_rs_mha(output, lse, group)
    return output.to(dtype)


def reduce_dcp_topk(
    topk_idx,
    scores,
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    block_size_q,
    block_size_k,
    init_blocks,
    local_blocks,
    max_seqblock_q,
    *,
    group: GroupCoordinator,
):
    # For MAX scoring, every globally winning block is a local top-k candidate
    # on an owner attaining its score. Exchange candidates, not the dense matrix.
    from sglang.kernels.ops.attention.minimax_sparse.common.dcp import (
        merge_topk_candidates,
        pack_topk_candidates,
    )

    packed = pack_topk_candidates(
        scores,
        topk_idx,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        block_size_q,
        block_size_k,
        init_blocks,
        local_blocks,
        max_seqblock_q,
    )
    packed = group.all_gather(packed, dim=0)
    return merge_topk_candidates(packed, topk_idx, cu_seqblocks_q)


def _gather_dcp_index_query(
    query: torch.Tensor,
    group: GroupCoordinator,
    replica_size: int,
    disable_value: bool,
) -> torch.Tensor:
    if disable_value and replica_size >= group.world_size:
        # TP already replicates this index head across the entire DCP group.
        return query
    query = group.all_gather(query.contiguous(), dim=1)
    if disable_value and replica_size > 1:
        # Duplicate heads select identical blocks. Keep one per replica group;
        # the index-value path retains every head for its output reduce-scatter.
        query = query[:, ::replica_size]
    return query


def minimax_sparse_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged index)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged index); None when disable_index_value
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    cu_seqlens: torch.Tensor,  # [batch_size + 1, ] (Q-side cumulative)
    seq_lens: torch.Tensor,  # [batch_size, ] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_msa: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_q_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
    dcp_group: Optional[GroupCoordinator] = None,
    dcp_comm_backend: str = "ag_rs",
    score_reduce: Optional[Callable] = None,
    idx_q_replica_size: int = 1,
    topk_reduce: Optional[Callable] = None,
):
    """Run MiniMax-M3 sparse prefill.

    ``cu_seqblocks_q``, ``max_seqblock_q``, and ``all_seqblock_q`` are optional
    precomputed query-block metadata shared by the index and value sparse
    kernels. Supplying them avoids recomputing the same block layout twice.
    ``seqlens_cpu`` (host copy of ``torch.diff(cu_seqlens)``) is forwarded to
    ``get_cu_seqblocks`` to avoid a per-layer device sync when it recomputes.
    """
    dcp_size = 1 if dcp_group is None else dcp_group.world_size
    dcp_rank = 0 if dcp_group is None else dcp_group.rank_in_group
    if dcp_size > 1:
        assert not use_msa
        q = dcp_group.all_gather(q.contiguous(), dim=1).contiguous()
        idx_q = _gather_dcp_index_query(
            idx_q, dcp_group, idx_q_replica_size, disable_index_value
        )
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
        )

    # All seqlen is less than topk, use full attention
    # Step 1: Flash attention with topk index (using index head)
    index_result = flash_prefill_with_topk_index(
        q=idx_q,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        sink=idx_sink,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
        all_seqblock_q=all_seqblock_q,
        q_scale=idx_q_scale,
        k_scale=idx_k_scale,
        v_scale=idx_v_scale,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        return_lse=dcp_size > 1,
        score_reduce=score_reduce,
        topk_reduce=topk_reduce,
    )
    if dcp_size > 1:
        idx_o, topk_idx, idx_lse = index_result
    else:
        idx_o, topk_idx = index_result
    # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )
    # Step 3: Sparse attention using topk index (main head). The MSA path only
    # replaces this step; the indexer above is unchanged. MSA has no attn-sink
    # input, so keep the Triton path when sink is present.
    if use_msa and sink is None:
        from .msa import MSAUnavailableError, msa_sparse_prefill_main

        try:
            o = msa_sparse_prefill_main(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                topk_idx=topk_idx,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_lens,
                block_size_k=block_size_k,
                sm_scale=sm_scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        except MSAUnavailableError as err:
            _warn_msa_fallback(err)
            o = flash_prefill_with_gqa_share_sparse(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                sink=sink,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                topk_idx=topk_idx,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_lens,
                max_seqlen_q=max_seqlen_q,
                sm_scale=sm_scale,
                cu_seqblocks_q=cu_seqblocks_q,
                max_seqblock_q=max_seqblock_q,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
    else:
        o = flash_prefill_with_gqa_share_sparse(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            sink=sink,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            topk_idx=topk_idx,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            sm_scale=sm_scale,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            return_lse=dcp_size > 1,
        )
    if dcp_size > 1:
        o, lse = o
        o = _merge_dcp_output(o, lse, dcp_group, dcp_comm_backend, q.dtype)
        if idx_o is not None:
            idx_o = _merge_dcp_output(
                idx_o, idx_lse, dcp_group, dcp_comm_backend, idx_q.dtype
            )
    return idx_o, o


def minimax_sparse_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, qk_head_dim]
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    idx_q: torch.Tensor,  # [batch_size, num_idx_heads, idx_head_dim], num_idx_heads >= num_kv_heads
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged); None when disable_index_value
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    seq_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen: int,  # max of seq_lens, passed from caller to avoid sync during CUDA graph capture
    block_size_q: int,  # useless for now, will always be 1
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    dense_main_attn_fn: Optional[Callable] = None,
    page_size: int = 1,
    use_msa: bool = False,
    msa_kv_indices: Optional[
        torch.Tensor
    ] = None,  # per-forward MSA page table (cached)
    msa_plan=None,  # per-forward MSA fmha_sm100 plan (cached)
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_q_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
    dcp_group: Optional[GroupCoordinator] = None,
    dcp_comm_backend: str = "ag_rs",
    score_reduce: Optional[Callable] = None,
    idx_q_replica_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dcp_size = 1 if dcp_group is None else dcp_group.world_size
    dcp_rank = 0 if dcp_group is None else dcp_group.rank_in_group
    if dcp_size > 1:
        assert not use_msa and dense_main_attn_fn is None
        q = dcp_group.all_gather(q.contiguous(), dim=1).contiguous()
        idx_q = _gather_dcp_index_query(
            idx_q, dcp_group, idx_q_replica_size, disable_index_value
        )
    # Step 1: Flash decode with topk index (using index head). When the dense main
    # attention is used, the indexer emits the page table directly (fused
    # transform) instead of block ids, plus the per-query effective KV length.
    index_result = flash_decode_with_topk_idx(
        q=idx_q,
        sink=idx_sink,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        slot_ids=slot_ids,
        block_size=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        use_dense_main_attn=dense_main_attn_fn is not None,
        page_size=page_size,
        q_scale=idx_q_scale,
        k_scale=idx_k_scale,
        v_scale=idx_v_scale,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        return_lse=dcp_size > 1,
        score_reduce=score_reduce,
    )
    if dcp_size > 1:
        idx_o, topk_idx, real_seq_lens, idx_lse = index_result
    else:
        idx_o, topk_idx, real_seq_lens = index_result
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if dense_main_attn_fn is not None:
        # topk_idx is the page table; real_seq_lens is the per-query cache_seqlens
        assert idx_group_size == 1
        o = dense_main_attn_fn(q, topk_idx, real_seq_lens)
    else:
        # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
        if idx_group_size > 1:
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
            )
        # Step 3: Sparse attention using topk index (main head). The MSA path
        # only replaces this step; keep the Triton path when sink is present.
        if use_msa and sink is None:
            from .msa import MSAUnavailableError, msa_sparse_decode_main

            try:
                o = msa_sparse_decode_main(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    topk_idx=topk_idx,
                    req_to_token=req_to_token,
                    slot_ids=slot_ids,
                    seq_lens=seq_lens,
                    block_size_k=block_size_k,
                    sm_scale=sm_scale,
                    kv_indices=msa_kv_indices,
                    plan=msa_plan,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
            except MSAUnavailableError as err:
                _warn_msa_fallback(err)
                o = flash_decode_with_gqa_share_sparse(
                    q=q,
                    sink=sink,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    req_to_token=req_to_token,
                    seq_lens=seq_lens,
                    slot_ids=slot_ids,
                    block_size=block_size_k,
                    topk_idx=topk_idx,
                    sm_scale=sm_scale,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
        else:
            o = flash_decode_with_gqa_share_sparse(
                q=q,
                sink=sink,
                k_cache=k_cache,
                v_cache=v_cache,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                slot_ids=slot_ids,
                block_size=block_size_k,
                topk_idx=topk_idx,
                sm_scale=sm_scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                return_lse=dcp_size > 1,
            )
    if dcp_size > 1:
        o, lse = o
        o = _merge_dcp_output(o, lse, dcp_group, dcp_comm_backend, q.dtype)
        if idx_o is not None:
            idx_o = _merge_dcp_output(
                idx_o, idx_lse, dcp_group, dcp_comm_backend, idx_q.dtype
            )
    return idx_o, o
