# Copyright 2025 XunhaoLai. All rights reserved.

"""Owner-local sparse attention; logical positions and block IDs stay global."""

import torch
import triton
import triton.language as tl

from .utils import check_sparse_kv_fp8, sparse_out_dtype, unit_scale


@triton.jit
def _owner_attention_kernel(
    Q,
    K,
    V,
    Sink,
    Map,
    Slots,
    Lens,
    Cu,
    Prefix,
    CuBlocks,
    Topk,
    Score,
    Out,
    Lse,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    TS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    MAX_LEN: tl.constexpr,
    MAX_SLOTS: tl.constexpr,
    MAX_REQS: tl.constexpr,
    BLOCKS: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    K_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    PREFILL: tl.constexpr,
    INDEX: tl.constexpr,
    VALUE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    FP8: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    CHUNKS: tl.constexpr,
    BQ: tl.constexpr,
    BH: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
):
    tile_chunk, head_group, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    chunk = tile_chunk % CHUNKS
    tile = tile_chunk // CHUNKS
    if PREFILL:
        start = tl.load(Cu + batch)
        qlen = tl.load(Cu + batch + 1) - start
        prefix = tl.load(Prefix + batch)
    else:
        start = batch
        qlen = 1
        prefix = tl.load(Lens + batch) - 1
    if tile * BQ >= qlen:
        return
    seq_len = tl.minimum(tl.load(Lens + batch), MAX_LEN)
    sid = tl.load(Slots + batch).to(tl.int64)
    request_valid = (sid >= 0) & (sid < MAX_REQS)
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, BD)
    rv = tl.arange(0, BV)
    rg = tl.arange(0, BK)
    if INDEX and PREFILL:
        head = tl.full((BM,), head_group, tl.int32)
        qi = tile * BQ + rm
        kh = head_group // GROUP
        row_valid = (rm < BQ) & (qi < qlen)
    else:
        head = head_group * GROUP + rm % BH
        qi = tile * BQ + rm // BH
        kh = head_group
        row_valid = (rm % BH < GROUP) & (rm // BH < BQ) & (qi < qlen)
    row = start + qi
    qpos = prefix + qi
    q = tl.load(
        Q + row[:, None] * QS[0] + head[:, None] * QS[1] + rd[None, :] * QS[2],
        mask=row_valid[:, None] & (rd[None, :] < KD),
        other=0.0,
    )
    m = tl.full((BM,), float("-inf"), tl.float32)
    z = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BV), tl.float32)
    # The sink has zero value and exists once in the global distribution.
    if VALUE and HAS_SINK and DCP_RANK == 0:
        if chunk == 0:
            sink = tl.load(
                Sink + head[:, None] * SS[0] + rd[None, :] * SS[1],
                mask=row_valid[:, None] & (rd[None, :] < KD),
                other=0.0,
            )
            m = tl.sum(q.to(tl.float32) * sink.to(tl.float32), 1) * SCALE
            z = tl.full((BM,), 1.0, tl.float32)
    if INDEX:
        count = tl.minimum(tl.cdiv(seq_len, BK), BLOCKS)
        # Future blocks remain -inf in the initialized fixed-shape score buffer.
        count = tl.minimum(
            count, tl.cdiv(prefix + tl.minimum((tile + 1) * BQ, qlen), BK)
        )
        topk_row = 0
    else:
        count = TOPK
        if PREFILL:
            topk_row = tl.load(CuBlocks + batch) + tile
        else:
            topk_row = batch
    chunk_size = tl.cdiv(count, CHUNKS)
    begin = chunk * chunk_size
    end = tl.minimum(begin + chunk_size, count)
    for selected in range(begin, end):
        if INDEX:
            block = selected
        else:
            block = tl.load(Topk + kh * TS[0] + topk_row * TS[1] + selected * TS[2])
        block_valid = (block >= 0) & (block * BK < seq_len) & request_valid
        pos = block * BK + rg
        virtual = tl.load(
            Map + sid * MAP_STRIDE + pos, mask=block_valid & (pos < seq_len), other=-1
        ).to(tl.int64)
        owned = (
            block_valid
            & (pos < seq_len)
            & (virtual >= 0)
            & (virtual % DCP_SIZE == DCP_RANK)
        )
        owned = owned & (virtual // DCP_SIZE < MAX_SLOTS)
        # Compact before either matrix multiply. Under cyclic allocation these
        # lanes have stride DCP_SIZE, but ownership comes from the live map,
        # not an assumed global-position residue. Sorting also handles arbitrary
        # virtual-location permutations and unusually unbalanced blocks exactly.
        order = tl.sort(tl.where(owned, rg, BK), descending=False)
        n_owned = tl.sum(owned.to(tl.int32), 0)
        block_m = tl.full((BM,), float("-inf"), tl.float32)
        block_z = tl.zeros((BM,), tl.float32)
        for local_start in range(0, n_owned, BN):
            lane = local_start + rn
            offsets = tl.gather(order, tl.minimum(lane, BK - 1), axis=0)
            local_pos = block * BK + offsets
            local_valid = (lane < n_owned) & (offsets < BK)
            loc = (
                tl.load(
                    Map + sid * MAP_STRIDE + local_pos, mask=local_valid, other=0
                ).to(tl.int64)
                // DCP_SIZE
            )
            k = tl.load(
                K + loc[None, :] * KS[0] + kh * KS[1] + rd[:, None] * KS[2],
                mask=local_valid[None, :] & (rd[:, None] < KD),
                other=0.0,
            )
            if FP8:
                k = k.to(q.dtype)
            logits = tl.dot(q, k) * (SCALE * K_SCALE)
            logits = tl.where(
                row_valid[:, None]
                & local_valid[None, :]
                & (qpos[:, None] >= local_pos[None, :]),
                logits,
                float("-inf"),
            )
            tile_m = tl.max(logits, 1)
            if INDEX:
                new_block_m = tl.maximum(block_m, tile_m)
                safe_block_m = tl.where(new_block_m == float("-inf"), 0.0, new_block_m)
                block_z = block_z * tl.exp2(block_m - safe_block_m) + tl.sum(
                    tl.exp2(logits - safe_block_m[:, None]), 1
                )
                block_m = new_block_m
            if VALUE:
                new_m = tl.maximum(m, tile_m)
                safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
                p = tl.exp2(logits - safe_m[:, None])
                alpha = tl.exp2(m - safe_m)
                z = z * alpha + tl.sum(p, 1)
                v = tl.load(
                    V + loc[:, None] * VS[0] + kh * VS[1] + rv[None, :] * VS[2],
                    mask=local_valid[:, None] & (rv[None, :] < VD),
                    other=0.0,
                )
                if FP8:
                    v = v.to(q.dtype)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v) * V_SCALE
                m = new_m
        if INDEX:
            if SCORE_TYPE == "max":
                score = block_m
            else:
                score = tl.where(block_z > 0, block_m + tl.log2(block_z), float("-inf"))
            tl.store(
                Score + (head * ROWS + row) * BLOCKS + block,
                score,
                mask=row_valid & (block >= 0) & (block < BLOCKS),
            )
    if VALUE:
        normalized = acc / tl.where(z > 0, z, 1.0)[:, None]
        natural_lse = tl.where(
            z > 0, (m + tl.log2(z)) * 0.6931471805599453, float("-inf")
        )
        output_row = (chunk * ROWS + row) * HEADS + head
        tl.store(
            Out + output_row[:, None] * VD + rv[None, :],
            normalized,
            mask=row_valid[:, None] & (rv[None, :] < VD),
        )
        tl.store(Lse + output_row, natural_lse, mask=row_valid)


@triton.jit
def _merge_owner_chunks(
    Out,
    Lse,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    VD: tl.constexpr,
    CHUNKS: tl.constexpr,
    BV: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, CHUNKS)
    d = tl.arange(0, BV)
    l = tl.load(Lse + c * ROWS * HEADS + row)
    m = tl.max(l, 0)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(l - safe_m)
    z = tl.sum(w, 0)
    w = w / tl.where(z > 0, z, 1.0)
    o = tl.load(
        Out + (c[:, None] * ROWS * HEADS + row) * VD + d[None, :],
        mask=d[None, :] < VD,
        other=0,
    )
    tl.store(Out + row * VD + d, tl.sum(o * w[:, None], 0), mask=d < VD)
    tl.store(Lse + row, tl.where(z > 0, safe_m + tl.log(z), float("-inf")))


@triton.jit
def _decode_score_priorities(
    Score,
    Lens,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCKS: tl.constexpr,
    BK: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    TILE: tl.constexpr,
):
    row_head = tl.program_id(0)
    b = row_head % ROWS
    block = tl.program_id(1) * TILE + tl.arange(0, TILE)
    count = tl.cdiv(tl.load(Lens + b), BK)
    valid = (block < count) & (block < BLOCKS)
    ptr = Score + row_head * BLOCKS + block
    score = tl.load(ptr, mask=block < BLOCKS, other=float("-inf"))
    score = tl.where(valid & (block < INIT), 1e30, score)
    score = tl.where(valid & (block >= count - LOCAL), 1e29, score)
    tl.store(ptr, score, mask=block < BLOCKS)


def owner_attention(
    q,
    k_cache,
    v_cache,
    sink,
    req_to_token,
    slot_ids,
    seq_lens,
    block_size,
    sm_scale,
    q_scale,
    k_scale,
    v_scale,
    *,
    dcp_size,
    dcp_rank,
    return_lse,
    max_seqlen_q=1,
    max_seqlen_k=None,
    cu_seqlens=None,
    prefix_lens=None,
    cu_seqblocks_q=None,
    block_size_q=1,
    topk_idx=None,
    score_type="max",
    disable_index_value=False,
):
    """Return (output, natural LSE, raw base-2 block scores).

    Score layout is contiguous [query_heads, query_rows, global_blocks].
    Index-value-disabled calls return None for both output and LSE.
    """
    if dcp_size < 1 or not 0 <= dcp_rank < dcp_size:
        raise ValueError("invalid DCP size/rank")
    assert block_size == triton.next_power_of_2(block_size)
    assert q.shape[1] % k_cache.shape[1] == 0
    is_fp8 = check_sparse_kv_fp8(
        q, k_cache, None if disable_index_value else v_cache, label="owner-local"
    )
    prefill = cu_seqlens is not None
    index = topk_idx is None
    value = not disable_index_value
    rows, heads, kd = q.shape
    vd = v_cache.shape[-1] if value else kd
    group = heads // k_cache.shape[1]
    batches = seq_lens.shape[0]
    bq = 16 if index and prefill else block_size_q if prefill else 1
    tiled_index = (
        index
        and prefill
        and not value
        and torch.version.hip
        and q.dtype == torch.bfloat16
        and kd == 128
        and max_seqlen_q >= 128
    )
    if tiled_index:
        bq = 128
    bh = 1 if index and prefill else triton.next_power_of_2(group)
    head_groups = heads if index and prefill else k_cache.shape[1]
    bm = max(16, bq * bh)
    # BF16/FP16 MFMA needs 16 K lanes for PV. FP8 requires 32.
    bn = max(
        32 if q.dtype == torch.float8_e4m3fn else 16,
        triton.next_power_of_2(triton.cdiv(block_size, dcp_size)),
    )
    if max_seqlen_k is None:
        max_seqlen_k = req_to_token.shape[1]
    blocks = triton.cdiv(min(max_seqlen_k, req_to_token.shape[1]), block_size)
    topk = topk_idx.shape[-1] if not index else 0
    target = max(1, min(256, 256 // max(1, batches * head_groups)))
    chunks = 1 if prefill else 1 << (target.bit_length() - 1)
    if tiled_index:
        # Larger query tiles reuse index K; splitting disjoint score columns
        # supplies enough CTAs without any additional softmax merge.
        query_tiles = triton.cdiv(max_seqlen_q, bq) * head_groups * batches
        chunks = max(1, min(8, blocks, triton.cdiv(512, query_tiles)))
    score = (
        torch.full(
            (heads, rows, blocks), float("-inf"), device=q.device, dtype=torch.float32
        )
        if index
        else None
    )
    out = (
        torch.empty((chunks, rows, heads, vd), device=q.device, dtype=torch.float32)
        if value
        else None
    )
    lse = (
        torch.empty((chunks, rows, heads), device=q.device, dtype=torch.float32)
        if value
        else None
    )
    scale = (
        (kd**-0.5 if sm_scale is None else sm_scale)
        * unit_scale(q_scale)
        * 1.4426950408889634
    )
    grid = (triton.cdiv(max_seqlen_q, bq) * chunks, head_groups, batches)
    _owner_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        req_to_token,
        slot_ids,
        seq_lens,
        cu_seqlens,
        prefix_lens,
        cu_seqblocks_q,
        topk_idx,
        score,
        out,
        lse,
        q.stride(),
        k_cache.stride(),
        v_cache.stride() if value else (0, 0, 0),
        sink.stride() if sink is not None else (0, 0),
        topk_idx.stride() if not index else (0, 0, 0),
        req_to_token.stride(0),
        rows,
        heads,
        group,
        kd,
        vd,
        req_to_token.shape[1],
        k_cache.shape[0],
        req_to_token.shape[0],
        blocks,
        topk,
        scale,
        unit_scale(k_scale),
        unit_scale(v_scale),
        dcp_size,
        dcp_rank,
        prefill,
        index,
        value,
        sink is not None,
        is_fp8,
        score_type,
        chunks,
        bq,
        bh,
        bm,
        block_size,
        bn,
        triton.next_power_of_2(kd),
        triton.next_power_of_2(vd),
        num_warps=4,
    )
    if value:
        if chunks > 1:
            _merge_owner_chunks[(rows * heads,)](
                out,
                lse,
                rows,
                heads,
                vd,
                chunks,
                triton.next_power_of_2(vd),
                num_warps=4,
            )
        out, lse = out[0], lse[0]
        if not return_lse:
            out = out.to(sparse_out_dtype(q))
    return out, lse, score


def reduce_decode_scores(
    score, seq_lens, block_size, init_blocks, local_blocks, score_type, score_reduce
):
    # Raw scores must be globally combined before forced priorities are applied.
    if score_reduce is not None:
        score = score_reduce(score, score_type)
    _decode_score_priorities[
        (score.shape[0] * score.shape[1], triton.cdiv(score.shape[2], 256))
    ](
        score,
        seq_lens,
        score.shape[0],
        score.shape[1],
        score.shape[2],
        block_size,
        init_blocks,
        local_blocks,
        256,
    )
    return score


def live_query_blocks(cu_seqlens, block_size_q):
    """Device-only metadata calculation, deliberately not identity-cached."""
    cu_blocks = torch.empty_like(cu_seqlens)
    cu_blocks[0] = 0
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    torch.cumsum(
        (lengths + block_size_q - 1) // block_size_q,
        0,
        dtype=cu_seqlens.dtype,
        out=cu_blocks[1:],
    )
    return cu_blocks


@triton.jit
def _pack_topk_candidates(
    Scores,
    Ids,
    Packed,
    Cu,
    CuBlocks,
    Prefix,
    SCORE_STRIDES: tl.constexpr,
    ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    BT: tl.constexpr,
):
    query_block, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row_start = tl.load(CuBlocks + batch)
    count = tl.load(CuBlocks + batch + 1) - row_start
    if query_block >= count:
        return
    row = row_start + query_block
    query = tl.load(Cu + batch) + query_block * BLOCK_Q
    valid_blocks = (
        tl.load(Prefix + batch) + query_block * BLOCK_Q + BLOCK_K
    ) // BLOCK_K
    lanes = tl.arange(0, BT)
    offsets = (head * ROWS + row) * TOPK + lanes
    ids = tl.load(Ids + offsets, lanes < TOPK, -1)
    scores = tl.load(
        Scores
        + head * SCORE_STRIDES[0]
        + query * SCORE_STRIDES[1]
        + ids * SCORE_STRIDES[2],
        (lanes < TOPK) & (ids >= 0),
        float("-inf"),
    )
    scores = tl.where(scores != scores, -1e30, scores)
    scores = tl.where((ids >= 0) & (ids < INIT), 1e30, scores)
    scores = tl.where(
        (ids >= 0) & (ids >= tl.maximum(0, valid_blocks - LOCAL)), 1e29, scores
    )
    tl.store(Packed + offsets * 2, ids, lanes < TOPK)
    tl.store(Packed + offsets * 2 + 1, scores.to(tl.int32, bitcast=True), lanes < TOPK)


@triton.jit
def _merge_topk_candidates(
    Packed,
    Ids,
    CuBlocks,
    BATCHES: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    WORLD: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    if row >= tl.load(CuBlocks + BATCHES):
        return
    lanes = tl.arange(0, BN)
    offsets = (((lanes // TOPK) * HEADS + head) * ROWS + row) * TOPK + lanes % TOPK
    ids = tl.load(Packed + offsets * 2, lanes < WORLD * TOPK, -1).to(tl.uint32)
    bits = tl.load(Packed + offsets * 2 + 1, lanes < WORLD * TOPK, 0).to(tl.uint32)
    # IEEE float bits become an unsigned monotonic score key, including negatives.
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    by_id = tl.sort((ids.to(tl.uint64) << 32) | ordered.to(tl.uint64), descending=False)
    ids = (by_id >> 32).to(tl.uint32)
    next_ids = (tl.gather(by_id, tl.minimum(lanes + 1, BN - 1), 0) >> 32).to(tl.uint32)
    # The last occurrence of each block has its maximum score across owners.
    last = (lanes == BN - 1) | (ids != next_ids)
    ranked = tl.where(
        last & (ids != 0xFFFFFFFF),
        ((by_id & 0xFFFFFFFF) << 32) | (0xFFFFFFFF - ids).to(tl.uint64),
        0,
    )
    ranked = tl.sort(ranked, descending=True)
    out_lanes = tl.arange(0, BT)
    best = tl.gather(ranked, out_lanes, 0)
    best_ids = (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32)
    best_ids = tl.sort(
        tl.where((out_lanes < TOPK) & (best_ids >= 0), best_ids, 0x7FFFFFFF),
        descending=False,
    )
    tl.store(
        Ids + (head * ROWS + row) * TOPK + out_lanes,
        tl.where(best_ids == 0x7FFFFFFF, -1, best_ids),
        out_lanes < TOPK,
    )


def pack_topk_candidates(
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
):
    """Pack lossless int32 block ids and FP32 scores into one collective payload."""
    heads, rows, topk = topk_idx.shape
    packed = torch.empty(
        (*topk_idx.shape, 2), dtype=torch.int32, device=topk_idx.device
    )
    _pack_topk_candidates[(max_seqblock_q, cu_seqlens.numel() - 1, heads)](
        scores,
        topk_idx,
        packed,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        scores.stride(),
        rows,
        topk,
        block_size_q,
        block_size_k,
        init_blocks,
        local_blocks,
        triton.next_power_of_2(topk),
    )
    return packed


def merge_topk_candidates(packed, topk_idx, cu_seqblocks_q):
    """Exact MAX-score global top-k from owner-local top-k candidates."""
    heads, rows, topk = topk_idx.shape
    world = packed.shape[0] // heads
    _merge_topk_candidates[(rows, heads)](
        packed,
        topk_idx,
        cu_seqblocks_q,
        cu_seqblocks_q.numel() - 1,
        heads,
        rows,
        topk,
        world,
        triton.next_power_of_2(world * topk),
        triton.next_power_of_2(topk),
    )
    return topk_idx
