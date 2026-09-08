# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exact-rounding FP8 prefix probabilities for the gfx942 extend-attention path."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import mfma

_NUM_PARTITIONS = 32
_BLOCK_M = 64
_BLOCK_N = 64
_NUM_HEADS = 12


@gluon.jit
def _prefix_probability_kernel(
    Q,
    K_Buffer,
    QO_Indptr,
    KV_Indptr,
    KV_Indices,
    Partition_Max,
    Probabilities,
    Stats,
    k_scale,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    PREFIX_CAPACITY: gl.constexpr,
    PREFIX_Q_ROWS: gl.constexpr,
    NUM_PARTITIONS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    STORE_PROBS: gl.constexpr,
):
    seq = gl.program_id(0)
    head = gl.program_id(1)
    partition = gl.program_id(2)
    q_start = gl.load(QO_Indptr + seq)
    q_len = gl.load(QO_Indptr + seq + 1) - q_start
    kv_start = gl.load(KV_Indptr + seq)
    prefix_len = gl.load(KV_Indptr + seq + 1) - kv_start
    if (
        (prefix_len < 8192)
        | (prefix_len > PREFIX_CAPACITY)
        | (q_len <= 0)
        | (q_len > PREFIX_Q_ROWS)
    ):
        return

    # Fix the original MFMA layout and FP32 sum tree: another reduction tree
    # can cause rare BF16 output changes even with identical 64-token tiles.
    MMA: gl.constexpr = AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    QL: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    KL: gl.constexpr = gl.BlockedLayout([16, 1], [32, 2], [1, 4], [0, 1])
    qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QL))
    qd = gl.arange(0, 512, layout=gl.SliceLayout(0, QL))
    qpe_d = 512 + gl.arange(0, 64, layout=gl.SliceLayout(0, QL))
    q = gl.load(
        Q + (q_start + qm[:, None]) * stride_qbs + head * stride_qh + qd[None, :],
        mask=qm[:, None] < q_len,
        other=0.0,
    )
    qpe = gl.load(
        Q + (q_start + qm[:, None]) * stride_qbs + head * stride_qh + qpe_d[None, :],
        mask=qm[:, None] < q_len,
        other=0.0,
    )
    q = gl.convert_layout(
        q.to(K_Buffer.dtype.element_ty), gl.DotOperandLayout(0, MMA, 16)
    )
    qpe = gl.convert_layout(
        qpe.to(K_Buffer.dtype.element_ty), gl.DotOperandLayout(0, MMA, 16)
    )
    kd = gl.arange(0, 512, layout=gl.SliceLayout(1, KL))
    kdpe = 512 + gl.arange(0, 64, layout=gl.SliceLayout(1, KL))
    kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, KL))
    m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MMA))
    n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MMA))
    row = (seq.to(gl.int64) * 12 + head) * PREFIX_Q_ROWS + m
    mask_m = m < q_len
    tiles_per_partition = gl.cdiv(gl.cdiv(prefix_len, BLOCK_N), NUM_PARTITIONS)
    start = partition * tiles_per_partition * BLOCK_N
    end = gl.minimum(prefix_len, start + tiles_per_partition * BLOCK_N)
    e_max = gl.full(
        (BLOCK_M,), float("-inf"), gl.float32, layout=gl.SliceLayout(1, MMA)
    )
    if STORE_PROBS:
        earlier = gl.arange(0, NUM_PARTITIONS, layout=gl.SliceLayout(0, MMA))
        earlier_max = gl.load(
            Partition_Max + row[:, None] * NUM_PARTITIONS + earlier[None, :],
            mask=mask_m[:, None] & (earlier[None, :] < partition),
            other=float("-inf"),
        )
        e_max = gl.max(earlier_max, 1)

    for offset in range(start, end, BLOCK_N):
        mask_kn = offset + kn < prefix_len
        locations = gl.load(KV_Indices + kv_start + offset + kn, mask=mask_kn, other=0)
        k = gl.load(
            K_Buffer + locations[None, :] * stride_buf_kbs + kd[:, None],
            mask=mask_kn[None, :],
            other=0.0,
        )
        kpe = gl.load(
            K_Buffer + locations[None, :] * stride_buf_kbs + kdpe[:, None],
            mask=mask_kn[None, :],
            other=0.0,
        )
        k = gl.convert_layout(k, gl.DotOperandLayout(1, MMA, 16))
        kpe = gl.convert_layout(kpe, gl.DotOperandLayout(1, MMA, 16))
        qk = mfma(q, k, gl.full((BLOCK_M, BLOCK_N), 0.0, gl.float32, layout=MMA))
        qk = mfma(qpe, kpe, qk)
        qk *= sm_scale * k_scale
        mask = mask_m[:, None] & (offset + n[None, :] < prefix_len)
        qk = gl.where(mask, qk, float("-inf"))
        row_max = gl.max(qk, 1)
        row_max_fixed = gl.where(row_max == float("-inf"), -1e20, row_max)
        e_max = gl.maximum(e_max, row_max_fixed)
        if STORE_PROBS:
            p = gl.exp(qk - e_max[:, None])
            p_sum = gl.sum(p, 1)
            gl.store(
                Probabilities + row[:, None] * PREFIX_CAPACITY + offset + n[None, :],
                p.to(Probabilities.dtype.element_ty),
                mask=mask,
            )
            stat_offset = (row * (PREFIX_CAPACITY // BLOCK_N) + offset // BLOCK_N) * 2
            gl.store(Stats + stat_offset, e_max, mask=mask_m)
            gl.store(Stats + stat_offset + 1, p_sum, mask=mask_m)

    if not STORE_PROBS:
        gl.store(Partition_Max + row * NUM_PARTITIONS + partition, e_max, mask=mask_m)


def _prepare_fp8_prefix_probabilities(
    q, k_buffer, qo_indptr, kv_indptr, kv_indices, k_scale, sm_scale, max_len_extend
):
    """Prepare FP8 probabilities and FP32 tile maxima/sums for serial consumption.

    The caller gates the gfx942/Triton 3.6 specialization. Live lengths are
    checked on-device; skipped sequences must use the original consumer loop.
    """
    batch = qo_indptr.numel() - 1
    capacity = triton.cdiv(k_buffer.shape[0], _BLOCK_N) * _BLOCK_N
    # Scratch is batch * 12 * max_len_extend * capacity * (1 + 8 / 64)
    # bytes, plus the small FP32 partition-maxima tensor.
    probabilities = torch.empty(
        (batch, _NUM_HEADS, max_len_extend, capacity),
        dtype=k_buffer.dtype,
        device=q.device,
    )
    stats = torch.empty(
        (batch, _NUM_HEADS, max_len_extend, capacity // _BLOCK_N, 2),
        dtype=torch.float32,
        device=q.device,
    )
    if batch == 0 or capacity < 8192:
        return probabilities, stats, capacity

    partition_max = torch.empty(
        (batch, _NUM_HEADS, max_len_extend, _NUM_PARTITIONS),
        dtype=torch.float32,
        device=q.device,
    )
    grid = (batch, _NUM_HEADS, _NUM_PARTITIONS)
    # Same-stream launches make every partition maximum visible to the second
    # pass, which starts each partition from the preceding running maximum.
    for store_probs in (False, True):
        _prefix_probability_kernel[grid](
            q,
            k_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            partition_max,
            probabilities,
            stats,
            k_scale,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            PREFIX_CAPACITY=capacity,
            PREFIX_Q_ROWS=max_len_extend,
            NUM_PARTITIONS=_NUM_PARTITIONS,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            STORE_PROBS=store_probs,
            num_warps=4,
            num_stages=1,
            waves_per_eu=1,
        )
    return probabilities, stats, capacity
