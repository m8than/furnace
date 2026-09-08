import random
import unittest

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from sglang.kernels.ops.attention.extend_attention import (
    _compact_extend_q_tiles_per_head,
    build_unified_kv_indices,
    extend_attention_fwd,
    extend_attention_fwd_unified,
    redundant_attention,
)
from sglang.kernels.ops.attention.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.utils import get_device
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_amd_ci

# Triton attention kernel unit tests (decode, extend, prefill)
register_cuda_ci(est_time=19, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


def extend_attention_fwd_torch(
    q: torch.Tensor,  # [extend_tokens, H_Q, D]
    k: torch.Tensor,  # [extend_tokens, H_KV, D]
    v: torch.Tensor,  # [extend_tokens, H_KV, D]
    o: torch.Tensor,  # [extend_tokens, H_Q, D]
    k_cache: torch.Tensor,  # [total_tokens, H_KV, D]
    v_cache: torch.Tensor,  # [total_tokens, H_KV, D]
    qo_indptr: torch.Tensor,  # [B+1]
    kv_indptr: torch.Tensor,  # [B+1]
    kv_indices: torch.Tensor,  # [prefix_tokens]
    sliding_window_size: int,
):
    B = qo_indptr.size(0) - 1
    _, H_Q, D = q.shape
    _, H_KV, _ = k.shape

    group_size = H_Q // H_KV
    scale = 1.0 / D**0.5

    for i in range(B):
        q_start = int(qo_indptr[i].item())
        q_end = int(qo_indptr[i + 1].item())
        kv_start = int(kv_indptr[i].item())
        kv_end = int(kv_indptr[i + 1].item())

        prefix_indices = kv_indices[kv_start:kv_end]
        k_prefix = k_cache[prefix_indices]  # [prefix_len, H_KV, D]
        v_prefix = v_cache[prefix_indices]  # [prefix_len, H_KV, D]

        k_extend = k[q_start:q_end]  # [extend_len, H_KV, D]
        v_extend = v[q_start:q_end]  # [extend_len, H_KV, D]
        q_extend = q[q_start:q_end]  # [extend_len, H_Q,  D]

        k_full = torch.cat([k_prefix, k_extend], dim=0)  # [total_len, H_KV, D]
        v_full = torch.cat([v_prefix, v_extend], dim=0)  # [total_len, H_KV, D]

        if group_size != 1:
            k_full_hq = k_full.repeat_interleave(
                group_size, dim=1
            )  # [total_len, H_Q, D]
            v_full_hq = v_full.repeat_interleave(
                group_size, dim=1
            )  # [total_len, H_Q, D]
        else:
            k_full_hq = k_full
            v_full_hq = v_full

        prefix_len = k_prefix.size(0)
        extend_len = k_extend.size(0)
        total_len = prefix_len + extend_len

        # causal
        pos_keys = torch.arange(total_len, device=q.device)
        t = prefix_len + torch.arange(extend_len, device=q.device)  # [extend_len]
        causal_mask = pos_keys.unsqueeze(0) <= t.unsqueeze(1)

        # sliding window
        if sliding_window_size is not None and sliding_window_size > 0:
            start = (t - (sliding_window_size)).clamp_min(0)  # [extend_len]
        else:
            start = torch.zeros_like(t)
        window_mask = pos_keys.unsqueeze(0) >= start.unsqueeze(1)

        final_mask = causal_mask & window_mask

        attn_scores = (
            torch.einsum("qhd,khd->qhk", q_extend, k_full_hq) * scale
        )  # [extend_len, H_Q, total_len]
        attn_scores = attn_scores.masked_fill(~final_mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        o[q_start:q_end] = torch.einsum("qhk,khd->qhd", attn_weights, v_full_hq)


def decode_attention_fwd_torch(
    q: torch.Tensor,  # [B, H_Q, D]
    k_buffer: torch.Tensor,  # [total_tokens, H_KV, D]
    v_buffer: torch.Tensor,  # [total_tokens, H_KV, D]
    kv_indptr: torch.Tensor,  # [B+1]
    kv_indices: torch.Tensor,  # [prefix_tokens]
    sm_scale: float,
):
    """
    Torch reference implementation for decode attention with stable softmax.
    Supports both MHA and GQA configurations.
    """
    B = kv_indptr.size(0) - 1
    _, H_Q, D = q.shape
    _, H_KV, _ = k_buffer.shape

    assert H_Q % H_KV == 0, "H_Q must be divisible by H_KV for GQA"
    group_size = H_Q // H_KV

    o_ref = torch.empty((B, H_Q, D), dtype=torch.float32, device=q.device)

    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        idx = kv_indices[start:end]

        k_seq = k_buffer.index_select(0, idx)  # [L, H_KV, D]
        v_seq = v_buffer.index_select(0, idx)  # [L, H_KV, D]

        if H_KV != H_Q:
            k_seq = k_seq.repeat_interleave(group_size, dim=1)  # [L, H_Q, D]
            v_seq = v_seq.repeat_interleave(group_size, dim=1)  # [L, H_Q, D]

        q_f32 = q[b].to(torch.float32)  # [H_Q, D]
        k_f32 = k_seq.to(torch.float32)  # [L, H_Q, D]
        v_f32 = v_seq.to(torch.float32)  # [L, H_Q, D]

        # logits: [H_Q, L]
        logits = torch.einsum("hd,lhd->hl", q_f32, k_f32) * float(sm_scale)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.softmax(logits, dim=-1)  # [H_Q, L]

        # out: [H_Q, D]
        o_ref[b] = torch.einsum("hl,lhd->hd", p, v_f32)

    return o_ref


class TestTritonAttention(CustomTestCase):
    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):
        dtype = torch.bfloat16
        device = get_device()

        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device=device
        )
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device=device
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

        b_req_idx = torch.arange(B, dtype=torch.int32, device=device)
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        kv_indices = torch.zeros(
            (b_seq_len_prefix.sum().item(),), dtype=torch.int32, device=device
        )

        for i in range(B):
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device=device
            ).normal_(mean=0.1, std=0.2)

        o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
        o_extend_mask = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device=device
        )
        o_redundant = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device=device
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        custom_mask = None
        mask_indptr = None

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            True,
            mask_indptr,
            max_len_extend,
            1.0,
            1.0,
        )

        b_seq_mask_len = b_seq_len_extend * b_seq_len
        custom_mask = torch.ones(
            (b_seq_mask_len.sum().item(),), dtype=torch.bool, device=device
        )
        mask_indptr = torch.zeros((B + 1,), dtype=torch.int64, device=device)
        mask_indptr[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
        for i in range(B):
            causal_mask = (
                torch.tril(
                    torch.ones(b_seq_len_extend[i], b_seq_len_extend[i]), diagonal=0
                )
                == 1
            )
            prefix_mask = torch.ones(
                b_seq_len_extend[i], b_seq_len_prefix[i], dtype=torch.bool
            )
            mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
            custom_mask[mask_indptr[i] : mask_indptr[i + 1]] = mask_flatten

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend_mask,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            True,
            mask_indptr,
            max_len_extend,
            1.0,
            1.0,
        )

        redundant_attention(
            q_extend,
            o_redundant,
            k_buffer,
            v_buffer,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            b_seq_len_prefix,
            max_len_in_batch,
        )

        self.assertTrue(torch.allclose(o_extend, o_redundant, rtol=1e-2, atol=1e-3))
        self.assertTrue(
            torch.allclose(o_extend_mask, o_redundant, rtol=1e-2, atol=1e-3)
        )

    def test_extend_attention(self):

        # Define the varying parameter values
        # 256 covers the head_dim > 128 block-size branch (tuned on gfx95)
        attention_values = [256, 128, 96, 80, 13]

        # Loop through the values and call the method
        for value in attention_values:
            self._test_extend_attention_once(19, 12331, 12, 4, value)

    def test_gfx942_mla_verify_preserves_serial_output(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not ea._is_gfx942:
            self.skipTest("gfx942 target/draft MLA")
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(402)
        for heads, cache_dtype, causal, window, prefix_lens, extend_lens in (
            (12, torch.float8_e4m3fnuz, True, -1, [129], [8]),
            (12, torch.float8_e4m3fnuz, True, -1, [4097, 1000, 63, 0], [8, 5, 1, 0]),
            (12, torch.float8_e4m3fnuz, True, -1, [129], [10]),
            (12, torch.float8_e4m3fnuz, True, -1, [4097, 1000, 63, 0], [16, 12, 3, 0]),
            (8, torch.bfloat16, False, -1, [129], [8]),
            (8, torch.bfloat16, False, 4096, [4097, 1000, 63, 0], [12, 7, 3, 0]),
            (
                12,
                torch.float8_e4m3fnuz,
                True,
                -1,
                [129, 63, 0, 65] * 2,
                [12, 7, 0, 1] * 2,
            ),
            (8, torch.bfloat16, False, -1, [129, 63, 0, 65] * 4, [12, 7, 0, 1] * 4),
            (
                8,
                torch.bfloat16,
                False,
                4096,
                [4097, 63, 0, 65] * 16,
                [12, 7, 0, 1] * 16,
            ),
        ):
            with self.subTest(prefix=prefix_lens, extend=extend_lens):
                n_ext, n_prefix = sum(extend_lens), sum(prefix_lens)
                q = torch.randn(
                    n_ext + 2,
                    heads,
                    576,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                k = torch.randn(
                    n_ext + 2,
                    1,
                    576,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                k_buffer = torch.randn(
                    n_prefix + 2,
                    1,
                    576,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                ).to(cache_dtype)
                v = k[..., :512]
                v_buffer = k_buffer[..., :512]
                qo_indptr = torch.tensor(
                    [0, *torch.tensor(extend_lens).cumsum(0).tolist()],
                    dtype=torch.int32,
                    device=device,
                )
                kv_indptr = torch.tensor(
                    [0, *torch.tensor(prefix_lens).cumsum(0).tolist()],
                    dtype=torch.int32,
                    device=device,
                )
                kv_indices = (
                    torch.randperm(
                        n_prefix,
                        generator=generator,
                        device=device,
                    )
                    + 1
                )
                reference = torch.full(
                    (n_ext + 2, heads, 512),
                    -99,
                    dtype=torch.bfloat16,
                    device=device,
                )
                candidate = torch.full_like(reference, -99)

                def run(output):
                    extend_attention_fwd(
                        q,
                        k,
                        v,
                        output,
                        k_buffer,
                        v_buffer,
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        None,
                        causal,
                        None,
                        max(extend_lens),
                        1.0,
                        1.0,
                        sm_scale=192**-0.5,
                        sliding_window_size=window,
                        extend_seq_lens_cpu=extend_lens,
                    )

                with unittest.mock.patch.object(ea, "_is_gfx942", False):
                    run(reference)
                run(candidate)
                torch.testing.assert_close(
                    candidate.view(torch.int16),
                    reference.view(torch.int16),
                    atol=0,
                    rtol=0,
                )
                self.assertTrue(bool((candidate[n_ext:] == -99).all()))

    def test_gfx942_fp8_verify_index_capacity_preserves_serial_output(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not ea._is_gfx942:
            self.skipTest("gfx942 FP8 target verification")
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(9432)
        # Graph capacity is not prefix length. Exercise both a short live prefix
        # in a large index buffer and a genuinely long prefix spanning splits.
        for prefix_len in (129, 8193):
            with self.subTest(prefix_len=prefix_len):
                q = torch.randn(
                    12,
                    12,
                    576,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                k = torch.randn(
                    12, 1, 576, dtype=q.dtype, device=device, generator=generator
                )
                v = torch.randn(
                    12, 1, 512, dtype=q.dtype, device=device, generator=generator
                )
                kb = torch.randn(
                    prefix_len,
                    1,
                    576,
                    dtype=q.dtype,
                    device=device,
                    generator=generator,
                ).to(torch.float8_e4m3fnuz)
                vb = torch.randn(
                    prefix_len,
                    1,
                    512,
                    dtype=q.dtype,
                    device=device,
                    generator=generator,
                ).to(torch.float8_e4m3fnuz)
                qo_indptr = torch.tensor([0, 12], dtype=torch.int32, device=device)
                kv_indptr = torch.tensor(
                    [0, prefix_len], dtype=torch.int32, device=device
                )
                indices = torch.randperm(prefix_len, device=device, generator=generator)
                # Poison unused capacity: neither eager nor graph replay may
                # read it or let its extent change FP8 probability rounding.
                graph_indices = torch.full(
                    (16384,), -1, dtype=torch.int64, device=device
                )
                graph_indices[:prefix_len].copy_(indices)
                reference = torch.empty((12, 12, 512), dtype=q.dtype, device=device)
                candidate = torch.empty_like(reference)

                def run(output, kv_indices):
                    extend_attention_fwd(
                        q,
                        k,
                        v,
                        output,
                        kb,
                        vb,
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        None,
                        True,
                        None,
                        12,
                        1.7,
                        0.375,
                        sm_scale=192**-0.5,
                    )

                with unittest.mock.patch.object(ea, "_is_gfx942", False):
                    run(reference, indices)
                run(candidate, graph_indices)
                torch.testing.assert_close(candidate, reference, atol=0, rtol=0)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(candidate, graph_indices)
                v.add_(1)
                with unittest.mock.patch.object(ea, "_is_gfx942", False):
                    run(reference, indices)
                candidate.fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(candidate, reference, atol=0, rtol=0)

    def test_gfx942_fp8_verify_long_ragged_prefix_preserves_serial_output(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not ea._is_gfx942 or ea._triton_version_parts != (3, 6):
            self.skipTest("gfx942 FP8 target verification on Triton 3.6")
        device = get_device()
        # These seeds expose BF16 output changes from a different FP32
        # probability-sum reduction tree, even with identical FP8 probabilities.
        for batch, prefix, input_scale, seed in (
            (4, 120000, 0.25, 1730),
            (1, 1000000, 1.0, 1732),
        ):
            with self.subTest(
                batch=batch, prefix=prefix, input_scale=input_scale, seed=seed
            ):
                generator = torch.Generator(device=device).manual_seed(seed)

                def normal(shape, dtype):
                    return (
                        torch.randn(
                            shape,
                            dtype=torch.float32,
                            device=device,
                            generator=generator,
                        )
                        * input_scale
                    ).to(dtype)

                # Keep generation order and FP32 scaling: generating BF16
                # directly does not reproduce the reduction-sensitive inputs.
                q = normal((batch * 8, 12, 576), torch.bfloat16)
                k = normal((batch * 8, 1, 576), torch.bfloat16)
                v = normal((batch * 8, 1, 512), torch.bfloat16)
                k_buffer = normal((prefix, 1, 576), torch.float8_e4m3fnuz)
                v_buffer = k_buffer[..., :512]
                lengths = [prefix - 37 * request for request in range(batch)]
                offsets = [0]
                for length in lengths:
                    offsets.append(offsets[-1] + length)
                kv_indices = torch.cat(
                    [
                        torch.randperm(prefix, generator=generator, device=device)[
                            :length
                        ]
                        for length in lengths
                    ]
                ).to(torch.int32)
                qo_indptr = torch.arange(
                    0, (batch + 1) * 8, 8, dtype=torch.int32, device=device
                )
                kv_indptr = torch.tensor(offsets, dtype=torch.int32, device=device)
                reference = torch.empty(
                    (batch * 8, 12, 512), dtype=torch.bfloat16, device=device
                )
                candidate = torch.empty_like(reference)

                def run(output):
                    extend_attention_fwd(
                        q,
                        k,
                        v,
                        output,
                        k_buffer,
                        v_buffer,
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        None,
                        True,
                        None,
                        8,
                        0.73,
                        1.13,
                        sm_scale=192**-0.5,
                    )

                def assert_exact_output():
                    self.assertTrue(bool(torch.isfinite(reference).all()))
                    self.assertTrue(bool(torch.isfinite(candidate).all()))
                    torch.testing.assert_close(
                        candidate.view(torch.int16),
                        reference.view(torch.int16),
                        atol=0,
                        rtol=0,
                    )

                with unittest.mock.patch.object(ea, "_is_gfx942", False):
                    run(reference)
                run(candidate)
                assert_exact_output()

                if batch == 4:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run(candidate)
                    # Change prefix probabilities as well as fresh values so
                    # replay must recompute its statistics and final output.
                    q.mul_(1.5)
                    v.add_(128)
                    with unittest.mock.patch.object(ea, "_is_gfx942", False):
                        run(reference)
                    self.assertFalse(torch.equal(candidate, reference))
                    candidate.fill_(float("nan"))
                    graph.replay()
                    assert_exact_output()
                    del graph

    def test_gfx942_split_prefix_verify_reference(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not ea._is_gfx942:
            self.skipTest("gfx942 small-query split-prefix attention")
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(9402)
        # Long ragged tiles, a two-tile prefix, an entirely empty prefix, and
        # a zero-query request. Short rows leave most split partitions empty.
        prefix_lens, extend_lens = [32769, 65, 0, 4097], [16, 3, 1, 0]
        q_offsets = [0, *torch.tensor(extend_lens).cumsum(0).tolist()]
        kv_offsets = [0, *torch.tensor(prefix_lens).cumsum(0).tolist()]
        n_ext, n_prefix = q_offsets[-1], kv_offsets[-1]
        qo_indptr = torch.tensor(q_offsets, dtype=torch.int32, device=device)
        kv_indptr = torch.tensor(kv_offsets, dtype=torch.int32, device=device)
        kv_indices = torch.randperm(n_prefix, device=device, generator=generator)
        q = torch.randn(
            n_ext + 2,
            12,
            576,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        k = torch.randn(
            n_ext + 2,
            1,
            576,
            dtype=q.dtype,
            device=device,
            generator=generator,
        )
        v = (
            torch.randn(
                n_ext + 2,
                1,
                512,
                dtype=q.dtype,
                device=device,
                generator=generator,
            )
            + 3
        )
        prefix_k = torch.randn(
            n_prefix,
            1,
            576,
            dtype=q.dtype,
            device=device,
            generator=generator,
        )
        prefix_v = torch.randn(
            n_prefix,
            1,
            512,
            dtype=q.dtype,
            device=device,
            generator=generator,
        )
        scale, k_scale, v_scale = 192**-0.5, 1.7, 0.375
        for cache_dtype, use_mask, skip_prefix_mask in (
            (torch.bfloat16, False, True),
            (torch.float8_e4m3fnuz, False, True),
            (torch.float8_e4m3fnuz, True, False),
            (torch.float8_e4m3fnuz, True, True),
        ):
            with self.subTest(
                cache_dtype=cache_dtype,
                custom_mask=use_mask,
                skip_prefix_mask=skip_prefix_mask,
            ):
                kb, vb = prefix_k.to(cache_dtype), prefix_v.to(cache_dtype)
                masks, mask_offsets = [], [0]
                for prefix, length in zip(prefix_lens, extend_lens):
                    mask = torch.ones(
                        length, prefix + length, dtype=torch.bool, device=device
                    )
                    mask[:, prefix:] = torch.ones(
                        length, length, dtype=torch.bool, device=device
                    ).tril()
                    if use_mask and length:
                        # Prefix masks must be honored only when requested.
                        mask[:, :prefix] = False
                        if length > 1:
                            mask[1:, :prefix:127] = True
                            # Custom masks replace causality, including future
                            # fresh tokens, and may contain all-masked rows.
                            mask[0, prefix:] = False
                            mask[0, -1] = True
                            mask[-1, :] = False
                    masks.append(mask)
                    mask_offsets.append(mask_offsets[-1] + mask.numel())
                custom_mask = (
                    torch.cat([mask.flatten() for mask in masks]) if use_mask else None
                )
                mask_indptr = (
                    torch.tensor(mask_offsets, dtype=torch.int64, device=device)
                    if use_mask
                    else None
                )
                out = torch.full(
                    (n_ext + 2, 12, 512), -99, dtype=q.dtype, device=device
                )
                # Non-contiguous consumer LSE exercises its actual stride contract.
                lse = torch.full(
                    (12, n_ext + 2), -99, dtype=torch.float32, device=device
                ).T
                extend_attention_fwd(
                    q,
                    k,
                    v,
                    out,
                    kb,
                    vb,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    custom_mask,
                    True,
                    mask_indptr,
                    max(extend_lens),
                    k_scale,
                    v_scale,
                    sm_scale=scale,
                    skip_prefix_custom_mask=skip_prefix_mask,
                    lse_extend=lse,
                    extend_seq_lens_cpu=extend_lens,
                )
                kb_reference, vb_reference = kb.double(), vb.double()
                for seq, (prefix, length) in enumerate(zip(prefix_lens, extend_lens)):
                    if not length:
                        continue
                    start, end = q_offsets[seq : seq + 2]
                    indices = kv_indices[kv_offsets[seq] : kv_offsets[seq + 1]]
                    # Q is cast to the cache dtype for prefix QK; fresh Q/K/V
                    # stay BF16. FP64 logits independently check masks and LSE.
                    prefix_logits = torch.einsum(
                        "qhd,kd->qhk",
                        q[start:end].to(cache_dtype).double(),
                        kb_reference[indices, 0],
                    ) * (scale * k_scale)
                    fresh_logits = (
                        torch.einsum(
                            "qhd,kd->qhk",
                            q[start:end].double(),
                            k[start:end, 0].double(),
                        )
                        * scale
                    )
                    logits = torch.cat([prefix_logits, fresh_logits], dim=-1)
                    allowed = masks[seq].clone()
                    if not use_mask or skip_prefix_mask:
                        allowed[:, :prefix] = True
                    logits.masked_fill_(~allowed[:, None, :], -torch.inf)
                    ref_lse = logits.logsumexp(-1)
                    valid = allowed.any(-1)
                    values = torch.cat(
                        [vb_reference[indices, 0] * v_scale, v[start:end, 0].double()]
                    )
                    fp8 = cache_dtype == torch.float8_e4m3fnuz
                    if fp8:
                        # Serial FP8 attention rounds unnormalized probabilities
                        # after each tile's running-maximum update, not after a
                        # global softmax. Model those rounding boundaries with
                        # independent PyTorch FP64 arithmetic; fresh P·V rounds
                        # to BF16 instead. Neither stage quantizes the denominator.
                        numerator = torch.zeros(
                            (length, 12, 512), dtype=torch.float64, device=device
                        )
                        denominator = torch.zeros_like(ref_lse)
                        maximum = torch.full_like(ref_lse, -torch.inf)
                        for lo, hi, probability_dtype in (
                            (0, prefix, cache_dtype),
                            (prefix, prefix + length, v.dtype),
                        ):
                            for tile_start in range(lo, hi, 64):
                                tile_end = min(tile_start + 64, hi)
                                tile = logits[..., tile_start:tile_end]
                                tile_max = tile.amax(-1)
                                tile_max = torch.where(
                                    tile_max == -torch.inf, -1e20, tile_max
                                )
                                next_max = torch.maximum(maximum, tile_max)
                                rescale = torch.exp(maximum - next_max)
                                probabilities = torch.exp(tile - next_max[..., None])
                                denominator = denominator * rescale + probabilities.sum(
                                    -1
                                )
                                numerator = numerator * rescale[
                                    ..., None
                                ] + torch.einsum(
                                    "qhk,kd->qhd",
                                    probabilities.to(probability_dtype).double(),
                                    values[tile_start:tile_end],
                                )
                                maximum = next_max
                        reference = (
                            numerator
                            / torch.where(denominator > 0, denominator, 1.0)[..., None]
                        )
                    else:
                        weights = torch.where(
                            valid[:, None, None], logits, 0.0
                        ).softmax(-1)
                        weights = torch.where(valid[:, None, None], weights, 0.0)
                        reference = torch.einsum("qhk,kd->qhd", weights, values)
                    torch.testing.assert_close(
                        out[start:end].double(),
                        reference,
                        atol=1e-2 if fp8 else 3e-3,
                        rtol=6e-2 if fp8 else 1e-2,
                    )
                    torch.testing.assert_close(
                        lse[start:end].double(),
                        ref_lse,
                        atol=2e-4,
                        rtol=2e-5,
                    )
                self.assertTrue(bool((out[n_ext:] == -99).all()))
                self.assertTrue(bool((lse[n_ext:] == -99).all()))

    def test_gfx942_draft_split_prefix_noncausal_reference(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not ea._is_gfx942:
            self.skipTest("gfx942 DFlash MLA split-prefix attention")
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(9428)
        # Include empty prefix/split partitions and a graph-padding query row.
        prefix_lens = [32769, 32769, 65, 0, 129, 1, 0, 4097]
        extend_lens = [12, 7, 3, 12, 1, 8, 0, 0]
        q_offsets = [0, *torch.tensor(extend_lens).cumsum(0).tolist()]
        kv_offsets = [0, *torch.tensor(prefix_lens).cumsum(0).tolist()]
        n_ext, n_prefix = q_offsets[-1], kv_offsets[-1]
        qo_indptr = torch.tensor(q_offsets, dtype=torch.int32, device=device)
        kv_indptr = torch.tensor(kv_offsets, dtype=torch.int32, device=device)
        kv_indices = torch.randperm(n_prefix, device=device, generator=generator)
        q = torch.randn(
            n_ext + 2,
            8,
            576,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        k = torch.randn(
            n_ext + 2,
            1,
            576,
            dtype=q.dtype,
            device=device,
            generator=generator,
        )
        # Distinct fresh values expose both accidental causality and counting
        # the proposal block more than once during the split merge.
        v = (
            torch.randn(
                n_ext + 2,
                1,
                512,
                dtype=q.dtype,
                device=device,
                generator=generator,
            )
            + 3
        )
        kb = torch.randn(
            n_prefix,
            1,
            576,
            dtype=q.dtype,
            device=device,
            generator=generator,
        )
        vb = kb[..., :512]
        out = torch.full((n_ext + 2, 8, 512), -99, dtype=q.dtype, device=device)
        scale = 192**-0.5

        def run():
            extend_attention_fwd(
                q,
                k,
                v,
                out,
                kb,
                vb,
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                False,
                None,
                max(extend_lens),
                1.0,
                1.0,
                sm_scale=scale,
                extend_seq_lens_cpu=extend_lens,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Replay must consume live proposal values, not stale capture contents.
        v.add_(1)
        out.fill_(-99)
        graph.replay()
        for seq, length in enumerate(extend_lens):
            if not length:
                continue
            start, end = q_offsets[seq : seq + 2]
            indices = kv_indices[kv_offsets[seq] : kv_offsets[seq + 1]]
            keys = torch.cat([kb[indices, 0], k[start:end, 0]]).double()
            values = torch.cat([vb[indices, 0], v[start:end, 0]]).double()
            logits = torch.einsum("qhd,kd->qhk", q[start:end].double(), keys) * scale
            reference = torch.einsum("qhk,kd->qhd", logits.softmax(-1), values)
            torch.testing.assert_close(
                out[start:end].double(),
                reference,
                atol=3e-3,
                rtol=1e-2,
            )
        self.assertTrue(bool((out[n_ext:] == -99).all()))

    def test_gfx942_draft_split_window_reference(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not ea._is_gfx942:
            self.skipTest("gfx942 windowed DFlash MLA split-prefix attention")
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(16483)
        # Straddle the left-window boundary, with short/empty split partitions,
        # a zero-query request, and a nonstandard proposal length below sixteen.
        prefix_lens = [32769, 16383, 16384, 65, 0, 16399]
        extend_lens = [15, 3, 1, 7, 4, 0]
        q_offsets = [0, *torch.tensor(extend_lens).cumsum(0).tolist()]
        full_offsets = [0, *torch.tensor(prefix_lens).cumsum(0).tolist()]
        n_ext, n_prefix = q_offsets[-1], full_offsets[-1]
        qo_indptr = torch.tensor(q_offsets, dtype=torch.int32, device=device)
        full_indices = torch.randperm(n_prefix, device=device, generator=generator)
        q = torch.randn(
            n_ext + 2, 8, 576, dtype=torch.bfloat16, device=device, generator=generator
        )
        k = torch.randn(
            n_ext + 2, 1, 576, dtype=q.dtype, device=device, generator=generator
        )
        v = (
            torch.randn(
                n_ext + 2, 1, 512, dtype=q.dtype, device=device, generator=generator
            )
            + 3
        )
        kb = torch.randn(
            n_prefix, 1, 576, dtype=q.dtype, device=device, generator=generator
        )
        vb = kb[..., :512]
        scale = 192**-0.5
        # Make each row's leftmost key consequential: uniformly random long
        # prefixes can hide an off-by-one window error below BF16 tolerance.
        first_indices = full_indices[: prefix_lens[0]]
        q[: extend_lens[0]] = 0
        q[: extend_lens[0], :, 0] = 1

        for window, compact, use_mask in (
            (16383, False, False),
            (16383, True, False),
            (16383, True, True),
            (4095, False, False),
            (4095, True, False),
            (4095, True, True),
        ):
            with self.subTest(window=window, compact=compact, custom_mask=use_mask):
                kb[first_indices, 0, 0] = 0
                boundary = prefix_lens[0] - window
                boundary_indices = first_indices[
                    boundary - 1 : boundary + extend_lens[0]
                ]
                kb[boundary_indices, 0, 0] = 12 / scale
                # V is the latent prefix of K; use another coordinate for the signal.
                kb[boundary_indices, 0, 1] = torch.arange(
                    boundary_indices.numel(), device=device, dtype=q.dtype
                )
                retained = [
                    min(prefix, window) if compact else prefix for prefix in prefix_lens
                ]
                dropped = [prefix - kept for prefix, kept in zip(prefix_lens, retained)]
                kv_offsets = [0, *torch.tensor(retained).cumsum(0).tolist()]
                indices = torch.cat(
                    [
                        full_indices[full_offsets[seq] + offset : full_offsets[seq + 1]]
                        for seq, offset in enumerate(dropped)
                    ]
                )
                # Graph capacity must not become live history, even for short
                # rows. Poison padding and pass nonzero absolute mask offsets.
                kv_indices = torch.full(
                    (max(indices.numel(), len(retained) * 2048) + 64,),
                    -1,
                    dtype=torch.int64,
                    device=device,
                )
                kv_indices[: indices.numel()] = indices
                kv_indptr = torch.tensor(kv_offsets, dtype=torch.int32, device=device)
                window_offsets = torch.tensor(dropped, dtype=torch.int32, device=device)
                masks, mask_offsets = [], [0]
                for prefix, length in zip(prefix_lens, extend_lens):
                    mask = torch.ones(
                        length, prefix + length, dtype=torch.bool, device=device
                    )
                    if use_mask and length:
                        mask[:, :prefix:3] = False
                        # An empty softmax row and a row that sees only the
                        # final future proposal defend merge and noncausality.
                        mask[-1] = False
                        if length > 1:
                            mask[0] = False
                            mask[0, -1] = True
                    masks.append(mask)
                    mask_offsets.append(mask_offsets[-1] + mask.numel())
                custom_mask = (
                    torch.cat([mask.flatten() for mask in masks]) if use_mask else None
                )
                mask_indptr = (
                    torch.tensor(mask_offsets, dtype=torch.int64, device=device)
                    if use_mask
                    else None
                )
                out = torch.full((n_ext + 2, 8, 512), -99, dtype=q.dtype, device=device)
                lse = torch.full(
                    (8, n_ext + 2), -99, dtype=torch.float32, device=device
                ).T

                def run():
                    extend_attention_fwd(
                        q,
                        k,
                        v,
                        out,
                        kb,
                        vb,
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        custom_mask,
                        False,
                        mask_indptr,
                        max(extend_lens),
                        1.0,
                        1.0,
                        sm_scale=scale,
                        sliding_window_size=window,
                        window_kv_offsets=window_offsets,
                        skip_prefix_custom_mask=False,
                        lse_extend=lse,
                        extend_seq_lens_cpu=extend_lens,
                    )

                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                v.add_(0.25)
                out.fill_(-99)
                lse.fill_(-99)
                graph.replay()
                # Reference uses full, absolute history, never the compact
                # indices or split boundaries supplied to the implementation.
                for seq, (prefix, length) in enumerate(zip(prefix_lens, extend_lens)):
                    if not length:
                        continue
                    start, end = q_offsets[seq : seq + 2]
                    absolute_indices = full_indices[
                        full_offsets[seq] : full_offsets[seq + 1]
                    ]
                    keys = torch.cat(
                        [kb[absolute_indices, 0], k[start:end, 0]]
                    ).double()
                    values = torch.cat(
                        [vb[absolute_indices, 0], v[start:end, 0]]
                    ).double()
                    logits = (
                        torch.einsum("qhd,kd->qhk", q[start:end].double(), keys) * scale
                    )
                    positions = prefix + torch.arange(length, device=device)
                    key_positions = torch.arange(prefix + length, device=device)
                    allowed = key_positions[None, :] >= positions[:, None] - window
                    if use_mask:
                        allowed &= masks[seq]
                    logits.masked_fill_(~allowed[:, None, :], -torch.inf)
                    valid = allowed.any(-1)
                    weights = torch.where(valid[:, None, None], logits, 0.0).softmax(-1)
                    weights = torch.where(valid[:, None, None], weights, 0.0)
                    reference = torch.einsum("qhk,kd->qhd", weights, values)
                    torch.testing.assert_close(
                        out[start:end].double(), reference, atol=3e-3, rtol=1e-2
                    )
                    torch.testing.assert_close(
                        lse[start:end].double(),
                        logits.logsumexp(-1),
                        atol=2e-4,
                        rtol=2e-5,
                    )
                self.assertTrue(bool((out[n_ext:] == -99).all()))
                self.assertTrue(bool((lse[n_ext:] == -99).all()))

    def test_gfx942_chunked_prefill_preserves_serial_output(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not ea._is_gfx942:
            self.skipTest("gfx942 long-prefill query tiling")
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(403)
        length = 32769
        q = torch.randn(
            length + 2,
            12,
            192,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        k = torch.randn_like(q)
        v = torch.randn(length + 2, 12, 128, dtype=q.dtype, device=device)
        cache = torch.empty((1, 1, 576), dtype=torch.float8_e4m3fnuz, device=device)
        qo_indptr = torch.tensor([0, length], dtype=torch.int32, device=device)
        kv_indptr = torch.zeros(2, dtype=torch.int32, device=device)
        kv_indices = torch.empty(0, dtype=torch.int64, device=device)
        reference = torch.full_like(v, -99)
        candidate = torch.full_like(v, -99)

        def run(output):
            extend_attention_fwd(
                q,
                k,
                v,
                output,
                cache,
                cache[..., :512],
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                True,
                None,
                length,
                1.0,
                1.0,
                sm_scale=192**-0.5,
                extend_seq_lens_cpu=[length],
            )

        with unittest.mock.patch.object(ea, "_is_gfx942", False):
            run(reference)
        run(candidate)
        torch.testing.assert_close(
            candidate.view(torch.int16), reference.view(torch.int16), atol=0, rtol=0
        )
        self.assertTrue(bool((candidate[length:] == -99).all()))

    def test_gfx942_native_prefill_matches_causal_fp64(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not ea._is_gfx942:
            self.skipTest("gfx942 native QK192/V128 prefill")
        try:
            __import__("aiter.ops.mha")
        except ImportError:
            self.skipTest("AITER is not installed")

        device = get_device()
        generator = torch.Generator(device=device).manual_seed(704)
        length = 32768
        q, k = [
            torch.randn(
                length,
                12,
                192,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            for _ in range(2)
        ]
        v = torch.randn(
            length, 12, 128, dtype=q.dtype, device=device, generator=generator
        )
        out = torch.empty_like(v)
        qo_indptr = torch.tensor([0, length], dtype=torch.int32, device=device)
        kv_indptr = torch.zeros(2, dtype=torch.int32, device=device)
        kv_indices = torch.empty(0, dtype=torch.int64, device=device)
        with temp_set_env(allow_sglang=True, SGLANG_USE_AITER="1"):
            extend_attention_fwd(
                q,
                k,
                v,
                out,
                k[:0],
                v[:0],
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                True,
                None,
                length,
                1.0,
                1.0,
                sm_scale=192**-0.5,
                extend_seq_lens_cpu=[length],
            )

        rows = torch.tensor(
            [0, 1, 127, 128, 255, 256, length // 2, length - 1],
            device=device,
        )
        logits = (
            q[rows].transpose(0, 1).double()
            @ k.transpose(0, 1).double().transpose(1, 2)
        ) * 192**-0.5
        logits.masked_fill_(
            torch.arange(length, device=device)[None, :] > rows[:, None],
            -torch.inf,
        )
        reference = (logits.softmax(dim=-1) @ v.transpose(0, 1).double()).transpose(
            0, 1
        )
        torch.testing.assert_close(out[rows].double(), reference, atol=2e-3, rtol=1e-2)

    def test_extend_attention_triton37_lq576_n32(self):
        from sglang.kernels.ops.attention import extend_attention as ea

        if not (ea._is_gfx95 and ea._is_triton_ge_37):
            self.skipTest("Triton >=3.7 gfx950-only spill workaround")

        device = get_device()
        dtype = torch.bfloat16
        extend_lens = [32, 23]
        prefix_lens = [64, 96]
        h_q, h_kv, l_q, l_v = 12, 1, 576, 512
        n_ext, n_prefix = sum(extend_lens), sum(prefix_lens)

        q = torch.randn(n_ext, h_q, l_q, dtype=dtype, device=device)
        k = torch.randn(n_ext, h_kv, l_q, dtype=dtype, device=device)
        v = torch.randn(n_ext, h_kv, l_v, dtype=dtype, device=device)
        k_buffer = torch.randn(n_prefix, h_kv, l_q, dtype=dtype, device=device)
        v_buffer = torch.randn(n_prefix, h_kv, l_v, dtype=dtype, device=device)
        qo_indptr = torch.tensor([0, 32, 55], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, 64, 160], dtype=torch.int32, device=device)
        kv_indices = torch.arange(n_prefix, dtype=torch.int64, device=device)
        reference = torch.empty(n_ext, h_q, l_v, dtype=dtype, device=device)
        candidate = torch.empty_like(reference)

        def run(output):
            extend_attention_fwd(
                q,
                k,
                v,
                output,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                None,
                True,
                None,
                max(extend_lens),
                1.0,
                1.0,
                sm_scale=1.0 / (l_q**0.5),
                extend_seq_lens_cpu=extend_lens,
            )

        with unittest.mock.patch.object(ea, "_is_triton_ge_37", False):
            run(reference)
        with unittest.mock.patch.object(ea, "_is_triton_ge_37", True):
            run(candidate)
        torch.testing.assert_close(candidate, reference, atol=2e-2, rtol=1e-2)

    def test_compact_extend_attention_tile_count(self):
        self.assertEqual(
            _compact_extend_q_tiles_per_head(
                batch_size=16,
                max_len_extend=1000,
                total_extend_tokens=1015,
                block_m=64,
                extend_seq_lens_cpu=[1] * 15 + [1000],
            ),
            31,
        )
        self.assertEqual(
            _compact_extend_q_tiles_per_head(
                batch_size=2,
                max_len_extend=4224,
                total_extend_tokens=5376,
                block_m=64,
                extend_seq_lens_cpu=[1152, 4224],
            ),
            84,
        )
        self.assertIsNone(
            _compact_extend_q_tiles_per_head(
                batch_size=4,
                max_len_extend=64,
                total_extend_tokens=256,
                block_m=64,
                extend_seq_lens_cpu=[64, 64, 64, 64],
            )
        )

    def test_extend_attention_compact_grid(self):
        dtype = torch.bfloat16
        device = get_device()
        B, H_Q, H_KV, D = 4, 8, 2, 64

        b_seq_len_prefix = torch.tensor(
            [8, 16, 32, 64], dtype=torch.int32, device=device
        )
        b_seq_len_extend = torch.tensor(
            [1, 7, 13, 129], dtype=torch.int32, device=device
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix, dim=0)
        kv_indices = torch.empty(
            (int(b_seq_len_prefix.sum().item()),), dtype=torch.int32, device=device
        )
        for i in range(B):
            kv_indices[int(kv_indptr[i]) : int(kv_indptr[i + 1])] = torch.arange(
                int(b_start_loc[i].item()),
                int(b_start_loc[i].item()) + int(b_seq_len_prefix[i].item()),
                device=device,
            )

        total_token_num = int(b_seq_len.sum().item())
        extend_token_num = int(b_seq_len_extend.sum().item())
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)
        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (int(b_seq_len_extend[i].item()), H_Q, D),
                dtype=dtype,
                device=device,
            ).normal_(mean=0.1, std=0.2)

        max_len_extend = int(b_seq_len_extend.max().item())
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend, dim=0)
        extend_seq_lens_cpu = b_seq_len_extend.cpu().tolist()

        o_legacy = torch.empty_like(q_extend)
        o_compact = torch.empty_like(q_extend)
        for output, use_compact in ((o_legacy, False), (o_compact, True)):
            with temp_set_env(
                allow_sglang=True,
                SGLANG_TRITON_COMPACT_EXTEND_ATTENTION=str(use_compact),
            ):
                extend_attention_fwd(
                    q_extend,
                    k_extend,
                    v_extend,
                    output,
                    k_buffer,
                    v_buffer,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    custom_mask=None,
                    is_causal=True,
                    mask_indptr=None,
                    max_len_extend=max_len_extend,
                    k_scale=1.0,
                    v_scale=1.0,
                    extend_seq_lens_cpu=extend_seq_lens_cpu,
                )

        self.assertTrue(
            torch.allclose(o_legacy, o_compact, rtol=1e-2, atol=1e-3),
            f"compact grid output differs from legacy grid. "
            f"Max diff: {(o_legacy - o_compact).abs().max()}",
        )

    def _test_extend_attention_sliding_window_once(
        self, B, N_CTX, H_Q, H_KV, D, WINDOW_SIZE
    ):
        dtype = torch.bfloat16
        device = get_device()

        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device=device
        )
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device=device
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend

        b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        kv_indices = torch.zeros(
            (b_seq_len_prefix.sum().item(),), dtype=torch.int32, device=device
        )

        for i in range(B):
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device=device
            ).normal_(mean=0.1, std=0.2)

        o_extend_triton = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device=device
        )
        o_extend_torch = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device=device
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend_triton,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=max_len_extend,
            k_scale=1.0,
            v_scale=1.0,
            sliding_window_size=WINDOW_SIZE,
        )

        extend_attention_fwd_torch(
            q_extend,
            k_extend,
            v_extend,
            o_extend_torch,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            WINDOW_SIZE,
        )

        self.assertTrue(
            torch.allclose(o_extend_triton, o_extend_torch, rtol=1e-3, atol=1e-3)
        )

    def test_extend_attention_sliding_window(self):
        window_sizes = [-1, 127]
        for window_size in window_sizes:
            self._test_extend_attention_sliding_window_once(
                19, 12331, 64, 8, 128, window_size
            )

    def _test_context_attention_once(self, head_dim, is_causal):
        # Set up a simple test case
        device = get_device()
        num_heads = 4
        seq_lens = [8, 12]
        max_seq_len = max(seq_lens)

        # Create random input tensors
        q = torch.randn(sum(seq_lens), num_heads, head_dim, device=device)
        k = torch.randn(sum(seq_lens), num_heads, head_dim, device=device)
        v = torch.randn(sum(seq_lens), num_heads, head_dim, device=device)
        o = torch.zeros(sum(seq_lens), num_heads, head_dim, device=device)

        # Create b_start_loc and b_seq_len tensors
        b_start_loc = torch.tensor([0, seq_lens[0]], device=device)
        b_seq_len = torch.tensor(seq_lens, device=device)

        context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
        )

        cu_seq_lens = [0] * (len(seq_lens) + 1)
        for i, seq_len in enumerate(seq_lens):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

        for i in range(len(seq_lens)):
            start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
            o_torch = torch.nn.functional.scaled_dot_product_attention(
                q[start:end].permute(1, 0, 2),
                k[start:end].permute(1, 0, 2),
                v[start:end].permute(1, 0, 2),
                is_causal=is_causal,
            ).permute(1, 0, 2)

            cos_sim = torch.nn.functional.cosine_similarity(
                o[start:end].flatten(), o_torch.flatten(), dim=0
            )
            self.assertTrue(cos_sim.item() > 1 - (1e-5))
            self.assertTrue(torch.allclose(o[start:end], o_torch, atol=1e-2))

    def test_context_attention(self):
        head_dim = [128, 96, 80, 13]

        for dim in head_dim:
            for is_causal in [True, False]:
                self._test_context_attention_once(dim, is_causal)

    def _test_decode_attention_once(self, B, H_Q, H_KV, D):
        device = get_device()
        dtype = torch.bfloat16
        seq_len = 10  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device=device)

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
        v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D, dtype=dtype, device=device)

        b_seq_len = torch.full((B,), seq_len, device=device)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens, device=device)

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D),
            dtype=torch.float32,
            device=device,
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device=device,
        )

        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            1.0,
            1.0,
        )

        # Correctness reference (float32, stable softmax)
        o_ref = decode_attention_fwd_torch(
            q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
        )

        max_abs_err = (o.to(torch.float32) - o_ref).abs().max().item()
        self.assertTrue(
            torch.allclose(o.to(torch.float32), o_ref, atol=1e-2, rtol=1e-2),
            msg=f"decode_attention mismatch, max_abs_err={max_abs_err}",
        )

    def test_decode_attention(self):
        # Test configurations
        configs = [
            (2, 4, 4, 64),  # MHA
            (2, 4, 2, 64),  # GQA
            (2, 4, 4, 80),  # Non-standard head dim
            (2, 4, 4, 13),  # Prime number head dim
        ]

        for B, H_Q, H_KV, D in configs:
            self._test_decode_attention_once(B, H_Q, H_KV, D)

    def _test_grouped_decode_attention_once(self, B, S, H_Q, H_KV, D, D_V):
        dtype = torch.bfloat16
        device = get_device()
        seq_len = S  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device=device)

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device=device)

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

        b_seq_len = torch.full((B,), seq_len, device=device)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens, device=device)

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device=device,
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device=device,
        )

        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            1.0,
        )

        attn_logits1 = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device=device,
        )
        attn_lse1 = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device=device,
        )

        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o_grouped,
            kv_indptr,
            kv_indices,
            attn_logits1,
            attn_lse1,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            1.0,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        print(cos_sim.item())
        self.assertTrue(cos_sim.item() > 0.99)
        if is_in_amd_ci():
            self.assertTrue(torch.allclose(o, o_grouped, atol=5e-2))
        else:
            self.assertTrue(torch.allclose(o, o_grouped, atol=3e-2))

    def test_grouped_decode_attention(self):
        seq_lens = [5, 100, 128, 500]
        configs = [
            (2, 16, 16, 64, 64),
            (2, 16, 1, 64, 64),
            (2, 64, 1, 13, 13),
            (2, 128, 1, 80, 80),
            (2, 128, 2, 512, 512),
            (2, 128, 1, 576, 512),
        ]

        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                self._test_grouped_decode_attention_once(B, S, H_Q, H_KV, D, D_V)

    def test_decode_attention_large_batch_int64_offset(self):
        """Regression for int32 Mid_O offset overflow (PR #28788).

        Under deterministic inference, max_kv_splits ~= ceil(context_len / 256)
        can be ~792 for long-context MLA models. Combined with CUDA-graph batch
        sizes, batch * num_head * max_kv_splits * head_dim can exceed 2**31 and
        int32 cur_batch * stride_mid_ob overflows into a GPU memory fault.
        """
        device = get_device()
        dtype = torch.bfloat16
        B = 64
        H_Q = 128
        H_KV = 1
        D = 576
        D_V = 512
        max_kv_splits = 792
        seq_len = 256
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        num_kv_splits = torch.full(
            (B,), max_kv_splits, dtype=torch.int32, device=device
        )

        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device=device)
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
        kv_indptr = torch.arange(
            0, (B + 1) * seq_len, seq_len, dtype=torch.int32, device=device
        )
        kv_indices = torch.arange(total_tokens, device=device)
        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D_V), dtype=torch.float32, device=device
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits), dtype=torch.float32, device=device
        )

        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            1.0,
            1.0,
            has_mla=True,
        )

        self.assertTrue(torch.isfinite(o).all())

    def _test_extend_attention_unified_vs_regular_once(self, B, N_CTX, H_Q, H_KV, D):
        """Test that unified kernel produces same results as 2-stage kernel."""
        dtype = torch.bfloat16
        device = get_device()

        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device=device
        )
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device=device
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend

        b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        # Setup prefix KV indices
        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        kv_indices = torch.zeros(
            (b_seq_len_prefix.sum().item(),), dtype=torch.int64, device=device
        )

        for i in range(B):
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)

        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device=device
            ).normal_(mean=0.1, std=0.2)

        # Setup for extend attention
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        # Run 2-stage kernel
        o_regular = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_regular,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=max_len_extend,
            k_scale=1.0,
            v_scale=1.0,
        )

        # Build unified KV indices
        extend_kv_indices = torch.arange(
            total_token_num - extend_token_num,
            total_token_num,
            dtype=torch.int64,
            device=device,
        )
        extend_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
        extend_start_loc[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        unified_kv_indptr, unified_kv_indices, prefix_lens = build_unified_kv_indices(
            kv_indptr,
            kv_indices,
            extend_start_loc,
            b_seq_len_extend,
            extend_kv_indices,
            B,
        )

        # Run unified kernel
        o_unified = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
        extend_attention_fwd_unified(
            q_extend,
            o_unified,
            k_buffer,
            v_buffer,
            1.0,
            1.0,
            qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            max_len_extend=max_len_extend,
            custom_mask=None,
            mask_indptr=None,
            sm_scale=None,
            logit_cap=0.0,
            is_causal=True,
        )

        # Compare results
        if is_in_amd_ci():
            self.assertTrue(
                torch.allclose(o_regular, o_unified, rtol=0.15, atol=0.17),
                f"Unified kernel output differs from 2-stage kernel. "
                f"Max diff: {(o_regular - o_unified).abs().max()}",
            )
        else:
            self.assertTrue(
                torch.allclose(o_regular, o_unified, rtol=0.15, atol=0.15),
                f"Unified kernel output differs from 2-stage kernel. "
                f"Max diff: {(o_regular - o_unified).abs().max()}",
            )

    def test_extend_attention_unified_vs_regular(self):
        """Test unified kernel matches 2-stage kernel across different configs."""
        configs = [
            (4, 512, 32, 8, 128),  # Standard config
            (2, 2048, 32, 8, 128),  # Long sequence (test 2048 specifically)
            (8, 256, 64, 8, 80),  # Non-standard head dim
        ]

        for B, N_CTX, H_Q, H_KV, D in configs:
            with self.subTest(B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D):
                self._test_extend_attention_unified_vs_regular_once(
                    B, N_CTX, H_Q, H_KV, D
                )

    def test_build_unified_kv_indices(self):
        """Test build_unified_kv_indices correctness."""
        B = 4
        dtype = torch.int64
        device = get_device()

        # Setup test data
        prefix_lens = torch.tensor([10, 20, 15, 25], dtype=torch.int32, device=device)
        extend_lens = torch.tensor([5, 3, 7, 4], dtype=torch.int32, device=device)

        # Build prefix indices
        prefix_kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        prefix_kv_indptr[1:] = torch.cumsum(prefix_lens, dim=0)
        prefix_kv_indices = torch.arange(
            prefix_lens.sum().item(), dtype=dtype, device=device
        )

        # Build extend indices
        extend_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
        extend_start_loc[1:] = torch.cumsum(extend_lens[:-1], dim=0)
        extend_kv_indices = torch.arange(
            prefix_lens.sum().item(),
            prefix_lens.sum().item() + extend_lens.sum().item(),
            dtype=dtype,
            device=device,
        )

        # Build unified indices
        unified_kv_indptr, unified_kv_indices, returned_prefix_lens = (
            build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_lens,
                extend_kv_indices,
                B,
            )
        )

        # Verify unified_kv_indptr
        expected_lens = prefix_lens + extend_lens
        expected_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
        expected_indptr[1:] = torch.cumsum(expected_lens, dim=0)
        self.assertTrue(torch.equal(unified_kv_indptr, expected_indptr))

        # Verify prefix_lens
        self.assertTrue(torch.equal(returned_prefix_lens, prefix_lens))

        # Verify unified_kv_indices structure
        for i in range(B):
            start_idx = int(unified_kv_indptr[i])
            end_idx = int(unified_kv_indptr[i + 1])
            prefix_len = int(prefix_lens[i])
            extend_len = int(extend_lens[i])

            # Check that prefix and extend are concatenated correctly
            unified_seq = unified_kv_indices[start_idx:end_idx]
            self.assertEqual(len(unified_seq), prefix_len + extend_len)

    def test_gfx942_chunked_mha_matches_full_causal_attention(self):
        from types import SimpleNamespace

        from sglang.kernels.ops.attention import extend_attention as ea
        from sglang.kernels.ops.attention.merge_state import merge_state_triton
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        if not ea._is_gfx942:
            self.skipTest("gfx942 bounded MHA prefill")
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(9433)
        for q_len, prefix_lens, chunk_size in (
            (7, [129, 3, 0], 64),
            (32768, [257], 128),
        ):
            with self.subTest(q_len=q_len, prefix_lens=prefix_lens):
                batch_size = len(prefix_lens)
                q = (
                    torch.randn(
                        batch_size * q_len,
                        12,
                        192,
                        dtype=torch.bfloat16,
                        device=device,
                        generator=generator,
                    )
                    * 0.2
                )
                k = torch.randn(
                    q.shape, dtype=q.dtype, device=device, generator=generator
                )
                v = (
                    torch.randn(
                        batch_size * q_len,
                        12,
                        128,
                        dtype=q.dtype,
                        device=device,
                        generator=generator,
                    )
                    + 3
                )
                prefix_k = [
                    torch.randn(
                        n, 12, 192, dtype=q.dtype, device=device, generator=generator
                    )
                    for n in prefix_lens
                ]
                prefix_v = [
                    torch.randn(
                        n, 12, 128, dtype=q.dtype, device=device, generator=generator
                    )
                    - 1
                    for n in prefix_lens
                ]
                chunks = []
                for start in range(0, max(prefix_lens), chunk_size):
                    lens = [max(0, min(chunk_size, n - start)) for n in prefix_lens]
                    chunks.append((start, lens))
                metadata = SimpleNamespace(
                    qo_indptr=torch.arange(
                        batch_size + 1, device=device, dtype=torch.int32
                    )
                    * q_len,
                    max_extend_len=q_len,
                )
                backend = TritonAttnBackend.__new__(TritonAttnBackend)
                backend.device = device
                backend.forward_metadata = metadata
                backend.extend_attention_fwd = extend_attention_fwd
                forward_batch = SimpleNamespace(
                    batch_size=batch_size,
                    extend_seq_lens_cpu=[q_len] * batch_size,
                    prefix_chunk_num_tokens=[sum(lens) for _, lens in chunks],
                    prefix_chunk_cu_seq_lens=[
                        torch.tensor(
                            [0, *torch.tensor(lens).cumsum(0).tolist()],
                            device=device,
                            dtype=torch.int32,
                        )
                        for _, lens in chunks
                    ],
                    attn_attend_prefix_cache=False,
                )
                backend.init_mha_chunk_metadata(forward_batch)
                layer = SimpleNamespace(
                    tp_q_head_num=12,
                    qk_head_dim=192,
                    v_head_dim=128,
                    scaling=192**-0.5,
                )
                with temp_set_env(allow_sglang=True, SGLANG_USE_AITER="1"):
                    out, lse = backend._forward_mha_chunked_kv(
                        q, k, v, layer, forward_batch
                    )
                    forward_batch.attn_attend_prefix_cache = True
                    for index, (start, lens) in enumerate(chunks):
                        forward_batch.prefix_chunk_idx = index
                        kk = torch.cat(
                            [x[start : start + n] for x, n in zip(prefix_k, lens)]
                        )
                        vv = torch.cat(
                            [x[start : start + n] for x, n in zip(prefix_v, lens)]
                        )
                        part, part_lse = backend._forward_mha_chunked_kv(
                            q, kk, vv, layer, forward_batch
                        )
                        merge_state_triton(out, lse, part, part_lse, out, lse)
                for seq, prefix in enumerate(prefix_lens):
                    first = seq * q_len
                    rows = torch.tensor(
                        sorted({0, q_len // 2, q_len - 1}), device=device
                    )
                    keys = torch.cat([prefix_k[seq], k[first : first + q_len]]).double()
                    values = torch.cat(
                        [prefix_v[seq], v[first : first + q_len]]
                    ).double()
                    logits = (
                        torch.einsum("rhd,khd->rhk", q[first + rows].double(), keys)
                        * layer.scaling
                    )
                    allowed = (
                        torch.arange(prefix + q_len, device=device)[None, :]
                        <= prefix + rows[:, None]
                    )
                    logits.masked_fill_(~allowed[:, None, :], -torch.inf)
                    expected = torch.einsum("rhk,khd->rhd", logits.softmax(-1), values)
                    torch.testing.assert_close(
                        out[first + rows].double(), expected, atol=2e-2, rtol=1e-2
                    )
                    torch.testing.assert_close(
                        lse[first + rows].double(),
                        logits.logsumexp(-1),
                        atol=3e-3,
                        rtol=1e-3,
                    )

    def test_mla_dcp_verify_replay_preserves_global_causality(self):
        device = get_device()
        generator = torch.Generator(device=device).manual_seed(713)
        q = (
            torch.randn(
                8, 4, 576, dtype=torch.bfloat16, device=device, generator=generator
            )
            * 0.15
        )
        current = (
            torch.randn(8, 1, 576, dtype=q.dtype, device=device, generator=generator)
            * 0.15
        )
        prefix = (
            torch.randn(17, 1, 576, dtype=q.dtype, device=device, generator=generator)
            * 0.15
        )
        qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device=device)
        scale = 576**-0.5
        states = []
        for rank in range(8):
            cache = torch.zeros(3, 1, 576, dtype=q.dtype, device=device)
            kv_indptr = torch.zeros(2, dtype=torch.int32, device=device)
            indices = torch.arange(3, dtype=torch.int64, device=device)
            prefix_len = torch.ones(1, dtype=torch.int32, device=device)
            out = torch.empty(8, 4, 512, dtype=torch.float32, device=device)
            lse = torch.empty(8, 4, dtype=torch.float32, device=device)

            def run():
                extend_attention_fwd(
                    q,
                    current,
                    current[..., :512],
                    out,
                    cache,
                    cache[..., :512],
                    qo_indptr,
                    kv_indptr,
                    indices,
                    None,
                    True,
                    None,
                    8,
                    1.0,
                    1.0,
                    sm_scale=scale,
                    lse_extend=lse,
                    dcp_size=8,
                    dcp_rank=rank,
                    global_prefix_lens=prefix_len,
                )

            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            states.append((graph, cache, kv_indptr, indices, prefix_len, out, lse))
        for length in (0, 1, 7, 8, 9, 17):
            for rank, state in enumerate(states):
                with self.subTest(prefix=length, rank=rank):
                    graph, cache, kv_indptr, indices, prefix_len, out, lse = state
                    local = prefix[rank:length:8]
                    cache.zero_()
                    cache[: local.shape[0]].copy_(local)
                    kv_indptr[1] = local.shape[0]
                    prefix_len.fill_(length)
                    graph.replay()

                    keys = torch.cat((local, current), dim=0).squeeze(1).float()
                    positions = torch.cat(
                        (
                            torch.arange(rank, length, 8, device=device)
                            if length > rank
                            else torch.empty(0, device=device),
                            torch.arange(length, length + 8, device=device),
                        )
                    )
                    visible = (
                        positions[None, :]
                        <= torch.arange(length, length + 8, device=device)[:, None]
                    ) & (positions[None, :] % 8 == rank)
                    scores = torch.einsum("qhd,kd->qhk", q.float(), keys) * scale
                    scores.masked_fill_(~visible[:, None, :], -torch.inf)
                    expected_lse = scores.logsumexp(-1)
                    probabilities = torch.nan_to_num(scores.softmax(-1), nan=0.0)
                    expected_out = torch.einsum(
                        "qhk,kd->qhd", probabilities, keys[:, :512]
                    )
                    torch.testing.assert_close(lse, expected_lse, atol=1e-4, rtol=1e-4)
                    torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=1e-2)
                    empty = torch.isneginf(expected_lse)
                    self.assertTrue(torch.equal(torch.isneginf(lse), empty))
                    self.assertEqual(torch.count_nonzero(out[empty]).item(), 0)


if __name__ == "__main__":
    unittest.main()
