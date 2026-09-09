"""Numerical and ownership invariants for token-striped MiniMax attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.dcp import (
    merge_topk_candidates,
    owner_attention,
    pack_topk_candidates,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=35, suite="nightly-amd-kernel-1-gpu", nightly=True)


def _reference(q, k, v, mapping, lens, cu=None, prefix=None):
    k, v = k.float(), v.float()
    output = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    bounds = list(range(q.shape[0] + 1)) if cu is None else cu.tolist()
    for request in range(len(bounds) - 1):
        for row in range(bounds[request], bounds[request + 1]):
            length = (
                int(lens[request])
                if cu is None
                else int(prefix[request]) + row - bounds[request] + 1
            )
            ids = mapping[request, :length]
            keys = k[ids].repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            values = v[ids].repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            logits = torch.einsum("hd,nhd->hn", q[row].float(), keys) * (
                0.5 * 1.25 / q.shape[-1] ** 0.5
            )
            output[row] = torch.einsum("hn,nhd->hd", logits.softmax(-1), values * 0.75)
            lse[row] = logits.logsumexp(-1)
    return output, lse


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestMiniMaxDcp(CustomTestCase):
    def test_fp8_owner_attention_and_live_graph_metadata(self):
        """Virtual page permutations and empty shards must preserve causal softmax."""
        dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
        torch.manual_seed(419)
        for prefill in (False, True):
            with self.subTest(prefill=prefill):
                mapping = torch.randperm(1536, device="cuda").reshape(3, 512)
                k = torch.randn(1536, 2, 128, device="cuda", dtype=torch.bfloat16).to(
                    dtype
                )
                v = torch.randn(1536, 2, 128, device="cuda", dtype=torch.bfloat16).to(
                    dtype
                )
                q = torch.randn(
                    5 if prefill else 3, 64, 128, device="cuda", dtype=torch.bfloat16
                )[:, ::2]
                cu = (
                    torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
                    if prefill
                    else None
                )
                prefix = (
                    torch.tensor([1, 128], device="cuda", dtype=torch.int32)
                    if prefill
                    else None
                )
                lens = torch.tensor(
                    [3, 131] if prefill else [1, 129, 385],
                    device="cuda",
                    dtype=torch.int32,
                )
                slots = torch.arange(lens.numel(), device="cuda", dtype=torch.int32)
                selected = (
                    torch.arange(4, device="cuda", dtype=torch.int32)
                    .expand(2, q.shape[0], 4)
                    .contiguous()
                )
                keys = [k[rank::8].contiguous() for rank in range(8)]
                values = [v[rank::8].contiguous() for rank in range(8)]

                def run():
                    partials, normalizers = [], []
                    for rank in range(8):
                        output, lse, _ = owner_attention(
                            q,
                            keys[rank],
                            values[rank],
                            None,
                            mapping,
                            slots,
                            lens,
                            128,
                            None,
                            0.5,
                            1.25,
                            0.75,
                            dcp_size=8,
                            dcp_rank=rank,
                            return_lse=True,
                            max_seqlen_q=3 if prefill else 1,
                            max_seqlen_k=512,
                            cu_seqlens=cu,
                            prefix_lens=prefix,
                            cu_seqblocks_q=cu,
                            topk_idx=selected,
                        )
                        partials.append(output)
                        normalizers.append(lse)
                    normalizers = torch.stack(normalizers)
                    lse = normalizers.logsumexp(0)
                    weights = (normalizers - lse).exp()
                    return (torch.stack(partials) * weights[..., None]).sum(0), lse

                actual = run()
                expected = _reference(q, k, v, mapping, lens, cu, prefix)
                torch.testing.assert_close(
                    actual[0], expected[0], atol=0.002, rtol=0.008
                )
                torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-5)
                if not prefill:
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = run()
                    q.mul_(0.75)
                    lens.copy_(
                        torch.tensor([7, 130, 511], device="cuda", dtype=torch.int32)
                    )
                    mapping.copy_(mapping.roll(1, 0))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = _reference(q, k, v, mapping, lens)
                    torch.testing.assert_close(
                        captured[0], expected[0], atol=0.002, rtol=0.008
                    )
                    torch.testing.assert_close(
                        captured[1], expected[1], atol=2e-5, rtol=2e-5
                    )

    def test_ragged_index_tiles_preserve_global_causal_scores(self):
        """Split score columns must cover partial query tiles and prefix blocks."""
        torch.manual_seed(827)
        q = torch.randn(386, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2048, 1, 128, device="cuda", dtype=torch.bfloat16)
        mapping = torch.randperm(2048, device="cuda").reshape(2, 1024)
        cu = torch.tensor([0, 129, 386], device="cuda", dtype=torch.int32)
        prefix = torch.tensor([127, 260], device="cuda", dtype=torch.int32)
        lens = torch.tensor([256, 517], device="cuda", dtype=torch.int32)
        slots = torch.arange(2, device="cuda", dtype=torch.int32)
        expected = torch.full((1, 386, 8), float("-inf"), device="cuda")
        positions = torch.arange(1024, device="cuda")
        for begin, end, cached, request in ((0, 129, 127, 0), (129, 386, 260, 1)):
            logits = q[begin:end, 0].float() @ k[mapping[request], 0].float().T
            logits *= 128**-0.5 * 1.4426950408889634
            causal = (
                positions[None, :]
                <= cached + torch.arange(end - begin, device="cuda")[:, None]
            )
            expected[0, begin:end] = (
                logits.masked_fill(~causal, float("-inf"))
                .reshape(end - begin, 8, 128)
                .amax(-1)
            )
        for world in (2, 8):
            with self.subTest(world=world):
                scores = [
                    owner_attention(
                        q,
                        k[rank::world].contiguous(),
                        None,
                        None,
                        mapping,
                        slots,
                        lens,
                        128,
                        None,
                        None,
                        None,
                        None,
                        dcp_size=world,
                        dcp_rank=rank,
                        return_lse=True,
                        max_seqlen_q=257,
                        max_seqlen_k=1024,
                        cu_seqlens=cu,
                        prefix_lens=prefix,
                        disable_index_value=True,
                    )[2]
                    for rank in range(world)
                ]
                torch.testing.assert_close(
                    torch.stack(scores).amax(0), expected, atol=3e-5, rtol=3e-5
                )

    def test_owner_empty_blocks_do_not_corrupt_prefill_topk(self):
        """Unowned blocks have -inf scores, not finite padding sentinels."""
        from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
            _topk_index_kernel,
        )

        scores = torch.full((1, 1, 300), float("-inf"), device="cuda")
        scores[0, 0, 17] = 5
        scores[0, 0, 258] = 4
        selected = torch.full((1, 1, 2), -1, device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        prefix = torch.tensor([299 * 128], device="cuda", dtype=torch.int32)
        _topk_index_kernel[(1, 1, 1)](
            scores,
            selected,
            1,
            128,
            cu,
            cu,
            prefix,
            2,
            0,
            0,
            *scores.stride(),
            *selected.stride(),
            MASK_INIT=False,
            MASK_LOCAL=False,
        )
        torch.testing.assert_close(
            selected,
            torch.tensor([[[17, 258]]], device="cuda", dtype=torch.int32),
            atol=0,
            rtol=0,
        )

    def test_candidate_union_preserves_global_max_topk(self):
        """Duplicate owner candidates must not consume slots or misorder negative scores."""
        torch.manual_seed(682)
        raw = -torch.rand(4, 2, 5, 32, device="cuda")
        raw[:, :, 2, 5] = 10
        raw[-1, :, 2, 7] = 11
        cu = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
        prefix = torch.tensor([0, 2051], device="cuda", dtype=torch.int32)
        valid_counts = [1, 1, 17, 17, 17]

        def reference(scores):
            ids = torch.full((2, 5, 4), -1, device="cuda", dtype=torch.int32)
            for row, count in enumerate(valid_counts):
                valid = scores[:, row, :count].clone()
                valid[:, 0] = 1e30
                valid[:, -1] = 1e29
                selected = valid.topk(min(4, count), dim=-1).indices.sort(-1).values
                ids[:, row, : min(4, count)] = selected.to(torch.int32)
            return ids

        packed = []
        for rank in range(4):
            local_ids = reference(raw[rank])
            packed.append(
                pack_topk_candidates(
                    raw[rank], local_ids, cu, cu, prefix, 1, 128, 1, 1, 3
                )
            )
        actual = merge_topk_candidates(
            torch.cat(packed, dim=0), torch.empty_like(local_ids), cu
        )
        torch.testing.assert_close(actual, reference(raw.max(0).values), atol=0, rtol=0)

    def test_dense_and_sparse_cache_locations_share_ownership(self):
        """Dense physical locations must not be collapsed a second time by the pool."""
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

        dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
        loc = torch.tensor([5, 9, 2, 6], device="cuda")
        k = (
            torch.arange(512, device="cuda", dtype=torch.float32)
            .reshape(4, 1, 128)
            .to(torch.bfloat16)
            / 64
        )
        mask = loc % 2 == 1
        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=2, attn_dcp_rank=1),
        ):
            pool = MiniMaxSparseKVPool(
                size=8,
                page_size=1,
                dtype=dtype,
                head_num=1,
                head_dim=128,
                idx_head_dim=128,
                dense_layer_ids=[0],
                sparse_layer_ids=[1],
                disable_value_sparse_layer_ids=[1],
                device="cuda",
                index_dtype=torch.bfloat16,
                start_layer=0,
                end_layer=2,
            )
            for layer in range(2):
                pool.get_key_buffer(layer).view(torch.uint8).zero_()
                pool.get_value_buffer(layer).view(torch.uint8).zero_()
            pool.get_index_k_buffer(1).zero_()
            expected = torch.zeros(9, 1, 128, device="cuda")
            expected_v = torch.zeros_like(expected)
            expected[loc[mask] // 2] = (k[mask] / 1.3).to(dtype).float()
            expected_v[loc[mask] // 2] = (-k[mask] / 0.7).to(dtype).float()
            pool.set_kv_buffer(
                SimpleNamespace(layer_id=0),
                loc // 2,
                k.clone(),
                -k,
                1.3,
                0.7,
                dcp_kv_mask=mask,
            )
            pool.set_fused_kv_index_buffer(
                SimpleNamespace(layer_id=1), loc, k, -k, k, None, 1.3, 0.7, 5.3
            )
            for layer in range(2):
                torch.testing.assert_close(
                    pool.get_key_buffer(layer).float(), expected, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    pool.get_value_buffer(layer).float(), expected_v, atol=0, rtol=0
                )
            expected[loc[mask] // 2] = k[mask].float()
            torch.testing.assert_close(
                pool.get_index_k_buffer(1).float(), expected, atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
