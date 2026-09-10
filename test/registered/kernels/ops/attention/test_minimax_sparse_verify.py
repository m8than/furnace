"""MiniMax linear verification must match sequential sparse causal attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx as indexer
from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxHybridAttnBackend,
    MiniMaxSparseAttnBackend,
)
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_prefill,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


class _SerialBlockScores:
    """Run the real score kernel without partitioning its independent KV blocks."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs["NUM_K_SPLITS"] = 1
            return self.kernel[lambda meta: (*grid(meta)[:2], 1)](*args, **kwargs)

        return launch


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestMiniMaxSparseVerify(CustomTestCase):
    def test_graph_verify_matches_sequential_sparse_attention(self):
        """Replay grows past capture and sparse-block boundaries, with FP8 main KV."""
        torch.manual_seed(419)
        dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
        num_draft_tokens = 9  # DSpark's eight draft queries plus the anchor.
        context_len = 32768
        q = torch.randn(num_draft_tokens, 4, 128, device="cuda", dtype=torch.bfloat16)
        idx_q = torch.randn(
            num_draft_tokens, 2, 128, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(context_len, 1, 128, device="cuda", dtype=torch.bfloat16).to(
            dtype
        )
        v = torch.randn(context_len, 1, 128, device="cuda", dtype=torch.bfloat16).to(
            dtype
        )
        idx_k = torch.randn(context_len, 1, 128, device="cuda", dtype=torch.bfloat16)
        req_to_token = torch.randperm(context_len, device="cuda").view(1, -1)
        slots = torch.zeros(1, device="cuda", dtype=torch.int32)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.ones(1, device="cuda", dtype=torch.int32),
            seq_lens_cpu=None,
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            extend_prefix_lens=None,
            spec_info=DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=num_draft_tokens,
            ),
        )
        # Metadata does not need a model runner or distributed/KV-pool setup.
        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend.max_context_len = context_len
        backend._msa_owns_decode = False

        # Match the target graph runner's mask negotiation for the hybrid
        # backend; its built-in causal path must not receive a tree mask.
        hybrid = MiniMaxHybridAttnBackend.__new__(MiniMaxHybridAttnBackend)
        _, build_custom_mask = resolve_dflash_verify_mask_policy(hybrid)
        if build_custom_mask:
            batch.spec_info.custom_mask = torch.ones(
                num_draft_tokens * context_len, device="cuda", dtype=torch.bool
            )

        def sparse(query, index_query, cu, seq, prefix, max_q, max_k):
            return minimax_sparse_prefill(
                q=query,
                k_cache=k,
                v_cache=v,
                sink=None,
                idx_q=index_query,
                idx_k_cache=idx_k,
                idx_v_cache=None,
                idx_sink=None,
                req_to_token=req_to_token,
                slot_ids=slots,
                cu_seqlens=cu,
                seq_lens=seq,
                prefix_lens=prefix,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                block_size_q=1,
                block_size_k=128,
                topk=2,
                init_blocks=0,
                local_blocks=1,
                disable_index_value=True,
                cu_seqblocks_q=cu,
                max_seqblock_q=max_q,
                all_seqblock_q=query.shape[0],
            )[1]

        def verify():
            backend.init_forward_metadata_in_graph(batch)
            cu, seq, prefix = backend._resolve_extend_meta(batch, q)
            return sparse(
                q,
                idx_q,
                cu,
                seq,
                prefix,
                backend._max_seqlen_q,
                backend._max_seqlen_k,
            )

        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            verify()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = verify()

        single_cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        # Cross sparse-block and KV-partition boundaries after graph capture.
        for prefix_len in (127, 8191, 32759):
            with self.subTest(prefix_len=prefix_len):
                batch.seq_lens.fill_(prefix_len)
                backend.init_forward_metadata_out_graph(batch)
                graph.replay()
                with patch.object(
                    indexer,
                    "_flash_attn_fwd_with_block_score_kernel",
                    _SerialBlockScores(indexer._flash_attn_fwd_with_block_score_kernel),
                ):
                    serial_output = verify()
                torch.testing.assert_close(output, serial_output, atol=0, rtol=0)
                expected = []
                for row in range(num_draft_tokens):
                    prefix = torch.tensor(
                        [prefix_len + row], device="cuda", dtype=torch.int32
                    )
                    expected.append(
                        sparse(
                            q[row : row + 1],
                            idx_q[row : row + 1],
                            single_cu,
                            prefix + 1,
                            prefix,
                            1,
                            context_len,
                        )
                    )
                torch.testing.assert_close(
                    output, torch.cat(expected), atol=2e-2, rtol=2e-2
                )


if __name__ == "__main__":
    unittest.main()
