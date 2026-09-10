"""DSpark window-only metadata must preserve absolute-position attention on replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import PretrainedConfig

import sglang.srt.layers.attention.triton_backend as triton_backend
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.dflash import _get_dflash_layer_attention_params
from sglang.srt.speculative.dflash_utils import get_dflash_attention_sliding_window_size
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA or ROCm GPU")
class TestDSparkWindowMetadata(CustomTestCase):
    def setUp(self):
        self.device = "cuda"
        self.bs, self.gamma, self.context = 2, 8, 8216
        self.q_heads, self.kv_heads, self.dim = 16, 4, 128
        generator = torch.Generator(device=self.device).manual_seed(1024)
        self.req_to_token = (
            torch.randperm(
                self.bs * self.context, device=self.device, generator=generator
            )
            .reshape(self.bs, self.context)
            .to(torch.int32)
        )
        shape = (self.bs * self.context, self.kv_heads, self.dim)
        self.cache_k = torch.randn(
            shape, device=self.device, dtype=torch.bfloat16, generator=generator
        )
        self.cache_v = torch.randn(
            shape, device=self.device, dtype=torch.bfloat16, generator=generator
        )
        self.q = torch.randn(
            self.bs * self.gamma,
            self.q_heads * self.dim,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        self.k = torch.randn(
            self.bs * self.gamma,
            self.kv_heads,
            self.dim,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        # Future proposal rows must be visible for the exported noncausal draft.
        self.v = (
            torch.randn(
                self.k.shape,
                device=self.device,
                dtype=self.k.dtype,
                generator=generator,
            )
            + 16
        )
        self.pool = SimpleNamespace(
            start_layer=0,
            get_key_buffer=lambda layer_id: self.cache_k,
            get_value_buffer=lambda layer_id: self.cache_v,
        )
        self.req_pool = SimpleNamespace(size=self.bs, req_to_token=self.req_to_token)
        self.allocator = SimpleNamespace()
        self.translator = KVIndexTranslator(
            req_to_token=self.req_to_token,
            token_to_kv_pool_allocator=self.allocator,
            token_to_kv_pool=self.pool,
            page_size=1,
            device=self.device,
        )

    def _backend(self, config, *, draft=True):
        model_config = SimpleNamespace(
            hf_config=config,
            hf_text_config=config,
            attention_arch=AttentionArch.MHA,
            is_draft_model=draft,
            is_encoder_decoder=False,
            v_head_dim=self.dim,
            swa_v_head_dim=self.dim,
            head_dim=self.dim,
            context_len=self.context,
            linear_attn_registry_result=None,
            get_max_num_attention_heads=lambda: self.q_heads,
            get_num_kv_heads=lambda *args: self.kv_heads,
        )
        runner = SimpleNamespace(
            model_config=model_config,
            is_draft_worker=draft,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            sliding_window_size=get_dflash_attention_sliding_window_size(config),
            req_to_token_pool=self.req_pool,
            token_to_kv_pool=self.pool,
            token_to_kv_pool_allocator=self.allocator,
            kv_index_translator=self.translator,
            page_size=1,
            device=self.device,
            gpu_id=torch.cuda.current_device(),
            server_args=SimpleNamespace(
                enable_lean_attention=False,
                triton_attention_split_tile_size=None,
            ),
        )
        spec = SimpleNamespace(
            speculative_num_draft_tokens=self.gamma + 1,
            speculative_num_steps=1,
            speculative_eagle_topk=0,
        )
        parallel = SimpleNamespace(attn_tp_size=1, attn_dcp_size=1)
        execution = SimpleNamespace(
            kernel=SimpleNamespace(triton_attention_num_kv_splits=1),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        )
        with (
            patch.object(triton_backend, "get_spec", return_value=spec),
            patch.object(triton_backend, "get_parallel", return_value=parallel),
            patch.object(triton_backend, "get_exec", return_value=execution),
            patch.object(
                triton_backend, "cuda_graph_fully_disabled", return_value=True
            ),
            patch.object(
                triton_backend,
                "get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=-1),
            ),
        ):
            backend = triton_backend.TritonAttnBackend(runner)
        backend.init_cuda_graph_state(self.bs, self.bs * self.gamma)
        return backend

    def _layer(self, config, layer_id):
        window, attn_type = _get_dflash_layer_attention_params(config, layer_id)
        return SimpleNamespace(
            layer_id=layer_id,
            tp_q_head_num=self.q_heads,
            tp_k_head_num=self.kv_heads,
            tp_v_head_num=self.kv_heads,
            qk_head_dim=self.dim,
            v_head_dim=self.dim,
            scaling=self.dim**-0.5,
            sliding_window_size=window,
            attn_type=attn_type,
            is_cross_attention=False,
            k_scale=None,
            v_scale=None,
            logit_cap=0,
            logit_capping_method="tanh",
            xai_temperature_len=-1,
        )

    def _batch(self):
        return SimpleNamespace(
            batch_size=self.bs,
            req_pool_indices=torch.arange(self.bs, device=self.device),
            seq_lens=torch.full(
                (self.bs,), 1023, dtype=torch.int32, device=self.device
            ),
            seq_lens_sum=None,
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=self.gamma, custom_mask=None),
            out_cache_loc=None,
            encoder_lens=None,
            extend_prefix_lens=None,
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
        )

    def _expected(self, batch, layer, *, custom_mask, causal):
        outputs = []
        for seq, prefix in enumerate(batch.seq_lens.tolist()):
            req = int(batch.req_pool_indices[seq])
            start = seq * self.gamma
            slots = self.req_to_token[req, :prefix].long()
            keys = torch.cat((self.cache_k[slots], self.k[start : start + self.gamma]))
            values = torch.cat(
                (self.cache_v[slots], self.v[start : start + self.gamma])
            )
            keys = keys.repeat_interleave(self.q_heads // self.kv_heads, dim=1).float()
            values = values.repeat_interleave(
                self.q_heads // self.kv_heads, dim=1
            ).float()
            query = self.q[start : start + self.gamma].view(
                self.gamma, self.q_heads, self.dim
            )
            scores = torch.einsum("qhd,khd->hqk", query.float(), keys) * layer.scaling
            q_pos = prefix + torch.arange(self.gamma, device=self.device)[:, None]
            k_pos = torch.arange(prefix + self.gamma, device=self.device)[None, :]
            allowed = torch.ones_like(q_pos + k_pos, dtype=torch.bool)
            if layer.sliding_window_size >= 0:
                allowed &= k_pos >= q_pos - layer.sliding_window_size
            # Expected causality comes from the export/override, not the resolver.
            if causal:
                allowed &= k_pos <= q_pos
            if custom_mask:
                allowed[:, prefix + 1 :: 2] = False
            scores.masked_fill_(~allowed[None], -torch.inf)
            outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
        return torch.cat(outputs).reshape_as(self.q)

    def _check_attention(
        self, layer_types, *, draft=True, custom_mask=False, causal=False
    ):
        config = PretrainedConfig(
            architectures=["Qwen3DSparkModel"],
            num_hidden_layers=len(layer_types),
            layer_types=layer_types,
            sliding_window=1024,
            dflash_config={"causal": False},
        )
        if causal:
            config.is_causal = True
        backend = self._backend(config, draft=draft)
        reference = self._backend(config, draft=draft)
        # Exercise the former full-prefix + window construction as a bitwise baseline.
        reference._target_verify_window_only = False
        batch = self._batch()
        layers = [self._layer(config, i) for i in range(len(layer_types))]
        graph = torch.cuda.CUDAGraph()

        def forward():
            return [
                backend.forward_extend(
                    self.q, self.k, self.v, layer, batch, save_kv_cache=False
                )
                for layer in layers
            ]

        def update(prefix, swap=False):
            lengths = [prefix, max(1, prefix - 31)]
            batch.seq_lens.copy_(
                torch.tensor(lengths, device=self.device, dtype=torch.int32)
            )
            batch.req_pool_indices.copy_(
                torch.tensor([1, 0] if swap else [0, 1], device=self.device)
            )
            if custom_mask:
                masks = []
                for length in lengths:
                    mask = torch.ones(
                        (self.gamma, length + self.gamma),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    mask[:, length + 1 :: 2] = False
                    masks.append(mask.flatten())
                batch.spec_info.custom_mask = torch.cat(masks)

        update(1023)
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            graph_outputs = forward()

        for prefix in (1023, 1024, 1025, 8192, 8195):
            with self.subTest(
                prefix=prefix, layers=layer_types, draft=draft, custom_mask=custom_mask
            ):
                update(prefix, swap=prefix == 8195)
                backend.init_forward_metadata_out_graph(batch)
                graph.replay()
                replay_outputs = [out.clone() for out in graph_outputs]
                backend.init_forward_metadata(batch)
                eager_outputs = forward()
                reference.init_forward_metadata(batch)
                for layer, eager, replay in zip(layers, eager_outputs, replay_outputs):
                    baseline = reference.forward_extend(
                        self.q, self.k, self.v, layer, batch, save_kv_cache=False
                    )
                    torch.testing.assert_close(eager, baseline, atol=0, rtol=0)
                    torch.testing.assert_close(replay, baseline, atol=0, rtol=0)
                    torch.testing.assert_close(
                        eager.float(),
                        self._expected(
                            batch, layer, custom_mask=custom_mask, causal=causal
                        ),
                        atol=2e-2,
                        rtol=2e-2,
                    )
                # Eager metadata replaced the capture views; restore them before replay.
                backend.forward_metadata = backend._build_cuda_graph_forward_metadata(
                    self.bs, batch.forward_mode, batch.spec_info
                )

    def test_all_sliding_draft_replay(self):
        self._check_attention(["sliding_attention"] * 6)

    def test_explicit_causal_override_replay(self):
        self._check_attention(["sliding_attention"], causal=True)

    def test_mixed_and_full_draft_fallback(self):
        for layer_types in (
            ["sliding_attention", "full_attention"],
            ["full_attention"],
        ):
            self._check_attention(layer_types)

    def test_target_custom_mask_fallback(self):
        self._check_attention(
            ["sliding_attention", "full_attention"], draft=False, custom_mask=True
        )


if __name__ == "__main__":
    unittest.main()
