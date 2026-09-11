"""GLM pipeline receivers must accept the graph allocator's mHC payload."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner_utils.buffers import _allocate_pp_proxy_tensors
from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ResidualConsumer(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states + residual, None


class TestGlm5NextPipeline(CustomTestCase):
    def receiver(self, mhc):
        model = Glm5NextModel.__new__(Glm5NextModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(mhc=mhc, hc_mult=4 if mhc else 1)
        model.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)
        model.start_layer = model.end_layer = 0
        model.first_k_dense_replace = 3
        model.dflash_capture = False
        model.enable_a2a_moe = False
        model.layers_to_capture = []
        model.norm = _ResidualConsumer()
        return model

    def forward_proxy(self, model, tensors):
        tokens = tensors["hidden_states"].shape[0]
        return model(
            torch.zeros(tokens, dtype=torch.int64),
            torch.arange(tokens),
            SimpleNamespace(can_run_tbo=False, forward_mode=ForwardMode.DECODE),
            pp_proxy_tensors=PPProxyTensors(tensors),
        )

    def test_mhc_graph_payload_preserves_folded_hidden_state(self):
        tensors = _allocate_pp_proxy_tensors(
            max_num_tokens=2,
            max_hidden_tokens=2,
            hidden_size=3,
            hc_hidden_size=12,
            dtype=torch.float32,
        )
        expected = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        tensors["hidden_states"].copy_(expected)
        actual = self.forward_proxy(self.receiver(True), tensors)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mhc_pipeline_output_supports_graph_replay_slicing(self):
        middle = self.receiver(True)
        middle.pp_group.is_last_rank = False
        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        proxy = self.forward_proxy(middle, {"hidden_states": hidden})
        actual = self.forward_proxy(self.receiver(True), proxy[:1].tensors)
        torch.testing.assert_close(actual, hidden[:1], rtol=0, atol=0)

    def test_plain_pipeline_preserves_separate_residual(self):
        hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        residual = torch.full_like(hidden, 7)
        actual = self.forward_proxy(
            self.receiver(False), {"hidden_states": hidden, "residual": residual}
        )
        torch.testing.assert_close(actual, hidden + residual, rtol=0, atol=0)

    def test_plain_pipeline_rejects_missing_residual(self):
        with self.assertRaises(KeyError):
            self.forward_proxy(
                self.receiver(False), {"hidden_states": torch.zeros(2, 3)}
            )


if __name__ == "__main__":
    unittest.main()
