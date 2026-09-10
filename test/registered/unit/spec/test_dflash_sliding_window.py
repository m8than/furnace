import unittest

from transformers import Qwen3Config

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.dflash import _get_dflash_layer_attention_params
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashSlidingWindow(CustomTestCase):
    def test_qwen_draft_retains_trained_window_after_hf_normalization(self):
        """Exported draft windows survive Qwen3 clearing the top-level window."""
        config = Qwen3Config(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            sliding_window=1024,
            dflash_config={"use_swa": True, "swa_window_size": 1024},
        )
        self.assertEqual(get_dflash_attention_sliding_window_size(config), 1023)

        # A surviving explicit attention window remains authoritative.
        config.sliding_window = 2048
        self.assertEqual(get_dflash_attention_sliding_window_size(config), 2047)

    def test_exported_noncausal_draft_overrides_sliding_layer_default(self):
        """Exported bidirectional drafts must not become causal sliding layers."""
        config = Qwen3Config(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            dflash_config={"causal": False, "swa_window_size": 1024},
        )
        self.assertEqual(
            _get_dflash_layer_attention_params(config, layer_id=0),
            (1023, AttentionType.ENCODER_ONLY),
        )

    def test_explicit_causality_overrides_conflicting_export(self):
        """Both explicit causal and noncausal overrides supersede the export."""
        for is_causal, expected_type in (
            (True, AttentionType.DECODER),
            (False, AttentionType.ENCODER_ONLY),
        ):
            with self.subTest(is_causal=is_causal):
                config = Qwen3Config(
                    num_hidden_layers=1,
                    layer_types=["sliding_attention"],
                    is_causal=is_causal,
                    dflash_config={
                        "causal": not is_causal,
                        "swa_window_size": 1024,
                    },
                )
                self.assertEqual(
                    _get_dflash_layer_attention_params(config, layer_id=0),
                    (1023, expected_type),
                )

    def test_missing_causality_preserves_mixed_layer_defaults(self):
        config = Qwen3Config(
            num_hidden_layers=2,
            layer_types=["full_attention", "sliding_attention"],
            dflash_config={"swa_window_size": 1024},
        )
        self.assertEqual(
            _get_dflash_layer_attention_params(config, layer_id=0),
            (-1, AttentionType.ENCODER_ONLY),
        )
        self.assertEqual(
            _get_dflash_layer_attention_params(config, layer_id=1),
            (1023, AttentionType.DECODER),
        )


if __name__ == "__main__":
    unittest.main()
