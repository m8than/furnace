"""FP8 projection tiling must preserve BF16 results, including cancellation.

Dropping AITER's MFMA/K-packing settings while shrinking tiles changed a few
outputs by one BF16 ULP. Gaussian-only inputs missed this; finite FP8 values
covering the exponent range expose it without loading model weights.
"""

import unittest

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


@unittest.skipUnless(
    fp8_utils._use_aiter and fp8_utils._is_fp8_fnuz,
    "requires gfx94 and AITER",
)
class TestDenseFP8Rounding(CustomTestCase):
    @torch.no_grad()
    def test_projection_tiles_match_vendor_accumulation(self):
        torch.manual_seed(9127)
        # Cover verification and prefill, including partial tiles and both K widths.
        for m, n, k in (
            (32, 2560, 6144),
            (9, 4992, 6144),
            (9, 6144, 4096),
            (8192, 6144, 4096),
            (8193, 1536, 6144),
        ):
            bits = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.uint8)
            bits[bits == 128] = 0  # FNUZ's sole NaN encoding.
            weight = bits.view(torch.float8_e4m3fnuz)
            scales = torch.rand(n // 128, k // 128, device="cuda") * 0.001 + 0.0005
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            q, qs = fp8_utils.aiter_per1x128_quant(
                x, quant_dtype=fp8_utils.aiter.dtypes.fp8
            )
            bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            expected = fp8_utils.triton_gemm_a8w8_blockscale(
                q, weight, qs, scales, dtype=torch.bfloat16
            )
            expected += bias

            for prequantized in (False, True):
                with self.subTest(m=m, n=n, k=k, prequantized=prequantized):

                    def run():
                        return fp8_utils.aiter_w8a8_block_fp8_linear(
                            q if prequantized else x,
                            weight,
                            [128, 128],
                            scales,
                            input_scale=qs if prequantized else None,
                            bias=bias,
                        )

                    actual = run()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = run()
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
                    graph.reset()


if __name__ == "__main__":
    unittest.main()
