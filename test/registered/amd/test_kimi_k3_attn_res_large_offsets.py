"""Residual-bank addressing must remain correct beyond signed 32-bit offsets."""

import unittest

import torch

from sglang.kernels.ops.kimi_k3.attn_res_hip import attn_res_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires ROCm")
class TestAttnResidualLargeOffsets(CustomTestCase):
    def test_large_bank_offsets_match_compact_bank(self):
        # Touch only small logical rows. Either token 2 or snapshot 2 crosses
        # 2**31 elements while each individual stride still fits signed int32.
        hidden, stride = 128, 1 << 30
        size = 2 * stride + 2 * hidden
        if torch.cuda.mem_get_info()[0] < size * 2 + (1 << 30):
            self.skipTest("requires 5 GiB free for the sparse-stride allocation")
        generator = torch.Generator(device="cuda").manual_seed(49043)
        backing = torch.empty(size, device="cuda", dtype=torch.bfloat16)
        cw = torch.randn(
            hidden, device="cuda", dtype=torch.float32, generator=generator
        )
        ow = torch.randn(
            hidden, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        cases = (
            (3, 2, (stride, hidden, 1), 1, False),
            (3, 2, (stride, hidden, 1), 1, True),
            (2, 3, (hidden, stride, 1), 3, False),
            (2, 3, (hidden, stride, 1), 2, True),
        )
        for tokens, rows, strides, nvb, write in cases:
            with self.subTest(strides=strides, nvb=nvb, write_prefix=write):
                prefix = torch.randn(
                    tokens,
                    hidden,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                addend = torch.randn(
                    tokens,
                    hidden,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                compact = torch.randn(
                    tokens,
                    rows,
                    hidden,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                bank = backing.as_strided((tokens, rows, hidden), strides)
                bank.copy_(compact)
                expected_bank = compact.clone()
                expected, actual = torch.empty_like(prefix), torch.empty_like(prefix)
                expected_prefix, actual_prefix = (
                    torch.empty_like(prefix),
                    torch.empty_like(prefix),
                )
                attn_res_hip(
                    prefix,
                    expected_bank,
                    cw,
                    ow,
                    expected,
                    nvb,
                    1e-5,
                    1e-5,
                    addend=addend,
                    prefix_out=expected_prefix,
                    write_prefix=write,
                )
                attn_res_hip(
                    prefix,
                    bank,
                    cw,
                    ow,
                    actual,
                    nvb,
                    1e-5,
                    1e-5,
                    addend=addend,
                    prefix_out=actual_prefix,
                    write_prefix=write,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(actual_prefix, expected_prefix))
                self.assertTrue(torch.equal(bank, expected_bank))


if __name__ == "__main__":
    unittest.main()
