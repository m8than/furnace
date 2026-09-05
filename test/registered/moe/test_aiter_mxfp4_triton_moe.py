# SPDX-License-Identifier: Apache-2.0
"""Numerical regressions for the gfx942 MXFP4 route from PR #35525.

Compare packed checkpoint bytes against an independent FP32 dequantization
reference, with the BF16 GEMM/activation boundaries of the A16W4 contract.
"""

import os
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")


def _dequant_mxfp4(weight, scale):
    # Low nibble is the even K element; E8M0 is an unsigned exponent.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=weight.device,
    )
    unpacked = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
    return lut[unpacked.long()] * torch.exp2(scale.float() - 127).repeat_interleave(
        32, dim=-1
    )


def _reference(
    x,
    w13,
    w2,
    s13,
    s2,
    routing_weights,
    ids,
    activation,
    beta,
    linear_beta,
    rank,
    num_global,
    weight_on_input,
):
    experts = w13.shape[0]
    inter = w13.shape[1] // 2
    out = torch.zeros_like(x, dtype=torch.float32)
    offset = rank * experts if num_global != experts else 0
    # Only dequantize experts actually referenced; the E=896 shape test must
    # not expand the complete checkpoint to FP32 to compute a reference.
    for global_id in ids.unique().tolist():
        local = global_id - offset
        if global_id < 0 or not 0 <= local < experts:
            continue
        tokens, slots = torch.where(ids == global_id)
        route = routing_weights[tokens, slots, None]
        gu = x[tokens].float() @ _dequant_mxfp4(w13[local], s13[local]).T
        if weight_on_input:
            gu *= route
        gate, up = gu.to(torch.bfloat16).float().split(inter, dim=-1)
        if activation == "situ":
            gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            up = linear_beta * torch.tanh(up / linear_beta)
            activated = gate * up
        else:
            activated = torch.nn.functional.silu(gate) * up
        down = (
            activated.to(torch.bfloat16).float()
            @ _dequant_mxfp4(w2[local], s2[local]).T
        )
        if not weight_on_input:
            down *= route
        out.index_add_(0, tokens, down.to(torch.bfloat16).float())
    return out.to(torch.bfloat16)


def _is_gfx942():
    return bool(
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"
    )


@unittest.skipUnless(_is_gfx942(), "gfx942 GPU required")
class TestAiterMxfp4TritonMoE(CustomTestCase):
    def _run(
        self,
        tokens=16,
        activation="situ",
        beta=4.0,
        linear_beta=25.0,
        rank=0,
        ep_size=1,
        weight_on_input=False,
        hidden=256,
        inter=128,
        experts=8,
        padded_inter=None,
        sentinel=False,
    ):
        from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import (
            fused_moe_mxfp4_triton,
        )

        generator = torch.Generator(device="cuda").manual_seed(13)
        padded_inter = padded_inter or inter
        num_global = experts * ep_size
        ids = torch.randint(
            num_global,
            (tokens, 2),
            device="cuda",
            dtype=torch.int32,
            generator=generator,
        )
        # Exercise the first and last expert offsets, including E=896.
        ids[0, 0], ids[0, 1] = 0, num_global - 1
        if sentinel:
            ids[0] = -1
        route = torch.rand(tokens, 2, device="cuda", generator=generator)
        route /= route.sum(dim=-1, keepdim=True)
        x = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        w13 = torch.zeros(
            experts, 2 * padded_inter, hidden // 2, dtype=torch.uint8, device="cuda"
        )
        w2 = torch.zeros(
            experts, hidden, padded_inter // 2, dtype=torch.uint8, device="cuda"
        )
        s13 = torch.full(
            (experts, 2 * padded_inter, hidden // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        )
        s2 = torch.full(
            (experts, hidden, padded_inter // 32), 127, dtype=torch.uint8, device="cuda"
        )
        for global_id in ids.unique().tolist():
            local = global_id - rank * experts
            if global_id < 0 or not 0 <= local < experts:
                continue
            for start in (0, padded_inter):
                w13[local, start : start + inter] = torch.randint(
                    256,
                    (inter, hidden // 2),
                    device="cuda",
                    dtype=torch.uint8,
                    generator=generator,
                )
                s13[local, start : start + inter] = torch.randint(
                    124,
                    128,
                    (inter, hidden // 32),
                    device="cuda",
                    dtype=torch.uint8,
                    generator=generator,
                )
            w2[local, :, : inter // 2] = torch.randint(
                256,
                (hidden, inter // 2),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            )
            s2[local, :, : inter // 32] = torch.randint(
                124,
                128,
                (hidden, inter // 32),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            )
        result = fused_moe_mxfp4_triton(
            x,
            w13,
            w2,
            s13,
            s2,
            route,
            ids,
            activation=activation,
            situ_beta=beta,
            situ_linear_beta=linear_beta,
            num_global_experts=num_global,
            ep_rank=rank,
            apply_router_weight_on_input=weight_on_input,
        )
        reference = _reference(
            x,
            w13,
            w2,
            s13,
            s2,
            route,
            ids,
            activation,
            beta,
            linear_beta,
            rank,
            num_global,
            weight_on_input,
        )
        self.assertEqual(result.shape, reference.shape)
        self.assertTrue(torch.isfinite(result).all().item())
        error = torch.linalg.vector_norm(result.float() - reference.float())
        norm = torch.linalg.vector_norm(reference.float())
        self.assertLess((error / norm.clamp_min(1e-8)).item(), 1e-2)
        if sentinel:
            self.assertEqual(torch.count_nonzero(result[0]).item(), 0)

    def test_situ_and_silu_both_tile_sizes(self):
        for tokens in (1, 512):
            for activation in ("situ", "silu"):
                with self.subTest(tokens=tokens, activation=activation):
                    self._run(tokens=tokens, activation=activation)

    def test_nondefault_situ_parameters(self):
        self._run(beta=2.5, linear_beta=9.0)

    def test_router_weight_before_activation(self):
        self._run(weight_on_input=True)

    def test_ep_unowned_and_invalid_experts(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self._run(rank=rank, ep_size=2, sentinel=True)

    def test_kimi_tp8_dimensions_and_expert_offsets(self):
        self._run(tokens=1, hidden=3584, inter=384, experts=896)

    def test_separated_gate_up_padding(self):
        # AITER's default I384 -> I512 padding must not move the up half.
        self._run(inter=384, padded_inter=512)

    def test_single_token_direct_routing_is_bitwise_equal_to_grouped(self):
        from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import _moe_config
        from sglang.srt.layers.moe.moe_runner.mxfp4_situ_fused import (
            fused_moe_mxfp4_act,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )

        generator = torch.Generator(device="cuda").manual_seed(42)
        # Unsorted IDs, a duplicate expert, and an unowned route must retain
        # their original output row identities without an expert-sort buffer.
        ids = torch.tensor(
            [[7, 1, -1, 0, 5, 1, 2, 6]], device="cuda", dtype=torch.int32
        )
        route = torch.rand(1, 8, device="cuda", generator=generator)
        valid = ids.flatten() >= 0
        config = _moe_config(1)
        routing = moe_align_block_size(ids, 16, 8, ignore_invalid_expert=True)
        for activation, n, k, topk in (("situ", 256, 256, 8), ("none", 256, 128, 1)):
            with self.subTest(activation=activation):
                x = torch.randn(
                    8 // topk,
                    k,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                weight = torch.randint(
                    256,
                    (8, n, k // 2),
                    device="cuda",
                    dtype=torch.uint8,
                    generator=generator,
                )
                scale = torch.randint(
                    120,
                    125,
                    (8, n, k // 32),
                    device="cuda",
                    dtype=torch.uint8,
                    generator=generator,
                )
                width = n // 2 if activation == "situ" else n
                grouped = torch.full(
                    (8, width), float("nan"), device="cuda", dtype=torch.bfloat16
                )
                direct = torch.empty_like(grouped)
                fused_moe_mxfp4_act(
                    x,
                    weight,
                    grouped,
                    scale,
                    route,
                    ids,
                    *routing,
                    True,
                    topk,
                    config,
                    activation=activation,
                )
                fused_moe_mxfp4_act(
                    x,
                    weight,
                    direct,
                    scale,
                    route,
                    ids,
                    None,
                    None,
                    None,
                    True,
                    topk,
                    config,
                    activation=activation,
                )
                self.assertTrue(
                    torch.equal(
                        direct[valid].view(torch.int16),
                        grouped[valid].view(torch.int16),
                    )
                )
                self.assertEqual(torch.count_nonzero(direct[~valid]).item(), 0)


class TestTritonMxfp4Gate(CustomTestCase):
    def test_environment_cannot_enable_other_architectures(self):
        from sglang.srt.layers.moe.moe_runner import aiter_mxfp4_triton as mod

        self.addCleanup(mod.use_triton_mxfp4_moe.cache_clear)
        for arch in ("gfx950", "", "gfx1250"):
            with (
                mock.patch.dict(os.environ, {"SGLANG_AITER_MXFP4_TRITON": "1"}),
                mock.patch.object(mod, "_arch", return_value=arch),
            ):
                mod.use_triton_mxfp4_moe.cache_clear()
                self.assertFalse(mod.use_triton_mxfp4_moe())

    def test_environment_can_disable_gfx942(self):
        from sglang.srt.layers.moe.moe_runner import aiter_mxfp4_triton as mod

        self.addCleanup(mod.use_triton_mxfp4_moe.cache_clear)
        with (
            mock.patch.dict(os.environ, {"SGLANG_AITER_MXFP4_TRITON": "0"}),
            mock.patch.object(mod, "_arch", return_value="gfx942"),
        ):
            mod.use_triton_mxfp4_moe.cache_clear()
            self.assertFalse(mod.use_triton_mxfp4_moe())


if __name__ == "__main__":
    unittest.main()
