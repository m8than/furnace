import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_glm5_next_dflash_contracts_mhc_hidden_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = True

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    residual = torch.full_like(hidden_states, 2)

    actual = model._prepare_aux_hidden_state(hidden_states, residual)
    expected = (hidden_states + residual).unflatten(-1, (4, -1)).mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3)


def test_glm5_next_eagle_capture_keeps_mhc_hidden_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = False

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    residual = torch.full_like(hidden_states, 2)

    actual = model._prepare_aux_hidden_state(hidden_states, residual)

    torch.testing.assert_close(actual, hidden_states + residual)


def test_glm5_next_dflash_contracts_mhc_hidden_state_without_residual():
    # GLM-5.3-Flash runs with mhc=True, where MHCLayerCommunicator folds the
    # residual into the widened hidden state and returns residual=None.
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = True

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    actual = model._prepare_aux_hidden_state(hidden_states, None)
    expected = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3)


def test_glm5_next_eagle_capture_without_residual():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=False, hc_mult=1)
    model.dflash_capture = False

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    actual = model._prepare_aux_hidden_state(hidden_states, None)

    torch.testing.assert_close(actual, hidden_states)


def test_glm5_next_dflash_maps_target_layers_to_capture_points():
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.model = SimpleNamespace(dflash_capture=False, layers_to_capture=[])
    model.capture_aux_hidden_states = False

    model.set_dflash_layers_to_capture([5, 14, 24, 33, 42])

    assert model.capture_aux_hidden_states
    assert model.model.dflash_capture
    assert model.model.layers_to_capture == [6, 15, 25, 34, 43]


@pytest.mark.parametrize("use_norm", [False, True])
def test_glm5_next_rocm_mhc_pre_post(use_norm):
    """ROCm must implement the mHC math without CUDA-only backend defaults."""
    from sglang.srt.environ import envs
    from sglang.srt.models.glm5_next import Glm5NextDecoderLayer
    from sglang.srt.runtime_context import override_platform

    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        mhc=True, hc_mult=4, rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=3
    )
    generator = torch.Generator().manual_seed(42)
    residual = torch.randn(3, 4, 8, generator=generator).bfloat16()
    fn = torch.randn(24, 32, generator=generator) * 0.01
    scale = torch.tensor([0.5, 0.25, 0.1])
    base = torch.randn(24, generator=generator)
    norm_weight = torch.arange(1, 9).bfloat16() if use_norm else None
    flat = residual.flatten(1).float()
    mixes = (flat @ fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = (mixes[:, 8:] * scale[2] + base[8:]).view(3, 4, 4).softmax(-1) + 1e-6
    comb /= comb.sum(-2, keepdim=True) + 1e-6
    for _ in range(2):
        comb /= comb.sum(-1, keepdim=True) + 1e-6
        comb /= comb.sum(-2, keepdim=True) + 1e-6
    expected_input = (pre[:, :, None] * residual.float()).sum(1).bfloat16()

    with (
        override_platform(is_hip=True),
        envs.SGLANG_USE_AITER.override(True),
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
        envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
        envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
    ):
        layer_input, h_res, h_post, norm_fused = layer._hc_pre(
            fn, scale, base, residual.flatten(1), norm_weight, 1e-6
        )
        # The portable path leaves output RMSNorm to MHCLayerCommunicator.
        assert not norm_fused
        torch.testing.assert_close(layer_input, expected_input)
        torch.testing.assert_close(h_post, post)
        torch.testing.assert_close(h_res.view(3, 4, 4), comb)
        actual = layer.hc_post(layer_input, residual.flatten(1), h_res, h_post)
    expected = (
        post[:, :, None] * expected_input[:, None, :].float()
        + torch.einsum("bij,bih->bjh", comb, residual.float())
    ).bfloat16()
    torch.testing.assert_close(actual, expected.flatten(1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
