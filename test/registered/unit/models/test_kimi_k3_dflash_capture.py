"""Draft features must preserve completed-layer stream values and their lifetime."""

from types import SimpleNamespace

import torch
from sglang.srt.layers.attn_residual import AttnResidual
from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM, KimiK3LinearModel
from sglang.test.ci.ci_register import register_cpu_ci
from torch import nn

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ScoreProjection(nn.Linear):
    def forward(self, hidden):
        return super().forward(hidden), None


def _consumer(weights, valid_blocks):
    projection = _ScoreProjection(3, 1, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([weights]))
    return SimpleNamespace(
        self_attention_res_proj=projection,
        self_attention_res_norm=nn.RMSNorm(3, eps=1e-6),
        prev_valid_blocks=valid_blocks,
    )


def _target():
    # The capture API does not need embeddings, experts or attention weights.
    target = KimiK3LinearForCausalLM.__new__(KimiK3LinearForCausalLM)
    nn.Module.__init__(target)
    target.config = SimpleNamespace(num_hidden_layers=3, attn_res_block_size=2)
    target.pp_group = SimpleNamespace(world_size=1)
    target.capture_aux_hidden_states = False
    backbone = KimiK3LinearModel.__new__(KimiK3LinearModel)
    nn.Module.__init__(backbone)
    backbone.config = target.config
    backbone.end_layer = 3
    backbone.layers = [
        _consumer([0.0, 0.0, 0.0], 0),
        _consumer([1.0, -0.5, 0.25], 1),
        _consumer([-0.25, 1.0, 0.5], 2),
    ]
    output = _consumer([0.5, 0.25, -1.0], 2)
    backbone.output_attn_res_proj = output.self_attention_res_proj
    backbone.output_attn_res_norm = output.self_attention_res_norm
    target.model = backbone
    return target


def _reference_mix(prefix, bank, count, weights):
    rows = torch.cat((bank[:, :count], prefix.unsqueeze(1)), dim=1)
    normed = rows * torch.rsqrt(rows.square().mean(-1, keepdim=True) + 1e-6)
    scores = (normed * torch.tensor(weights)).sum(-1)
    return (scores.softmax(-1).unsqueeze(-1) * rows).sum(1)


def test_prefix_taps_exclude_banked_history_and_survive_block_reset():
    target = _target()
    target.set_dflash_layers_to_capture([0, 2])
    hidden = torch.tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    state = AttnResidual(hidden, block_num=2)
    state.write(torch.tensor([[20.0, 30.0, 40.0], [50.0, 60.0, 70.0]]))
    captures = AuxHiddenStatePacker(2)
    target.model._capture_aux_stream(0, hidden, None, state, captures)

    # Bank the old block and reuse its activation storage for the new block.
    state.write(hidden)
    hidden.fill_(5.0)
    residual = torch.tensor([[0.25, 0.5, 0.75], [0.125, 0.25, 0.5]])
    target.model._capture_aux_stream(2, hidden, residual, state, captures)
    hidden.zero_()
    residual.zero_()
    state.block_residual.zero_()

    torch.testing.assert_close(
        captures.finalize(),
        torch.tensor(
            [[1.0, 2.0, 3.0, 5.25, 5.5, 5.75], [7.0, 8.0, 9.0, 5.125, 5.25, 5.5]]
        ),
    )


def test_attn_res_taps_use_next_consumer_and_final_output_consumer(monkeypatch):
    # Exercise the eager CPU reference even when this unit suite runs on HIP.
    monkeypatch.setattr(
        "sglang.srt.layers.attn_residual._use_hip_fused", lambda *_: False
    )
    target = _target()
    target.set_dspark_layers_to_capture([0, 2])
    hidden = torch.tensor([[1.0, 2.0, -1.0]])
    residual = torch.tensor([[0.5, -0.25, 0.75]])
    prefix = hidden + residual
    state = AttnResidual(hidden, block_num=2)
    state.write(torch.tensor([[4.0, -1.0, 2.0]]))
    state.write(torch.tensor([[-2.0, 3.0, 1.0]]))

    captures = AuxHiddenStatePacker(2)
    target.model._capture_aux_stream(0, hidden, residual, state, captures)
    target.model._capture_aux_stream(2, hidden, residual, state, captures)
    next_consumer, output_consumer = captures.finalize().split(3, dim=-1)
    torch.testing.assert_close(
        next_consumer,
        _reference_mix(prefix, state.block_residual, 1, [1.0, -0.5, 0.25]),
    )
    torch.testing.assert_close(
        output_consumer,
        _reference_mix(prefix, state.block_residual, 2, [0.5, 0.25, -1.0]),
    )

    # DFLASH's implicit stream is prefix; an explicit trained AttnRes stream
    # must recover the same feature without changing completed-layer IDs.
    target.set_dflash_layers_to_capture([0, 2])
    prefix_capture = AuxHiddenStatePacker(1)
    target.model._capture_aux_stream(0, hidden, residual, state, prefix_capture)
    torch.testing.assert_close(prefix_capture.finalize(), prefix)
    target.set_dflash_aux_hidden_stream("attn_res")
    attn_res_capture = AuxHiddenStatePacker(1)
    target.model._capture_aux_stream(0, hidden, residual, state, attn_res_capture)
    torch.testing.assert_close(attn_res_capture.finalize(), next_consumer)


def test_zero_bank_attn_res_snapshot_does_not_alias_activation():
    target = _target()
    target.set_dspark_layers_to_capture([0])
    target.model.layers[1].prev_valid_blocks = 0
    hidden = torch.tensor([[1.0, 2.0, 3.0]])
    state = AttnResidual(hidden, block_num=2)
    captures = AuxHiddenStatePacker(1)
    target.model._capture_aux_stream(0, hidden, None, state, captures)
    snapshot = captures.finalize()
    hidden.add_(10.0)
    torch.testing.assert_close(snapshot, torch.tensor([[1.0, 2.0, 3.0]]))
