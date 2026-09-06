"""Strict draft checkpoint coverage excludes vocabulary modules owned by the target."""

import torch
from torch import nn

from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_mla_weight_reload_preserves_the_borrowed_target_head():
    # A minimal real module tree isolates checkpoint ownership from attention
    # execution. The worker attaches the target head only after initial loading.
    draft = DFlash2DraftModel.__new__(DFlash2DraftModel)
    nn.Module.__init__(draft)
    draft.uses_mla = True
    draft.layers = nn.ModuleList()
    draft.fc = nn.Linear(2, 3, bias=False)
    draft.lm_head = None
    draft.load_weights([("fc.weight", torch.ones(3, 2))])

    target_head = nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        target_head.weight.fill_(7.0)
    draft.lm_head = target_head
    draft.load_weights([("fc.weight", torch.arange(6.0).reshape(3, 2))])

    torch.testing.assert_close(
        draft.fc(torch.tensor([[2.0, 3.0]])), torch.tensor([[3.0, 13.0, 23.0]])
    )
    torch.testing.assert_close(
        target_head(torch.tensor([[1.0, 2.0, 3.0]])), torch.full((1, 5), 42.0)
    )
