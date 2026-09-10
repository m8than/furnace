import pytest
import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_append_keeps_casting_and_owns_copied_values():
    packer = AuxHiddenStatePacker(2, torch.empty((1, 2), dtype=torch.bfloat16))
    source = torch.tensor([[1.00390625, 1.01171875]], dtype=torch.float32)
    packer.append(source)
    source.zero_()
    destination = packer.reserve_next(torch.empty((1, 2), dtype=torch.bfloat16))
    destination.fill_(2)
    expected = torch.tensor([[1.0, 1.015625, 2.0, 2.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(packer.finalize(), expected, rtol=0, atol=0)


def test_rejected_reservation_does_not_consume_capture():
    packer = AuxHiddenStatePacker(1, torch.empty((1, 2), dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        packer.reserve_next(torch.empty((1, 2), dtype=torch.float32))
    with pytest.raises(RuntimeError):
        packer.finalize()
    packer.reserve_next(torch.empty((1, 2), dtype=torch.bfloat16)).fill_(3)
    torch.testing.assert_close(
        packer.finalize(), torch.full((1, 2), 3, dtype=torch.bfloat16), rtol=0, atol=0
    )
    with pytest.raises(RuntimeError):
        packer.reserve_next(torch.empty((1, 2), dtype=torch.bfloat16))
