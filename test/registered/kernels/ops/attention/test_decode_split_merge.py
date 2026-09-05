import sys

import pytest
import torch

from sglang.kernels.ops.attention.decode_attention import _decode_softmax_reducev_fwd
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, stage="jit-kernel-unit", runner_config="amd")


@pytest.mark.parametrize("case", ["mixed_with_sink", "forced", "large_batch"])
def test_split_merge_matches_float64_reference(case):
    heads, dim, capacity = 3, 512, 256
    if case == "mixed_with_sink":
        lengths = [0, 1, 31, 33, 8192, 16384]
        counts = [16, 256, 256, 256, 256, 96]
        forced, has_sink, dtype = 0, True, torch.float32
    elif case == "forced":
        lengths, counts = [8192], [1]
        forced, has_sink, dtype = 128, False, torch.bfloat16
    else:
        heads = 12
        lengths, counts = [8192] * 64, [34] * 64
        forced, has_sink, dtype = 0, False, torch.float32
    batch = len(lengths)
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    partial = torch.randn(
        (batch, heads, capacity, dim), device="cuda", generator=generator
    )
    lse = torch.randn((batch, heads, capacity), device="cuda", generator=generator)
    sink = torch.randn(heads, device="cuda", generator=generator) if has_sink else None
    length_tensor = torch.tensor(lengths, device="cuda", dtype=torch.int64)
    count_tensor = torch.tensor(counts, device="cuda", dtype=torch.int32)
    used_count = torch.full_like(count_tensor, forced) if forced else count_tensor
    chunk = ((length_tensor + used_count - 1) // used_count + 31) // 32 * 32
    split = torch.arange(capacity, device="cuda")
    valid = (split[None, :] < used_count[:, None]) & (
        split[None, :] * chunk[:, None] < length_tensor[:, None]
    )
    # Unwritten split entries must never contribute, even if poisoned with NaNs.
    partial.masked_fill_(~valid[:, None, :, None], float("nan"))
    lse.masked_fill_(~valid[:, None, :], float("nan"))
    logical_lse = lse.double().masked_fill(~valid[:, None, :], -float("inf"))
    maximum = logical_lse.max(-1, keepdim=True).values
    weights = torch.where(valid[:, None, :], torch.exp(logical_lse - maximum), 0.0)
    denominator = weights.sum(-1)
    if sink is not None:
        denominator += torch.exp(sink.double()[None, :] - maximum.squeeze(-1))
    values = partial.double().masked_fill(~valid[:, None, :, None], 0.0)
    scale = 0.75
    expected = (
        (values * weights[..., None]).sum(-2) / denominator[..., None] * scale
    ).to(dtype)
    indptr = torch.zeros(batch + 1, device="cuda", dtype=torch.int64)
    indptr[1:] = length_tensor.cumsum(0)
    q = torch.empty((batch, heads, 576), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((1, 1, dim), device="cuda", dtype=torch.bfloat16)
    backing = torch.full((batch, heads, dim + 1), 17.0, device="cuda", dtype=dtype)
    output = backing[..., :dim]
    _decode_softmax_reducev_fwd(
        partial,
        lse,
        q,
        output,
        scale,
        v,
        indptr,
        count_tensor,
        capacity,
        sinks=sink,
        forced_kv_splits=forced,
    )
    if dtype == torch.bfloat16:
        torch.testing.assert_close(output, expected, rtol=2**-7, atol=2e-5)
    else:
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
    assert torch.all(backing[..., -1] == 17)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
