"""Exact direct-write hc4 capture probe; explicit GPU execution, no model download.

Run with candidate Furnace/AITER PYTHONPATH. Fails on ANY differing BF16 bit,
including signed zero; no tolerances, NaN normalization, or fallback. Representative
activations span stream cancellation and BF16 rounding boundaries. Optional
--activations accepts a torch-saved dict with hidden/residual captured from a model.
"""

import argparse
import itertools
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.kernels.ops.layernorm.glm_aux_capture import capture_hc4_into
from sglang.kernels.ops.layernorm.mhc import hc_contract
from sglang.srt.environ import envs
from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStatePacker,
    pack_aux_hidden_states,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import glm5_next as glm

CAPTURES = [5, 14, 24, 33, 42]
HIDDEN = 4096


def exact(actual, expected, label):
    assert actual.shape == expected.shape, label
    assert actual.dtype == expected.dtype == torch.bfloat16, label
    different = actual.contiguous().view(torch.int16) != expected.contiguous().view(
        torch.int16
    )
    if different.any().item():
        where = different.nonzero()[0].tolist()
        raise AssertionError(
            f"{label}: {different.sum().item()} differing BF16 bits; first {where}"
        )


def representative(tokens, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        (tokens, 4 * HIDDEN), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    residual = torch.randn(
        hidden.shape, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    if tokens == 0:
        return hidden, residual
    # Four-stream order must not be replaced with pairwise/tree reduction.
    patterns = list(itertools.permutations([2.0**30, 1.0, -(2.0**30), 2.0]))
    patterns += [
        [1.0, 1.0078125, -1.0, -1.0078125],
        [-0.0, -0.0, -0.0, -0.0],
        [0.0, -0.0, 0.0, -0.0],
        [2.0**-133, -(2.0**-133), 2.0**-133, 0.0],
    ]
    # Cancellation over a wide finite exponent range, plus final BF16 ties.
    for exponent in range(-126, 121, 7):
        scale = 2.0**exponent
        patterns.extend(
            [
                [scale, scale / 256, -scale, scale / 128],
                [scale, scale / 128, scale, scale / 128],
            ]
        )
    values = torch.tensor(patterns, dtype=torch.bfloat16, device="cuda").T
    count = values.shape[1]
    hidden.view(tokens, 4, HIDDEN)[:, :, :count] = values
    residual.view(tokens, 4, HIDDEN)[:, :, :count] = 0
    # Materialized BF16 add ties, odd/even mantissas, then cancellation.
    hidden.view(tokens, 4, HIDDEN)[:, :, count : count + 4] = torch.tensor(
        [
            [1, 1.0078125, -1, -1.0078125],
            [-1, -1.0078125, 1, 1.0078125],
            [2**-9, 2**-9, -(2**-9), -(2**-9)],
            [0, 0, 0, 0],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    residual.view(tokens, 4, HIDDEN)[:, :, count : count + 4] = torch.tensor(
        [[2**-8, 2**-8, -(2**-8), -(2**-8)], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    return hidden, residual


def baseline(hidden, residual):
    captures = [
        hc_contract(hidden if residual is None else hidden + residual, 4)
        for _ in CAPTURES
    ]
    return pack_aux_hidden_states(captures)


def direct(hidden, residual):
    packer = AuxHiddenStatePacker(len(CAPTURES))
    for _ in CAPTURES:
        capture_hc4_into(hidden, residual, packer.reserve_next(hidden[:, :HIDDEN]))
    assert len(packer) == len(CAPTURES)
    return packer.finalize()


def event_ms(fn, iterations):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iterations):
        result = fn()
        del result
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def peak_bytes(fn):
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    del result
    return peak


def check_graph(hidden, residual):
    # Graph replay must overwrite every reserved slice, not return stale data.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        direct(hidden, residual)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = direct(hidden, residual)
    for _ in range(2):
        hidden.neg_()
        graph.replay()
        exact(output, baseline(hidden, residual), "graph replay")


def check_case(hidden, residual, iterations, label):
    reference = baseline(hidden, residual)
    exact(direct(hidden, residual), reference, label)
    # Independent left fold confirms the pinned PyTorch reduction dispatch.
    streams = (
        (hidden if residual is None else hidden + residual)
        .unflatten(-1, (4, HIDDEN))
        .float()
    )
    zero = torch.zeros_like(streams[:, 0])
    fold = zero + streams[:, 0]
    for index in range(1, 4):
        fold = fold + (zero + streams[:, index])
    exact((fold * 0.25).bfloat16(), reference[:, :HIDDEN], label + " left fold")
    del streams, zero, fold
    # Preserve packed consumer semantics for ragged/pruned rows, including repeats.
    rows = (
        torch.tensor([0, hidden.shape[0] - 1, 0], device="cuda")
        if hidden.shape[0]
        else torch.empty(0, device="cuda", dtype=torch.long)
    )
    output = direct(hidden, residual)
    exact(pack_aux_hidden_states(output[rows]), reference[rows], label + " pruning")
    # Unusual destination row stride, column offset, and untouched canaries.
    storage = torch.full(
        (hidden.shape[0], HIDDEN * 5 + 14), 37, device="cuda", dtype=torch.bfloat16
    )
    destination = storage[:, 7 : 7 + HIDDEN]
    capture_hc4_into(hidden, residual, destination)
    exact(destination, reference[:, :HIDDEN], label + " stride")
    assert (storage[:, :7] == 37).all().item()
    assert (storage[:, 7 + HIDDEN :] == 37).all().item()
    del output, storage, destination, reference
    check_graph(hidden, residual)
    one_capture = hidden.shape[0] * HIDDEN * 2
    print(
        json.dumps(
            {
                "case": label,
                "tokens": hidden.shape[0],
                "residual": residual is not None,
                "baseline_ms": event_ms(lambda: baseline(hidden, residual), iterations),
                "direct_ms": event_ms(lambda: direct(hidden, residual), iterations),
                "baseline_peak_allocated_bytes": peak_bytes(
                    lambda: baseline(hidden, residual)
                ),
                "direct_peak_allocated_bytes": peak_bytes(
                    lambda: direct(hidden, residual)
                ),
                "logical_capture_peak_bytes": {
                    "baseline": 10 * one_capture,
                    "direct": 5 * one_capture,
                },
                "logical_total_allocated_bytes": {
                    "baseline": (30 if residual is not None else 10) * one_capture,
                    "direct": 5 * one_capture,
                },
                "capture_launches": {
                    "baseline": (11 if residual is not None else 6)
                    if hidden.shape[0]
                    else 0,
                    "direct": 5 if hidden.shape[0] else 0,
                },
            }
        ),
        flush=True,
    )


class MarkerLayer(nn.Module):
    def __init__(self, index, residual):
        super().__init__()
        self.index, self.residual = index, residual

    def forward(self, positions, hidden, batch, residual, *args, **kwargs):
        # Mutate the bank to catch capture aliasing across completed layers.
        hidden.fill_(self.index + 1)
        return hidden, self.residual, None


class NarrowNorm(nn.Module):
    def forward(self, hidden, residual=None):
        result = hidden[:, :HIDDEN]
        return result if residual is None else (result, residual)


def check_forward_loop(tokens, residual_present):
    hidden = torch.empty((tokens, HIDDEN * 4), device="cuda", dtype=torch.bfloat16)
    residual = torch.full_like(hidden, 2**-8) if residual_present else None
    model = glm.Glm5NextModel.__new__(glm.Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4, hidden_size=HIDDEN)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.start_layer, model.end_layer = 0, 45
    model.layers = nn.ModuleList([MarkerLayer(i, residual) for i in range(45)])
    model.layers_to_capture = [i + 1 for i in CAPTURES]
    model.dflash_capture, model.enable_a2a_moe = True, False
    model.first_k_dense_replace = 3
    model.norm = NarrowNorm()
    batch = SimpleNamespace(can_run_tbo=False, forward_mode=ForwardMode.EXTEND)
    ids = torch.zeros(tokens, dtype=torch.long, device="cuda")
    positions = torch.arange(tokens, device="cuda")
    recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
    with (
        patch.object(
            glm, "get_global_expert_distribution_recorder", return_value=recorder
        ),
        patch.object(glm, "check_cuda_graph_backend", return_value=False),
    ):

        def run(enabled):
            with envs.SGLANG_OPT_GLM_PACK_AUX_CAPTURE.override(enabled):
                return model(ids, positions, batch, input_embeds=hidden)[1]

        expected = pack_aux_hidden_states(run(False))
        # Count actual fused calls: no passing parity via silent baseline fallback.
        with patch.object(glm, "capture_hc4_into", wraps=capture_hc4_into) as fused:
            output = run(True)
            assert isinstance(output, torch.Tensor)
            assert fused.call_count == len(CAPTURES)
        exact(output, expected, "full forward capture ordering")
        markers = torch.tensor(
            [i + 1 for i in CAPTURES], device="cuda", dtype=torch.bfloat16
        )
        if residual_present:
            markers = markers + 2**-8
        expected_markers = markers.repeat_interleave(HIDDEN).expand(tokens, -1)
        exact(output, expected_markers, "completed layer indices")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replay_output = run(True)
        replay_output.zero_()
        hidden.fill_(-19)
        graph.replay()
        exact(replay_output, expected, "full forward graph replay")
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
            batch.forward_mode = mode
            with patch.object(glm, "capture_hc4_into", wraps=capture_hc4_into) as fused:
                unchanged = run(True)
                assert isinstance(unchanged, list)
                assert fused.call_count == 0
            exact(pack_aux_hidden_states(unchanged), expected, f"baseline {mode}")
        batch.forward_mode = ForwardMode.EXTEND
        model.dflash_capture = False
        assert isinstance(run(True), list), "EAGLE must retain list capture"
    print(
        f"full-forward-loop tokens={tokens} residual={residual_present}: exact",
        flush=True,
    )


def check_packer(tokens):
    source = torch.ones((tokens, HIDDEN), device="cuda", dtype=torch.bfloat16)
    packer = AuxHiddenStatePacker(2, source)
    packer.append(source)
    source.fill_(2)
    destination = packer.reserve_next(source)
    destination.copy_(source)
    source.fill_(3)
    expected = torch.cat((torch.ones_like(source), torch.full_like(source, 2)), -1)
    exact(packer.finalize(), expected, "mixed append/reserve owns captures")
    incomplete = AuxHiddenStatePacker(2)
    incomplete.append(source)
    try:
        incomplete.finalize()
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial packer finalized")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 288, 0])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--activations", help="torch.save dict: hidden [N,16384], optional residual"
    )
    args = parser.parse_args()
    assert torch.version.hip is not None, "ROCm-only direct capture"
    print(
        f"torch={torch.__version__} device={torch.cuda.get_device_name()} source={glm.__file__}",
        flush=True,
    )
    with torch.inference_mode():
        for tokens in args.tokens:
            check_packer(tokens)
            hidden, residual = representative(tokens, 20260909)
            check_case(hidden, residual, args.iterations, "representative+adversarial")
            check_case(hidden, None, args.iterations, "residual-none")
            # Ragged/padded bank storage must not be flattened across row gaps.
            padded = torch.empty(
                (tokens, HIDDEN * 4 + 16), device="cuda", dtype=torch.bfloat16
            )
            padded[:, : HIDDEN * 4].copy_(hidden)
            exact(
                direct(padded[:, : HIDDEN * 4], residual),
                baseline(hidden, residual),
                "strided input",
            )
            del hidden, residual, padded
            check_forward_loop(tokens, True)
            check_forward_loop(tokens, False)
        if args.activations:
            data = torch.load(args.activations, map_location="cuda", weights_only=True)
            check_case(
                data["hidden"],
                data.get("residual"),
                args.iterations,
                "checkpoint-generated",
            )
    print(
        "PASS: exact bits, rounding/order, capture indices, strides, pruning, graph replay",
        flush=True,
    )


if __name__ == "__main__":
    main()
