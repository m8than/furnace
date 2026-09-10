#!/usr/bin/env python3
"""Manual raw-bit parity and CUDA-event probe; never promotes a scheduling variant.

Load baseline factory definitions from the required --original path, not a
reimplemented reference.
Run in the candidate Python environment on gfx942. Compilation happens only when
this script is run. Instrumented FP32 outputs are additional diagnostics: changing
an output type can change lowering, so they do not replace unmodified BF16 checks.
"""

import argparse
import ast
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import tilelang_kernel as candidate
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend


def load_original(path):
    spec = importlib.util.spec_from_file_location("_sparse_prefill_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fp32_output_factory(module, name, output_name, directory, label):
    """Expose the existing final FP32 accumulator, changing only output storage."""
    tree = ast.parse(Path(module.__file__).read_text())
    function = copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    function.name = label
    changed = 0
    for node in ast.walk(function):
        if isinstance(node, ast.arg) and node.arg == output_name:
            annotation = node.annotation
            if isinstance(annotation, ast.Call):
                assert len(annotation.args) == 2
                annotation.args[1] = ast.Name(id="accum_dtype", ctx=ast.Load())
            else:
                assert isinstance(annotation, ast.Subscript)
                assert isinstance(annotation.slice, ast.Tuple)
                annotation.slice.elts[1] = ast.Name(id="accum_dtype", ctx=ast.Load())
            changed += 1
    assert changed == 1
    source = (
        ast.unparse(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        )
        + "\n"
    )
    path = Path(directory) / (label + ".py")
    path.write_text(source)
    # A real source path lets TileLang/inspect recover the nested prim_func.
    namespace = dict(module.__dict__)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[label]


def compare(label, actual, expected):
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (label, actual.dtype, expected.dtype)
    bits = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    mismatch = actual.contiguous().view(bits) != expected.contiguous().view(bits)
    count = mismatch.sum().item()
    if count:
        first = mismatch.flatten().nonzero()[0].item()
        a = actual.flatten()[first].item()
        b = expected.flatten()[first].item()
        raise AssertionError(
            f"{label}: {count} raw-bit mismatches; first={first}, actual={a}, expected={b}"
        )


def make_case(seq, topk, kv_len, kind, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        (seq, 32, 512), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    kv = torch.randn(
        (kv_len, 1, 512), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    indices = torch.randint(
        kv_len, (seq, 1, topk), dtype=torch.int32, device="cuda", generator=generator
    )
    # Fixed selection order: fragmented random gathers, repeated indices, and the
    # last legal address in a large allocation. Never sort/deduplicate selections.
    indices[:, :, ::7] = kv_len - 1
    indices[:, :, 1::11] = 0
    indices[:, :, 2::13] = -1
    indices[:, :, 31::32] = kv_len // 2
    live_topk = 2051 if topk == 2112 else topk
    indices[:, :, live_topk:] = -1
    if seq > 1:
        indices[0] = -1
        indices[1, :, :32] = -1
    if kind == "masked":
        indices.fill_(-1)
    elif kind == "adversarial":
        # Cancellation, signed zeros, subnormal BF16, sharp softmax and exact ties.
        values = torch.tensor(
            [0x0000, -32768, 0x0001, -32767, 0x3F80, -16512, 0x4180, -15968],
            dtype=torch.int16,
            device="cuda",
        ).view(torch.bfloat16)
        q.copy_(values.repeat(64).view(1, 1, 512).expand_as(q))
        kv.copy_(values.flip(0).repeat(64).view(1, 1, 512).expand_as(kv))
        kv[::3].neg_()
        q[::4].zero_()
        kv[::5].mul_(16)
    elif kind == "nonfinite":
        # Masked indices still gather KV[0] in the original kernel. Preserve its
        # nonfinite behavior rather than silently zeroing those loaded values.
        kv[0, 0, :8] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1.0, -1.0, 1e30],
            dtype=torch.bfloat16,
            device="cuda",
        )
        q[::5, :, 0] = float("nan")
        q[::7, :, 1] = float("inf")
    elif kind != "random":
        raise ValueError(kind)
    return q, kv, indices, live_topk


def capture(function):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    return graph, output


def milliseconds(function, warmup, repeat):
    for _ in range(warmup):
        function()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeat):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def dump_kernel(kernel, directory, label):
    if directory is not None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (label + ".cu")).write_text(kernel.get_kernel_source())


def run_case(args, original, fp32, seq, topk, kind, record):
    q, kv, indices, live_topk = make_case(seq, topk, args.kv_len, kind, args.seed)
    inputs = (q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))
    inner = original._pick_inner_iter(seq, topk // 32, 304, 1)
    groups = topk // (32 * inner)
    config = dict(sm_scale=args.sm_scale, block_I=32, inner_iter=inner)
    candidate_config = dict(config, layout_bridge=args.layout_bridge)
    baseline_partial = original.sparse_mla_fwd_decode_partial(
        32, 512, 0, topk, threads=128, **config
    )
    combine = original.sparse_mla_fwd_decode_combine(
        32, 512, groups * 32, 4, block_I=32, threads=128
    )
    ref_partial, ref_lse = baseline_partial(*inputs)
    reference = combine(ref_partial, ref_lse)
    ref_fp32, ref_fp32_lse = fp32["original_partial"](
        32, 512, 0, topk, threads=128, **config
    )(*inputs)
    compare("original instrumented LSE", ref_fp32_lse, ref_lse)
    ref_combined_fp32 = fp32["original_combine"](
        32, 512, groups * 32, 4, block_I=32, threads=128
    )(ref_partial, ref_lse)
    record.update(inner_iter=inner, groups=groups, variants=[])
    label = f"q{seq}_k{topk}_{kind}"
    if args.graph_replays:
        baseline_graph, baseline_graph_output = capture(
            lambda: combine(*baseline_partial(*inputs))
        )
        for _ in range(args.graph_replays):
            baseline_graph.replay()
        compare("original graph BF16", baseline_graph_output, reference)
        if args.repeat:
            record["original_graph_ms"] = milliseconds(
                baseline_graph.replay, args.warmup, args.repeat
            )
    dump_kernel(baseline_partial, args.dump_dir, label + "_original_partial")
    dump_kernel(combine, args.dump_dir, label + "_original_combine")
    if args.repeat:
        record["original_ms"] = milliseconds(
            lambda: combine(*baseline_partial(*inputs)), args.warmup, args.repeat
        )
    for threads in args.threads:
        result = dict(
            threads=threads, fused=groups == 1, layout_bridge=args.layout_bridge
        )
        try:
            partial = candidate.sparse_mla_fwd_decode_partial(
                32, 512, 0, topk, threads=threads, **candidate_config
            )
            dump_kernel(
                partial,
                args.dump_dir,
                label + f"_partial_t{threads}_bridge{int(args.layout_bridge)}",
            )
            actual_partial, actual_lse = partial(*inputs)
            compare("BF16 partial", actual_partial, ref_partial)
            compare("FP32 LSE", actual_lse, ref_lse)
            actual_fp32, fp32_lse = fp32["candidate_partial"](
                32, 512, 0, topk, threads=threads, **candidate_config
            )(*inputs)
            compare("FP32 normalized PV accumulator", actual_fp32, ref_fp32)
            compare("instrumented FP32 LSE", fp32_lse, ref_lse)
            compare(
                "original combine of candidate partial",
                combine(actual_partial, actual_lse),
                reference,
            )
            if args.repeat:
                result["partial_ms"] = milliseconds(
                    lambda: partial(*inputs), args.warmup, args.repeat
                )
                result["partial_plus_combine_ms"] = milliseconds(
                    lambda: combine(*partial(*inputs)), args.warmup, args.repeat
                )
            kernel = partial
            if groups == 1:
                kernel = candidate.sparse_mla_fwd_decode_partial(
                    32,
                    512,
                    0,
                    topk,
                    threads=threads,
                    fuse_single_group=True,
                    **candidate_config,
                )
                fused, fused_lse = kernel(*inputs)
                compare("fused BF16 output", fused.squeeze(2), reference)
                compare("fused FP32 LSE", fused_lse, ref_lse)
                fused_fp32, fused_fp32_lse = fp32["candidate_partial"](
                    32,
                    512,
                    0,
                    topk,
                    threads=threads,
                    fuse_single_group=True,
                    **candidate_config,
                )(*inputs)
                compare(
                    "fused FP32 combine accumulator",
                    fused_fp32.squeeze(2),
                    ref_combined_fp32,
                )
                compare("fused instrumented FP32 LSE", fused_fp32_lse, ref_lse)
                function = lambda kernel=kernel: kernel(*inputs)[0].squeeze(2)
            else:
                function = lambda kernel=kernel: combine(*kernel(*inputs))
            dump_kernel(
                kernel,
                args.dump_dir,
                label + f"_candidate_t{threads}_bridge{int(args.layout_bridge)}",
            )
            if args.graph_replays:
                partial_graph, partial_graph_outputs = capture(
                    lambda kernel=kernel: kernel(*inputs)
                )
                for _ in range(args.graph_replays):
                    partial_graph.replay()
                graph_partial, graph_lse = partial_graph_outputs
                compare("graph FP32 LSE", graph_lse, ref_lse)
                compare(
                    "graph raw partial/fused BF16",
                    graph_partial.squeeze(2) if groups == 1 else graph_partial,
                    reference if groups == 1 else ref_partial,
                )
                graph, graph_output = capture(function)
                for _ in range(args.graph_replays):
                    graph.replay()
                compare("graph candidate BF16", graph_output, reference)
                if args.repeat:
                    result["graph_ms"] = milliseconds(
                        graph.replay, args.warmup, args.repeat
                    )
            if args.repeat:
                result["ms"] = milliseconds(function, args.warmup, args.repeat)
            result["raw_bits_equal"] = True
        except Exception as error:
            # Wider schedules are candidates, not approved defaults. Preserve
            # their compile/parity failures in the report and exit nonzero.
            result["error"] = repr(error)
            result["raw_bits_equal"] = False
        record["variants"].append(result)

    # Exercise the actual backend padding helper and actual flag dispatch; no
    # backend/model constructor or synthetic replacement attention is involved.
    page_table = indices[:, 0, :live_topk]

    def dispatch(is_prefill):
        return DeepseekSparseAttnBackend._forward_tilelang(
            None, q, kv, 512, page_table, args.sm_scale, is_prefill=is_prefill
        )

    real_factory = candidate.sparse_mla_fwd_decode_partial
    for enabled, is_prefill in ((False, True), (True, False), (True, True)):
        with envs.SGLANG_OPT_GLM_PREFILL_DSA_TILES.override(enabled):
            with patch.object(
                candidate, "sparse_mla_fwd_decode_partial", wraps=real_factory
            ) as called:
                output = dispatch(is_prefill)
            record.setdefault("dispatch", []).append(
                dict(
                    flag=enabled,
                    is_prefill=is_prefill,
                    factory_args=called.call_args.args,
                    factory_kwargs=called.call_args.kwargs,
                )
            )
            compare(f"dispatch flag={enabled} prefill={is_prefill}", output, reference)
            used_fusion = called.call_args.kwargs.get("fuse_single_group", False)
            expected_fusion = (
                enabled and is_prefill and groups == 1 and topk in (2048, 2112)
            )
            assert used_fusion == expected_fusion
            assert (
                called.call_args.kwargs.get("layout_bridge", False) == expected_fusion
            )
            assert called.call_args.kwargs["threads"] == (
                256 if expected_fusion else 128
            )
    if args.graph_replays:
        with envs.SGLANG_OPT_GLM_PREFILL_DSA_TILES.override(True):
            graph, graph_output = capture(lambda: dispatch(True))
            for _ in range(args.graph_replays):
                graph.replay()
            compare("backend graph BF16", graph_output, reference)
            # Change captured input storage, retaining valid fixed index order
            # for both implementations, and prove replay is not a stale output.
            q.neg_()
            indices.copy_(indices.roll(32, dims=-1))
            indices[:, :, live_topk:] = -1
            new_reference = combine(*baseline_partial(*inputs))
            for _ in range(args.graph_replays):
                graph.replay()
            compare("backend graph changed-input BF16", graph_output, new_reference)
            if args.repeat:
                record["dispatch_graph_ms"] = milliseconds(
                    graph.replay, args.warmup, args.repeat
                )
    record["dispatch_raw_bits_equal"] = True
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help=(
            "Unmodified tilelang_kernel.py to load the baseline factories from. "
            "Supply a pristine checkout; there is no default."
        ),
    )
    parser.add_argument("--queries", type=int, nargs="+", default=[8192, 288, 17, 1])
    parser.add_argument("--topks", type=int, nargs="+", default=[2048, 2112])
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["random", "adversarial", "masked", "nonfinite"],
        default=["random", "adversarial", "masked", "nonfinite"],
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        choices=[128, 256, 512],
        default=[128, 256, 512],
    )
    parser.add_argument(
        "--layout-bridge",
        action="store_true",
        help="Probe FP32 shared row-layout bridges; requires --threads 256",
    )
    parser.add_argument("--kv-len", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--sm-scale", type=float, default=256**-0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="CUDA-event timing iterations; zero disables timing",
    )
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="Optional generated HIP sources for MFMA/reduction audit",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    assert args.kv_len > 0 and all(s > 0 for s in args.queries)
    assert all(k > 0 and k % 64 == 0 for k in args.topks)
    assert args.repeat >= 0 and args.warmup >= 0 and args.graph_replays >= 0
    assert candidate._is_gfx942_supported and torch.version.hip, "gfx942 ROCm required"
    original = load_original(args.original)
    report = dict(
        original=str(args.original),
        candidate=candidate.__file__,
        torch=torch.__version__,
        hip=torch.version.hip,
        device=torch.cuda.get_device_properties(0).gcnArchName,
        sm_scale=args.sm_scale,
        cases=[],
    )
    with tempfile.TemporaryDirectory(prefix="glm_sparse_fp32_") as directory:
        fp32 = {
            "original_partial": fp32_output_factory(
                original,
                "sparse_mla_fwd_decode_partial",
                "Partial_O",
                directory,
                "original_partial_fp32",
            ),
            "original_combine": fp32_output_factory(
                original,
                "sparse_mla_fwd_decode_combine",
                "Output",
                directory,
                "original_combine_fp32",
            ),
            "candidate_partial": fp32_output_factory(
                candidate,
                "sparse_mla_fwd_decode_partial",
                "Partial_O",
                directory,
                "candidate_partial_fp32",
            ),
        }
        for seq in args.queries:
            for topk in args.topks:
                for kind in args.cases:
                    result = dict(queries=seq, topk=topk, case=kind)
                    try:
                        run_case(args, original, fp32, seq, topk, kind, result)
                    except Exception as error:
                        result["error"] = repr(error)
                    report["cases"].append(result)
                    print(json.dumps(result), flush=True)
                    if args.report:
                        args.report.write_text(json.dumps(report, indent=2) + "\n")
    failed = any(
        "error" in r or any("error" in v for v in r.get("variants", []))
        for r in report["cases"]
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
