# Furnace

The AMD ROCm fork of SGLang that lives at `/workspace/sglang`, plus the benchmark
evidence that backs every number below.

Everything here was measured on one host:

| | |
|---|---|
| GPUs | 8x AMD Instinct MI325X, 256 GiB each, gfx942, single node XGMI |
| CPU / RAM | 2x EPYC 9575F (128 physical cores), 3 TiB |
| Stack | ROCm 7.2, PyTorch 2.9.1+rocm7.2.0, Triton 3.6.0 (pinned), AITER 0.1.22.dev32+g456b92780 |
| Python | `/opt/venv/bin/python3` |

---

## So, about the fork

MI325 only for now.

---

## The three models at a glance

| Model | Best config on this host | Measured | Workload |
|---|---|---:|---|
| **Kimi-K3** | TP8 + DFlash2 block 8 | **2,306.67 out tok/s** | 1,000-in / 2,048-out, c64 |
| **MiniMax-M3** | TP4, tuned gfx942 dense tiles | **176.91 out tok/s / 11,499 total tok/s** | 8,192-in / 128-out, c32 |
| **GLM-5.3-Flash** | TP4/EP4 NVFP4 on 4x GB300 (NVIDIA) | **6,150.46 out tok/s** | 1,024-in / 256-out, c256 |
| GLM-5.3-Flash (this host) | TP8 TileLang DSA, BF16 KV | accuracy only, 0.9712 GSM8K | no AMD throughput measured |

---

## 1. Kimi-K3

**Best setup: TP8 with the DFlash2 drafter, block size 8, 524k context / 640k KV.**

```bash
SGLANG_USE_AITER=1 \
SGLANG_AITER_K3_OPT=1 \
SGLANG_AITER_MXFP4_TRITON=1 \
SGLANG_MLA_DECODE_TUNE=1 \
SGLANG_K3_SP_ATTN_RES=1 \
SGLANG_K3_PREFILL_BF16_MOE=1 \
SGLANG_GDN_CHUNK_H_BV=16 \
SGLANG_GDN_CHUNK_H_NUM_WARPS=4 \
SGLANG_GDN_CHUNK_H_NUM_STAGES=2 \
HF_HUB_OFFLINE=1 \
PYTHONPATH=/workspace/runtime/kimi-k3/triton-3.6.0:/workspace/sglang/python \
python3 -m sglang.launch_server \
  --model-path /workspace/models/Kimi-K3 \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs-decode 64 \
  --max-running-requests 64 \
  --disable-radix-cache \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --model-loader-extra-config '{"enable_multithread_load":true}' \
  --watchdog-timeout 1200 \
  --chunked-prefill-size 65536 \
  --max-prefill-tokens 120000 \
  --random-seed 288168147 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /workspace/models/Kimi-K3-DFlash2 \
  --speculative-dflash-block-size 8 \
  --speculative-draft-attention-backend triton \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --linear-attn-prefill-backend triton \
  --linear-attn-decode-backend triton \
  --linear-attn-verify-backend triton \
  --enable-linear-replayssm-spec \
  --linear-replayssm-cache-len 32 \
  --mamba-ssm-dtype float32 \
  --max-total-tokens 655360 \
  --context-length 524288 \
  --min-free-slots-delay 1 \
  --host 127.0.0.1 --port 30000
```

**Benchmark:**

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /workspace/models/Kimi-K3 \
  --tokenizer /workspace/models/Kimi-K3 \
  --dataset-name random \
  --tokenize-prompt \
  --random-input-len 1000 \
  --random-output-len 2048 \
  --random-range-ratio 1 \
  --num-prompts 128 \
  --max-concurrency 64 \
  --request-rate inf \
  --seed 42 \
  --temperature 0 \
  --warmup-requests 1 \
  --flush-cache \
  --cache-report \
  --output-details
```

| Metric | Value |
|---|---:|
| Aggregate output throughput | **2,306.67 tok/s** |
| Mean TTFT | 3,429.9 ms |
| Mean TPOT | 22.52 ms |
| Accept length | 7.364 |
| Completed | 128 / 128, 0 cached tokens |

Same profile at 64 requests: 2,149.44 tok/s (1,658.91 before the optimization, +29.6%).
Provenance: `benchmarks/kimi-k3/optimization/dflash2-long-batch/final-b8-queued/summary.json`
and `.../optimized-b8/summary.json`, launch in `.../optimized-b8-launch.json`.

**Regime map** (same server, different workloads):

| Workload | Aggregate out tok/s |
|---|---:|
| 1,000-in / 2,048-out, c64 | 2,306.67 |
| 32,000-in / 2,048-out, c16 | 305.15 |
| 32,000-in / 2,048-out, c8 | 248.71 |
| 120,000-in / 2,048-out, c4 | 64.03 |
| 1,000-in / 8,000-out, c1 (decode window) | 198.35 |

**Non-speculative base** (no draft model, `--speculative-*` omitted): 436.91 out
tok/s at 8,192-in / 1,024-out / c64, and 56.36 decode tok/s single-stream —
`optimization/accepted-base-summary.json`.

**DCP8 capacity lane** (only when you need the context, not the speed): 173.37
tok/s with DFlash2 at 32k x8, 121.89 without. DCP shards the MLA KV but does not
raise the KDA concurrency ceiling.

Caveats: every measured run disables the radix cache; the 655,360-token pool is
shared, so it cannot hold 64 simultaneous 512k contexts; `verification-summary.json`
records `quality.passed: false` for the DFlash2 long-batch candidate (aggregate
GSM8K/MMLU identical, strict per-item no-regression gate tripped) — it is not
certified quality-neutral.

---

## 2. MiniMax-M3

**Best setup: TP4 on four MI325X, no DCP, no speculation, gfx942-tuned dense
GEMM tiles (shipped in the build, nothing extra on the command line).**

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
HF_HUB_OFFLINE=1 \
SGLANG_USE_AITER=1 \
SGLANG_OPT_USE_BF16_ROUTER_GEMM=0 \
ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.80 \
python3 -m sglang.launch_server \
  --model-path /workspace/models/MiniMax-M3-MXFP8 \
  --trust-remote-code \
  --tp-size 4 \
  --quantization mxfp8 \
  --dtype bfloat16 \
  --attention-backend triton \
  --moe-runner-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --context-length 1048576 \
  --max-running-requests 64 \
  --cuda-graph-max-bs-decode 64 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 65536 \
  --disable-radix-cache \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --model-loader-extra-config '{"enable_multithread_load":true}' \
  --random-seed 288168147 \
  --watchdog-timeout 3600 \
  --host 127.0.0.1 --port 30001
```

**Benchmark** (32 concurrent, 8,192-in / 128-out, temperature 0, EOS ignored, no
prefix-cache hits; seeds are fixed in the harness):

```bash
python3 /workspace/benchmarks/minimax-m3/dspark-optimization/m3_tp2_control_bench.py \
  tp4-dense-tuned --port 30001 --cases 32:8192 --output-tokens 128
```

| Metric | Baseline TP4 | Tuned TP4 |
|---|---:|---:|
| Aggregate output tok/s | 160.66 | **176.91** |
| Aggregate input tok/s | 10,282.0 | 11,322.2 |
| Total tok/s | 10,442.7 | **11,499.1** |
| 32-request wall time | 25.50 s | 23.15 s |
| Median TTFT | 9.77 s | 9.81 s |
| Median steady-tail TPOT | 51.96 ms | **32.93 ms** |

Sustained 128-request run: 160.32 -> 176.75 tok/s. The delta is +10.1% end to end;
the 36.6% figure is decode-only and is not an end-to-end claim.
Provenance: `benchmarks/minimax-m3/high-concurrency-20260909/tp4-performance-comparison.json`.

**Workload-shaped alternatives:**

| Workload | Config delta | Result |
|---|---|---:|
| 32 x 60K-in / 60K-out closed batch | + DSpark block 4, `--mem-fraction-static 0.96 --max-total-tokens 3932160`, `SGLANG_RAGGED_VERIFY_MODE=static` | **828.14 tok/s** (vs 681.92 no-spec, 562.09 TP2xPP2) |
| 8 x 262,144-in prefill | PP4xTP1, `SGLANG_PP_LAYER_PARTITION=3,19,19,19`, `SGLANG_ENABLE_M3_ROCM_FP8_ATTN_GEMM=1`, `--disable-overlap-schedule` | **20,215.8 input tok/s** (vs 8,299 frozen) |
| 1,040,384-in single request | TP8 + DCP2 | 141.69 s TTFT vs 147.21 s TP-only; warmed 117.8 s |

The DSpark runner is
`python3 /workspace/benchmarks/minimax-m3/tp4-spec-60k-in-60k-out-c32/run_benchmark.py {baseline|dspark}`.

Caveats: gfx942 always runs the load-time MXFP8 -> `[128,128]` block-FP8
conversion, never native MXFP8. DCP2 loses on everything below a million tokens
(216.60 -> 181.41 tok/s at 32 x 8K). DSpark is workload-specific: 10 of 72 cold
cells qualified, both mixed-arrival schedules kept the non-speculative control.
Chunked prefill stays at 8,192 — 32,768 changed greedy responses for ~1% gain.

---

## 3. GLM-5.3-Flash

GLM-5.3-Flash is now in this fork (`ca34de84bb`, merged from
`xinyuan/glm-5.3-flash-support`): `Glm5NextForConditionalGeneration`,
`Glm5NextConfig`, the MTP head, the DSA k-pool indexer, mHC communicators, the
k-pool memory pools, and the AMD gfx942/gfx950 enablement.

### 3a. Best throughput (NVIDIA, measured)

4x GB300, TP4/EP4, speculative decoding **off**, `RadixArk/GLM-5.3-Flash-NVFP4`
with FP8 KV + TRT-LLM DSA:

```bash
sglang serve \
  --model-path RadixArk/GLM-5.3-Flash-NVFP4 \
  --quantization modelopt_fp4 \
  --tp-size 4 --ep-size 4 \
  --dsa-prefill-backend trtllm --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_cutlass \
  --mem-fraction-static 0.85 \
  --reasoning-parser glm45 --tool-call-parser glm47
```

```bash
python3 -m sglang.bench_serving \
  --backend sglang --host localhost --port 30000 \
  --model RadixArk/GLM-5.3-Flash-NVFP4 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --random-range-ratio 1.0 \
  --num-prompts 1280 --max-concurrency 256 \
  --request-rate inf --temperature 0 --seed 42 --flush-cache
```

| Arm (4x GB300, 1,024-in / 256-out) | c16 | c64 | c256 |
|---|---:|---:|---:|
| NVFP4 + FP8 KV/TRT-LLM (**best**) | 1,439.25 | 3,428.58 | **6,150.46** |
| NVFP4 + BF16 KV/TileLang | 1,352.86 | 3,291.36 | 5,919.49 |
| FP8 + FP8 KV/TRT-LLM | 1,227.07 | 2,738.61 | 4,977.02 |
| FP8 + BF16 KV/TileLang | 1,161.22 | 2,660.24 | 4,828.33 |

Provenance: `sglang/docs/src/snippets/configs/zai-org/glm-5.3-flash-benchmarks.jsx`.
The latency-first arm (adaptive MTP 5/1/6) peaks at 1,853.88 tok/s at c16 — for
sustained batches, turn speculation off instead.

### 3b. On this host (MI325X) — accuracy validated, throughput unmeasured

```bash
SGLANG_USE_AITER=1 \
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-5.3-Flash \
  --tp-size 8 \
  --trust-remote-code \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --kv-cache-dtype bfloat16 \
  --moe-runner-backend triton \
  --disable-prefill-cuda-graph \
  --disable-decode-cuda-graph \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --watchdog-timeout 1200 \
  --model-loader-extra-config '{"enable_multithread_load":true}'
```

GSM8K on gfx942: 0.9712 (12,259 s on the ROCm 7.2 image). This is an accuracy
gate, not a throughput result — **no performance job for GLM-5.3-Flash exists on
AMD**, and no GLM weights are downloaded under `/workspace/models`. Do not quote
a tok/s figure for this model on MI325X; measure it first, with
`python3 -m sglang.bench_serving` and the workload you actually care about.

gfx942 takes the portable unfused Torch DSA top-k and the generic mHC path;
gfx950 gets the fused SGL-Kernel k-pool top-k and AITER mHC. The gfx942 recipe
also exercises paths no other AMD nightly covers, which is why it is gated
separately.

---

## Repo map

```
sglang/            the fork (branch: main, remote: furnace)
benchmarks/kimi-k3/     launch records, summaries, quality gates, optimization tree
benchmarks/minimax-m3/  TP2/TP4/PP4 matrices, DSpark and DCP experiments
models/            Kimi-K3, Kimi-K3-DFlash2, MiniMax-M3-MXFP8, MiniMax-M3-DSpark
runtime/           pinned source trees (Triton 3.6.0, per-run SGLang checkouts)
```

Every benchmark directory carries a `launch.json` (exact argv and environment),
a `summary.json` (measured metrics) and, where quality was gated, the raw
outcomes. That is the point of the tree: the numbers are reproducible, and the
failures are still on the record.
