# Maintenance Tools

This folder contains tools and workflows for automating maintenance tasks.

## CI Permissions

`CI_PERMISSIONS.json` defines the CI permissions granted to each user.
Maintainers can directly edit the file to add entries with `"reason": "custom override"`.
Maintainers can also run `update_ci_permission.py` to update it with some auto rules (e.g., top contributors in the last 90 days get full permissions).

Recognized permission keys:

| Key | Grants |
| --- | --- |
| `can_tag_run_ci_label` | `/tag-run-ci-label`, `/tag-and-rerun-ci` |
| `can_rerun_failed_ci` | `/rerun-failed-ci`, `/tag-and-rerun-ci` |
| `cooldown_interval_minutes` | rate limit in `pr-gate.yml`; `0` also grants `/rerun-test`, `/rerun-group` |

`/rerun-test` and `/rerun-group` are gated on the commenter alone: either
`cooldown_interval_minutes: 0`, or `write`/`admin` permission on the repo. Where
the PR comes from makes no difference, and authoring it grants nothing.

Those are the same two signals `pr-gate.yml` already uses to waive its rate
limit, and that is the point -- a selective rerun dispatches `rerun-test.yml`
directly, which never passes through `pr-gate.yml`, so it bypasses the rate limit
by construction. Anyone allowed to run one is therefore unthrottled in practice,
which is exactly what a zero cooldown already declares.

Set a cooldown deliberately: `0` lets the holder run PR-head code on the
self-hosted GPU runners, and raising it above `0` takes that away again along
with their rate-limit waiver.

## GLM-5.3-Flash on MI325X

The fork carries an opt-in prefill path for `zai-org/GLM-5.3-Flash` on gfx942
(MI325X). It is mixed precision, not end-to-end FP8: MoE experts use native
AITER block-FP8 weights and activations, while the KV cache, sparse attention,
and the DFlash2 draft stay BF16 and the recurrent state stays FP32. Checkpoint
weights, dtype settings, and context/RoPE configuration are unchanged.

Three switches enable it. All default to off, and the matching AITER source is
required -- an arbitrary stock wheel is not equivalent:

```bash
export SGLANG_OPT_GLM_PREFILL_DSA_TILES=1
export SGLANG_OPT_GLM_PACK_AUX_CAPTURE=1
export AITER_GLM_PREFILL_STAGE1=1
```

- Sparse attention takes the 256-thread fused single-group schedule only for
  eligible BF16 gfx942 prefills (32 query heads, 512 value channels, top-k 2048
  or 2112). Multi-group and decode dispatch keep the original geometry.
- MoE stage one writes two non-atomic split-K partial planes and combines them
  while retaining the native FP32/BF16/FP8 rounding boundaries. Its GLM TP2
  contract covers 256-32,768 tokens, hidden 4096, intermediate 1024, 289
  experts, top-k 9. Extra workspace is roughly 1.27 GiB per rank at 8192 tokens
  and 4.64 GiB at 32,768.
- Completed-layer DFlash captures write straight into the packed feature buffer.

### Measured results

Same MI325X pair (TP2), cold prefixes, three-trial medians, one output token,
no logprobs; effective input throughput is input tokens divided by TTFT.

| Input tokens | Baseline TTFT (s) | Optimized TTFT (s) | Baseline input tok/s | Optimized input tok/s | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 7.83 | 5.96 | 8,372 | 11,000 | 31.4% |
| 262,144 | 32.57 | 25.04 | 8,048 | 10,469 | 30.1% |
| 500,000 | 65.28 | 50.73 | 7,659 | 9,856 | 28.7% |

Request-rate limits under a fixed 10 s mean / 20 s p95 end-to-end latency budget,
200 measured requests per point, on a 7,000-token prompt with 4,544 tokens cached
(64.9%) and exactly 300 output tokens:

| Configuration | Passing req/s | Completed req/s | Mean (s) | p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| TP2 + DFlash2 | 1.25 | 1.192 | 8.61 | 12.34 |
| TP2, no speculation | 0.75 | 0.734 | 9.93 | 12.26 |
| TP1 + PP2 | 0.10 | 0.0984 | 9.65 | 13.89 |
| TP1 + PP8 | 0.125 | 0.1220 | 9.72 | 13.59 |

Pipeline parallelism requires `--disable-overlap-schedule` and cannot be combined
with DFlash, which requires `pp_size == 1`. On identical work one two-GPU TP2
replica with DFlash2 sustained roughly twelve times the confirmed rate of the
two-GPU pipeline. Use `--language-only` (not `--language-model-only`) for
text-only GLM.

### Limits of this evidence

Quality was checked on a 1,351-task paired corpus with no aggregate regression
(GSM8K 1,287/1,319 versus 1,285/1,319) and on long-context retrieval and
arithmetic through 500,000 tokens, but generated-token identity is **not**
established: GSM8K lost eight answers and gained ten, and repeat controls moved
correctness on the unchanged baseline as well. Kernel checks were bitwise
(split-K cases, sparse dispatch, real-model stage checks on both ranks).

The rate measurements use a fixed-length synthetic plain-text workload with one
warmed shared prefix and forced 300-token outputs. They do not model a production
prompt distribution or JSON-schema traffic, and they are per configured server,
not capacity guarantees. See
[`docs/docs/hardware-platforms/amd_gpu.mdx`](../docs/docs/hardware-platforms/amd_gpu.mdx)
for the full configuration and caveats.

## Others
- `MAINTAINER.md` defines the code maintenance model.
