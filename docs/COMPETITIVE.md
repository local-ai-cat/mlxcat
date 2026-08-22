# Where mlxcat actually stands, and why

Derived from the 223 valid HTTP rows in `bench/results/` (M4 Pro, 48 GiB,
2026-08-21/22), newest-per-cell. Regenerate the numbers from the JSONL; do not
trust this file over the data.

## The headline

Against the best of oMLX / mlx-lm / mlx-serve, cell by cell, geometric mean:

| metric | mlxcat vs best other |
|---|---|
| prefill tok/s | **0.61×** |
| TTFT | **0.55×** |
| e2e tok/s | 0.85× |
| peak memory | 0.68× (we use ~1.5× the leanest) |

Fourth of four. But the average hides the shape, and the shape is the finding.

## The shape: we are competitive at c1 and collapse under concurrency

TTFT in ms at the short tier:

| engine | c1 | c2 | c4 | c8 | c8/c1 |
|---|---|---|---|---|---|
| **gemma-4-E2B** ||||||
| mlxcat | **69** | 891 | 2605 | 5929 | **85.8×** |
| mlx-serve | 83 | 132 | 230 | 460 | 5.6× |
| oMLX | 329 | 444 | 647 | 1247 | 3.8× |
| mlx-lm | 254 | 425 | 804 | 1571 | 6.2× |
| **Qwen3.8-27B** ||||||
| mlxcat | 3041 | 9318 | 21622 | 46199 | **15.2×** |
| mlx-serve | 2196 | 4119 | 6728 | 12121 | 5.5× |

mlxcat is the **fastest engine on the board at c1 on gemma-4-E2B** and the
slowest at c8 by 12.9×. Decode is not the problem — e2e ratios stay in the
0.6–1.1 band across the concurrency legs. This is an admission problem.

## The cause, confirmed in code

`NativeModelEngine.usesSerializedDecode` disables batched decode for two sets of
model families, and `Scheduler.admitWaiting` then returns early whenever anything
is already running. Mapping the benchmark models to their `config.json`
`model_type`:

| model | model_type | VLM | batched? |
|---|---|---|---|
| Qwen3.5-4B | `qwen3_5` | yes | **no** — scalar-offset list |
| Qwen3.8-27B | `qwen3_5` | yes | **no** — scalar-offset list |
| gemma-4-E2B | `gemma4` | yes | **no** — scalar-offset list |
| gemma-4-12B | `gemma4_unified` | yes | **no** — scalar-offset list |
| Qwen3-Coder-30B | `qwen3_moe` | no | **no** — regression list |
| gpt-oss-20b | `gpt_oss` | no | yes |

**Five of the six models on the leaderboard run with mlxcat's headline feature
switched off.** The one that does batch, gpt-oss-20b, is also the only one whose
ratios stay flat as concurrency rises (0.69 / 0.68 / 0.66 at c2/c4/c8) instead of
collapsing (gemma-4-E2B: 0.88 / 0.09 / 0.08). We are not losing at continuous
batching. We are not doing it.

VLM models additionally get `schedulerManagedTextPrefill: false`, so they lose
chunked prefill as well — the same rows, penalised twice.

The two exclusion lists are not the same kind of claim:

* **scalar-offset list** — a *correctness* workaround. Those architectures derive
  RoPE position ids, mask lengths, or shared-KV offsets from a scalar
  `cache.offset`, which is wrong for a batch of mixed offsets. Fixing it means
  per-row offsets in those model implementations. Real work, and it is the work:
  gemma-4 is the iOS flagship.
* **regression list** (`qwen2`, `qwen3`, `qwen3_moe`) — a *performance* opt-out,
  from "explicit-demand two-request benchmarks". Two requests, and if they were
  measured at a short tier then serial prefill dominated the wall and the
  experiment measured admission latency, not batched decode — which is exactly
  the trap `bench/README.md` now documents, and exactly the trap that made an
  earlier "batching is worth +22%" claim on this repo wrong. oMLX gets 1.55–3.14×
  at the longgen tier. This list deserves re-measurement before it is believed.

## Three caveats that all favour mlxcat

1. **Every mlxcat row predates the allocator fix.** They were measured at harness
   commit `e99f07c` on 08-21 13:46–15:07; `7f6cbb6` ("bound the MLX allocator")
   landed at 23:57 the same night. The memory column — mlxcat at 20.4 GiB on
   Qwen3.5-4B/16k against oMLX's 8.1 — is measuring a bug that is already fixed
   and not yet re-measured.
2. **mlxcat is the only engine running handicapped.** `--memory-ceiling-bytes`
   reaches only mlxcat's launch template (`bench/engines.json`); oMLX, mlx-lm and
   mlx-serve run with MLX's defaults, which on a 48 GiB box is roughly a 54 GiB
   cache limit. Post-fix mlxcat runs with a 4 GiB one. Buffer-cache starvation
   costs prefill throughput, which is the metric we lose. **The control arm now
   exists**: `mlxcat-defaults` is the same binary with the ceiling absent, so the
   pair measures what the memory guard costs instead of leaving it as an excuse.
   Until both arms are on the board, no single-stream gap here is settled.
3. **The prefix cache has never been measured.** All 223 rows are cold —
   `cache_mode` is null on every one of them. The tiered prefix KV cache is the
   feature mlxcat exists for and it contributes nothing to its current standing.

## What would move the number, in order

1. **Per-row offsets for the scalar-offset families**, starting with `gemma4`.
   Unlocks batching for four of six benchmark models and for the iOS flagship.
2. **Re-measure the regression list at `longgen` c2/c4/c8** and delete the
   entries that no longer earn their place.
3. **Re-run mlxcat post-allocator-fix, with and without the ceiling** — pass 3
   was written for exactly this and died in the panic.
4. **Run warm.** Nobody should treat this ranking as settled until the cache
   the engine is built around has been allowed to hit.
5. **gpt-oss-20b is our weakest model even at c1** (prefill 0.51× at 16k,
   0.66× at 4k). Batching is on there, so this one is a different defect —
   profile the MXFP4 path.
