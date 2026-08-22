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

## Three caveats — one disproven, one resolved, one still open

1. **Every mlxcat row on the board predates the allocator fix.** They were measured at harness
   commit `e99f07c` on 08-21 13:46–15:07; `7f6cbb6` ("bound the MLX allocator")
   landed at 23:57 the same night. The memory column — mlxcat at 20.4 GiB on
   Qwen3.5-4B/16k against oMLX's 8.1 — is measuring a bug that is already fixed
   and not yet re-measured.
2. ~~**mlxcat is the only engine running handicapped.**~~ **Disproven
   2026-08-22.** It was true that `--memory-ceiling-bytes` reaches only mlxcat's
   launch template while oMLX, mlx-lm and mlx-serve run at MLX's defaults
   (~54 GiB cache limit on a 48 GiB box against our 4 GiB), and it was a
   reasonable suspicion that buffer-cache starvation cost us prefill. It does
   not. `mlxcat-defaults` — the same binary with the ceiling absent — came out
   within noise of `mlxcat` on every metric of both models:

   | gemma-4-E2B longgen c1 | prefill | decode | peak |
   |---|---|---|---|
   | mlxcat (24 GiB ceiling) | 4707 | 80.8 | 5.80 GiB |
   | mlxcat-defaults (no ceiling) | 4701 | 81.0 | 5.68 GiB |

   The memory guard is not what is costing us. That excuse is gone, and the
   remaining gap is the serialization exclusion in item 1.
3. ~~**The prefix cache has never been measured.**~~ **Measured 2026-08-23, and
   it is not paying off.** The first warm-vs-cold run in this repo's history:

   | cell | cold TTFT | warm TTFT | |
   |---|---|---|---|
   | gemma-4-E2B short | 81 ms | 72 ms | 1.13× |
   | gemma-4-E2B 4k | 918 ms | 897 ms | 1.02× |
   | Qwen3.5-4B short | 299 ms | 182 ms | 1.64× |
   | Qwen3.5-4B 4k | 3,694 ms | **4,410 ms** | **0.84× — slower** |

   gemma-4 is expected: `prefixCacheEnabled` excludes windowed KV caches, so it
   has no prefix cache at all. Qwen3.5 is not. And warm costs memory — peak rises
   1.3–1.8 GiB, because the store holds ~200 MB per model even on a 0.6B.

   This is a **performance** finding, not a broken feature. `PrefixCacheHitRateTests`
   proves the cache genuinely hits on the anonymous-slot path real traffic uses
   (3 identical prompts → 2 fetch hits, 6 stores, 0 evictions), and
   `TrackAPrefixCacheTests` proves reconstruct is exact. It reuses correctly and
   the reuse does not currently pay: reconstruct costs about what prefill costs
   at these sizes, so the cache buys memory pressure and little else. Where it
   should win — long shared prefixes — is exactly where the 4k row went negative.

   That is the honest state of mlxcat's headline feature, and it is the first
   time anyone could say either way.

   The original caveat, for the record: all 223 rows were cold —
   `cache_mode` is null on every one of them. Not because warm was never
   scheduled: `--cache-modes` was parsed, printed in the plan, documented at
   length, and bound to nothing — `cache_mode` was read four times inside the
   cell loop and assigned nowhere, so the first run to reach that line died with
   a `NameError` (`666b240`). The tiered prefix KV cache is the feature mlxcat
   exists for, no run has ever been able to exercise it, and it contributes
   nothing to the current standing.

## What has actually been ruled in and out

| suspicion | verdict |
|---|---|
| the memory ceiling handicaps us | **disproven** — arms within noise |
| the sliding-window mask is wrong | **disproven** — gpt-oss passes while crossing its window |
| per-row RoPE offsets are scalar in a batch | **disproven** — resolves to `.batch` (`BatchRoPEOffsetTests`) |
| ragged rows break batched decode | **disproven** — ragged rows match over 24 steps; identical rows diverge over 64 |
| the serialization exclusions cost us concurrency | **confirmed for gemma4** — 50× TTFT, 2.17× aggregate at c4 |
| ...for every excluded family | **no** — `qwen3_5` returns zero tokens and `qwen3_moe` breaks the logit tolerance at width ≥4; both exclusions are load-bearing |
| the exclusions' stated reasons are accurate | **no** — `qwen3_moe`'s says "lower throughput" and throughput is 82% higher; it is excluded for numerics instead |

## Measured: what lifting the exclusion buys

A/B on an M5 Max, 2026-08-23, same binary and same 24 GiB ceiling; the only
difference is `MLXCAT_UNSERIALIZE_MODEL_TYPES`. longgen tier.

**gemma-4-E2B — the exclusion was costing us everything:**

| | serialized (ships today) | batched | |
|---|---|---|---|
| c4 TTFT | 32,425 ms | **640 ms** | **50.7× faster** |
| c4 prefill | 33 tok/s | **1,599 tok/s** | 48× |
| c4 aggregate | 36.7 tok/s | **79.5 tok/s** | **2.17×** |
| c8 TTFT | 52,830 ms | **1,259 ms** | **42× faster** |
| c8 prefill | 20 tok/s | **813 tok/s** | 41× |
| c8 aggregate | 43.4 tok/s | **85.2 tok/s** | 1.96× |

53 seconds to first token at c8 becomes 1.3 seconds, and aggregate throughput
roughly doubles — inside the 1.55–3.14× band oMLX gets at this tier. Per-request
decode falls (45.5 → 33.9 tok/s at c4), which is what batching is supposed to
trade.

**Qwen3.5-4B — the exclusion is load-bearing:**

Both concurrency cells failed with `no visible output`. The server logged no
error; the request simply completed empty. That is the same defect
`HybridBatchIntegrationTests` reports — `Optional([]) != Optional([760, 1156])`,
alongside `BatchKVCache cannot combine KVCacheSimple cache layout with existing
ArraysCache layout`. Batched decode over a hybrid cache silently produces
nothing.

This is why the A/B mattered and the logit gate alone was not enough: Qwen3.5
**passed** logit invariance at batch 2/4/8 (0.0/0.38/0.66, cleaner than the
reference model) and still returns no tokens under real concurrency. Four decode
steps do not reach the hybrid-cache path.

### Shipped: `.multimodalOnly`

The blocker below was real — batching gemma-4's **images** genuinely breaks, and
the VLM gate that proves it had to be unpinned from `qwen2_vl` first:

| policy | vlm-0 (96×96) | vlm-1 (160×112) | vlm-2 (112×160) |
|---|---|---|---|
| batch everything | OK 8/8 | **FAIL** 8/9 | **FAIL** 8/24 |
| `.multimodalOnly` | OK 8/8 | **OK 9/9** | **OK 24/24** |

All three ragged image rows stop at the same early token when batched — exactly
the scalar-offset defect the exclusion was written for. So the exclusion was
right, and far broader than the defect: it was serializing on image
*capability* when the defect is about image *content*, and `isVLM` comes from the
model directory, so a gemma-4 deployment that never sends an image paid the full
cost anyway.

`SerializationPolicy.multimodalOnly` gives any row carrying image/video/audio the
batch to itself and lets text rows batch. Default behaviour on the default path,
no env override — measured again with nothing set:

| gemma-4-E2B longgen | before | after |
|---|---|---|
| c4 TTFT | 32,425 ms | **641 ms** |
| c4 aggregate | 36.7 tok/s | **82.8 tok/s** |
| c8 TTFT | 52,830 ms | **1,272 ms** |
| c8 aggregate | 43.4 tok/s | **78.5 tok/s** |

### The original blocker, for the record

`gemma4` and `gemma4_unified` should come off `scalarOffsetVLMModelTypes`. The
evidence is a 50× TTFT improvement, a 2× throughput improvement, and logit-level
invariance cleaner than the model the gate was pinned to.

**Held, not shipped, for one reason:** the exclusion's stated cause is MRoPE
position ids derived from a scalar `cache.offset`, which is an *image* concern,
and every measurement above is text-only. There is no gate that batches gemma-4
with image input — `MLXSERVE_VLM_TEST_MODEL` only drives
`ModelCacheCapabilitiesTests` and `ModelDiscoveryTests`, neither of which runs
inference. Flipping the default on text evidence alone would be exactly the move
this file keeps criticising. The unlock is a one-line change once a VLM batch
gate exists.

`qwen3_5` stays excluded, and now has a reason on file rather than an assumption.

## What would move the number, in order

1. ~~**Per-row offsets for the scalar-offset families**, starting with `gemma4`.~~
   **Done for gemma4** — see `.multimodalOnly` above; text rows batch, image rows
   do not, and both are gated. Still open for `qwen3_5`, where the blocker is the
   hybrid cache returning no tokens rather than the offsets.
   Unlocks batching for four of six benchmark models and for the iOS flagship.
   Start at `docs/KNOWN-FAILURES.md` §1, not at the exclusion list: the gate that
   would prove batched decode correct for gemma is **red** — 89/160 mismatched
   tokens with a sustained run of 87 past the sliding window, reproducible on two
   machines and at the pre-branch commit. The exclusion is not conservatism.
2. ~~**Re-measure the regression list at `longgen` c2/c4/c8**~~ **Done for
   `qwen3_moe`, and the answer was double-edged.** Its stated reason — "lower
   aggregate throughput under batched decode" — is refuted: Qwen3-Coder-30B at
   longgen c4 goes 30.0 → **54.5 tok/s aggregate** batched (82% higher) and
   44,760 → **4,089 ms** TTFT. The original claim came from two requests at a
   short tier, i.e. it measured admission latency, which is the trap this file
   keeps finding. But the exclusion **stays**, on a correctness ground nobody had
   checked: `maxLogitError` is 2.69 at batch ≥4 against a 1.25 tolerance
   (gpt-oss-20b, batching enabled, measures 1.7e-05 at the same widths). A 10.9×
   TTFT and 1.82× throughput win is waiting behind the numerics. `qwen2` and
   `qwen3` are still unmeasured on both axes.
3. **Re-run mlxcat post-allocator-fix, with and without the ceiling** — pass 3
   was written for exactly this and died in the panic.
4. **Run warm.** Nobody should treat this ranking as settled until the cache
   the engine is built around has been allowed to hit.
5. **gpt-oss-20b is our weakest model even at c1** (prefill 0.51× at 16k,
   0.66× at 4k). Batching is on there, so this one is a different defect —
   profile the MXFP4 path.
