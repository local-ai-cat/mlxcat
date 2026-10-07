# Watch learnings

What the repos in the [`docs/ENGINES.md`](../ENGINES.md) watchlist learned about
inference, read pass by pass, and what we did about it. Newest pass first. A weekly
pass follows [`CHECKER.md`](CHECKER.md) and starts from the per-repo cursors in
[`cursors.json`](cursors.json).

## Entry format

```
- **What changed, in a few words** — [repo#PR](link) or [commit](link), YYYY-MM-DD
  The mechanism in two or three plain sentences.
  *mlxcat:* has it / partial / no, with `file:line`. *Cost:* rough size and risk.
  **Verdict: port now | measure first | skip** — the reason.
```

An entry needs a link. A repo that was read and had nothing relevant still gets
a line (`Nothing relevant since <date> (read: …)`), so the next pass knows it was
read. **Port now** means the mechanism is understood, mlxcat lacks it, and it fits
as a lever under [`docs/LEVERS.md`](../LEVERS.md). **Measure first** means we need a
number or a reproduction before writing code. **Skip** says why not.

---

## Pass 2026-10-02 · window 2026-08-03 → 2026-10-02

The first pass since the 2026-08 survey. All 17 watchlist repos plus the two
upstreams of our forked pins were read, from release notes and merged PRs down
to code where the PR body was not enough. Depth per repo is recorded in
`cursors.json`: llama.cpp, vllm, LiteRT-LM and mlx core were skimmed, the rest
read.

**Ported as levers this pass** (all off by default; see `docs/LEVERS.md`):

1. `MLXCAT_ADMISSION_CLEAR_CACHE`: clear MLX's buffer cache when a prefilled row
   joins the decode batch (from mlx-swift-lm#620).
2. `MLXCAT_WATCHDOG_TRIM_AFTER_EVICT`: clear the cache again after the memory
   watchdog evicts a model, before it re-measures (from mlx-serve#637,
   Rapid-MLX#3798).
3. `MLXCAT_PREFIX_FULL_MATCH_REUSE`: reuse N-1 tokens when a stored prefix covers
   the whole prompt, instead of prefilling it cold (the rule in swama#123). It was
   found while checking vllm-mlx#714 against our own server.

**Measured, not ported:** moving off our mlx-swift fork to upstream 0.32.3
(see the first dependency entry).

### Pinned dependencies

- **mlx-swift 0.32.3 vendors an mlx that has the RoPE batch-grid fix** —
  [ml-explore/mlx-swift 0.32.3](https://github.com/ml-explore/mlx-swift/releases/tag/0.32.3), 2026-09-30
  0.32.3 vendors mlx `1f8e74e3` (2026-08-25), 406 commits after
  [mlx#3498](https://github.com/ml-explore/mlx/pull/3498) (`76a977ca`), the fix our
  `atlas-open-sources/mlx-swift` fork exists to carry. The condition written in
  `Package.swift` for returning to upstream is met. 0.32 also changes `Stream` and
  `Device` semantics: streams are pooled, and the default device and stream are
  task-local.
  *mlxcat:* pinned to the fork at `ea227ae6` (`Package.swift:92`). Tried on a
  scratch branch with upstream 0.32.3 and our one mlx-swift-lm commit
  cherry-picked onto 3.32.3. It builds unchanged. The test harness's precompiled
  kernel list needs `gemv.metal` removed, because 0.32 moved `gemv` to the JIT set
  (`Tests/MLXCatTests/Support/MLXMetalRuntime.swift:66`, `scripts/build-metallib.sh`).
  After that, 505 tests run with 1 failure:
  `PerRequestRNGTests.testMixedSeededBatchUsesIndependentPerRequestRNGState`. An
  unseeded row in a mixed batch no longer draws the same tokens as the same row
  alone under one global seed. There are no Swift-side `Random` changes between
  0.31.6 and 0.32.3, so the cause is in mlx core or the 3.32 sampling path; not
  root-caused. *Cost:* small in code, but the mlx-swift-lm fork must first
  re-point its mlx-swift dependency, and the RNG isolation failure must be
  understood.
  Bench on the trial build against the fork build (Llama-3.2-3B and Qwen3.5-4B;
  short, 4k, 4k × 4; provisional, load 8–26): TTFT and decode moved between −7 %
  and +7 % with no consistent sign, which is inside the noise at that load.
  Peak footprint at 4k was lower on every cell (Llama 4k × 4: 22.3 → 18.6 GiB;
  Qwen3.5 4k × 4: 13.1 → 11.2 GiB).
  **Verdict: measure first** — un-forking is the right direction. The RNG
  failure is the blocker. Speed needs a quiet machine, and the footprint drop
  should be confirmed there too.

- **Upstream Gemma 4 still anchors RoPE on the scalar cache offset** —
  [ml-explore/mlx-swift-lm 3.32.3](https://github.com/ml-explore/mlx-swift-lm/releases/tag/3.32.3), 2026-09-30
  `Gemma4TextAttention` in the VLM path still reads `cache?.offset`
  (`Libraries/MLXVLM/Models/Gemma4.swift:871` at 3.32.3) and declares `rope` as
  `OffsetLayer`. [#437](https://github.com/ml-explore/mlx-swift-lm/pull/437) made
  `ropeOffset` overridable, but Gemma 4 does not use it.
  *mlxcat:* carries the per-row fix in `atlas-open-sources/mlx-swift-lm`
  (`b14d62da`). It cherry-picks cleanly onto 3.32.3. *Cost:* none to keep.
  **Verdict: skip** — keep the fork for mlx-swift-lm; offering the change
  upstream is a separate decision.

- **Clear the buffer cache on the first generated token** —
  [ml-explore/mlx-swift-lm#620](https://github.com/ml-explore/mlx-swift-lm/pull/620), 2026-09-15
  `TokenIterator` checked its 256-token clear after incrementing, so the first
  clear came at token 256. A request shorter than that never cleared and left its
  prefill buffers cached (8 short requests: 7.44 GB cached vs 0.09 GB with the fix).
  *mlxcat:* partial. Back-to-back requests are covered by the drain-to-idle
  release (`Sources/MLXCat/TrackB/Scheduler.swift`, `releaseCacheIfDrained`).
  A server that never drains kept each admission's prefill scratch until the
  512-step interval, and the final chunk is deliberately not cleared
  (`Scheduler.swift`, the `chunked, nextPrefillIndex < upperBound` guard).
  *Cost:* about 30 lines.
  **Verdict: port now** — done as `MLXCAT_ADMISSION_CLEAR_CACHE`. Greedy output is
  token-identical on Llama-3.2-3B and Qwen3.5-4B. The cache left after a
  4096-token prefill drops 1804 → 948 MiB (Llama) and 1284 → 394 MiB (Qwen3.5).
  Peak footprint at 4k with 4 concurrent requests falls 17–19 % with TTFT and
  decode flat. These are provisional numbers taken on a loaded machine; see
  `docs/LEVERS.md`.

- **`RotatingKVCache.trim` corrupted a wrapped ring** —
  [ml-explore/mlx-swift-lm#584](https://github.com/ml-explore/mlx-swift-lm/pull/584), 2026-09-10
  `trim(_:)` only decremented counters, which on a wrapped ring left stale
  entries in the middle of the timeline. The fix unrolls the ring before cutting.
  *mlxcat:* our pin still has the old `trim`, but neither caller can reach a
  wrapped ring. The decode-side one-step discard is gated on
  `offset + 1 < maxSize` (`Sources/MLXCat/TrackB/BatchGenerator.swift:769`), and
  the prefix store refuses to rebuild a `RotatingKVCache` at all
  (`Sources/MLXCat/Seam/PrefixKVStore.swift`, `cache(from:)` throws
  `unsupportedCacheClass` on non-empty `metaState`). *Cost:* none.
  **Verdict: skip** — guarded by construction. It arrives with any rebase onto 3.32.

- **Qwen3.5/3.6 decode through compiled traces, fused MoE router top-k, GDN conv folded into the compiled step** —
  [#467](https://github.com/ml-explore/mlx-swift-lm/pull/467), [#469](https://github.com/ml-explore/mlx-swift-lm/pull/469), [#468](https://github.com/ml-explore/mlx-swift-lm/pull/468), 2026-09
  Single-stream Qwen3.5 decode now runs through `compile()`d traces, with a
  bit-identical fused router kernel for the MoE variants.
  *mlxcat:* no. Our batched decode calls the model directly, and compiled paths
  have bitten batch invariance before (the rejected fused-SiLU lever in
  `docs/LEVERS.md`). *Cost:* arrives with a rebase onto 3.32; adopting it in the
  batched path is a separate question.
  **Verdict: measure first** — check width-1 and width-8 token exactness before
  relying on it.

- **mlx 0.32.x decode-attention and small-M matmul kernels** —
  [mlx#4077](https://github.com/ml-explore/mlx/pull/4077) (read each K/V byte once in GQA-8 decode), [mlx#4380](https://github.com/ml-explore/mlx/pull/4380) (2-pass vector SDPA for GQA 12 and 16), [mlx#3888](https://github.com/ml-explore/mlx/pull/3888) (`gemv_wide` for fp16/bf16 matmuls of a few rows), [mlx#3842](https://github.com/ml-explore/mlx/pull/3842) (fused head_dim-256 attention on M5), 2026-08/09
  These are kernel changes in the shapes our batched decode runs: a few rows,
  GQA attention against a growing cache.
  *mlxcat:* none of them reach us until the mlx-swift pin moves (first entry).
  *Cost:* comes free with the bump.
  **Verdict: measure first** — part of the un-fork measurement.

- **mlx-swift-lm: smaller items read** —
  [#611](https://github.com/ml-explore/mlx-swift-lm/pull/611) and [#579](https://github.com/ml-explore/mlx-swift-lm/pull/579) stop the generation loop and weight loading from blocking cooperative threads;
  [#613](https://github.com/ml-explore/mlx-swift-lm/pull/613) emits only new scalars when a token extends the previous character;
  [#515](https://github.com/ml-explore/mlx-swift-lm/pull/515) reuses the KV cache for append-only media turns;
  [#548](https://github.com/ml-explore/mlx-swift-lm/pull/548) adds bounded cross-dialect tool-call recovery.
  *mlxcat:* runs its own scheduler and detokenizing path, and its own Qwen-XML
  grammar. **Verdict: skip** — these arrive with a rebase. #548 is worth a
  comparison against our tool-call parser when that is next touched.

- **swift-transformers 1.3.4** —
  [release](https://github.com/huggingface/swift-transformers/releases/tag/1.3.4), 2026-09-02
  Jinja 2.4.2 and a fix for sampling with vocabularies over 65,536
  ([#384](https://github.com/huggingface/swift-transformers/pull/384)).
  *mlxcat:* we sample in `Sources/MLXCat/TrackB/Sampling.swift` on MLX and do not
  use swift-transformers' `Generation` module. Our fork is one commit ahead
  (download progress) and ten behind upstream. **Verdict: skip** — rebase the fork
  when the Jinja update is wanted.

- **WhisperKit: incremental audio loading** —
  [argmaxinc/argmax-oss-swift#507](https://github.com/argmaxinc/argmax-oss-swift/pull/507), 2026-08-06 (v1.1.0)
  `audioLoadingMode: .incremental` streams a file in bounded chunks cut at VAD
  silences, reporting over 70 % lower peak memory on a 3-hour file. The repository
  was renamed to `argmax-oss-swift`; the old URL redirects.
  *mlxcat:* we pin `dcf3a00f` (2026-07-01), before v1.1.0, and transcribe whole
  files (`Sources/MLXCatSpeechWhisperKit/WhisperKitAdapter.swift:79`). *Cost:* a
  pin bump plus one option.
  **Verdict: measure first** — worth it only if long recordings are in scope on iOS.

### Engines we benchmark

**jundot/omlx** (Apache-2.0, v0.7.0, 2026-09-30)

- **Refresh the memory sample right before the final admission check** —
  [omlx#4124](https://github.com/jundot/omlx/pull/4124), 2026-09-30
  After evicting to make room, admission re-checked the stale pre-eviction sample
  and rejected a request that now fit.
  *mlxcat:* same class, different shape. Our watchdog does re-sample after
  evicting, but the evicted weights are still on MLX's free list when it does
  (`Sources/MLXCat/Pool/MemoryWatchdog.swift`, `poll()` and `checkAdmission`).
  **Verdict: port now** — combined with mlx-serve#637 below as
  `MLXCAT_WATCHDOG_TRIM_AFTER_EVICT`.
- **Partial-block prefix reuse** —
  [omlx#3835](https://github.com/jundot/omlx/pull/3835), 2026-09-24
  Prefix reuse now also matches the trailing partial block, so a follow-up
  re-prefills tens of tokens instead of up to a block (1,174 → 37 tokens on a
  13.4K prompt).
  *mlxcat:* has it on the production path. The native engine's prefix store
  matches token by token (`Sources/MLXCat/Seam/SessionPrefixKVStore.swift`,
  `commonPrefixLength`). Only the older block cache
  (`Sources/MLXCat/TrackA/BlockAwarePrefixCache.swift:91`) is whole-block.
  **Verdict: skip.**
- **Memory guard reserves as a percentage of RAM** —
  [omlx#3933](https://github.com/jundot/omlx/pull/3933), 2026-09-30
  Tiers now reserve about 20 / 8 / 2 % of RAM (clamped) instead of fixed bytes.
  *mlxcat:* fixed 8 / 6 / 4 GiB reserves (`Sources/MLXCat/Pool/MemoryGuard.swift:14-16`).
  *Cost:* about 50 lines in one file.
  **Verdict: measure first** — needs a large-RAM and a 16 GiB machine to price.
- **Prune recurrent-state tails in the SSD cache for hybrid models** —
  [omlx a28e5a8](https://github.com/jundot/omlx/commit/a28e5a8), 2026-09-30
  One full recurrent tail per turn was kept, so SSD usage grew without bound.
  *mlxcat:* hybrid checkpoints live in RAM slots in `SessionPrefixKVStore`. The
  native engine has no SSD tier; `TrackA/PagedSSDCacheManager.swift` is not
  referenced outside its own file. **Verdict: skip** — revisit if prefix state is
  ever spilled to SSD.
- **XML tool-call parsing: CDATA, dotted names, fragmented arrays** —
  [omlx#3530](https://github.com/jundot/omlx/pull/3530), 2026-09-18
  *mlxcat:* our Qwen-XML grammar has no CDATA case
  (`Sources/MLXCat/TrackB/Grammar/QwenXMLToolGrammar.swift`). **Verdict: measure
  first** — only if a served model is seen emitting CDATA.
- Noise: kernel work for model families we do not serve (Qwen3.8-Flash-Next
  QSA, GLM-5.3, MiMo), cluster serving and app UI.

**ml-explore/mlx-lm** (MIT; no release since v0.31.3, main active)

- **Read batch left padding on the host, not with a reduction** —
  [mlx-lm#1824](https://github.com/ml-explore/mlx-lm/pull/1824), 2026-09-04
  `left_padding.min().item()` launched a reduction and waited on it for a value
  the host already knew, once per layer per `filter`. +11.6 % decode on a
  32-wide batch of a 0.5B model.
  *mlxcat:* same pattern in `BatchKVCache.filter`
  (`Sources/MLXCat/TrackB/BatchCache.swift:237`). But it runs only when a row
  retires, not every step, and our padding array is lazy after `take`, so a host
  read still waits on the GPU. A real fix keeps a host-side mirror of the padding.
  **Verdict: measure first** — at our widths (≤ 8) the saving is about one sync
  per layer per retirement.
- **ChunkedKVCache dropped live tokens by comparing buffer size, not token count** —
  [mlx-lm#1673](https://github.com/ml-explore/mlx-lm/pull/1673), 2026-08-22
  *mlxcat:* `BatchKVCache` trims by per-row valid lengths, not buffer shape.
  **Verdict: skip.**
- **KV quantization flags in the server** —
  [mlx-lm#1832](https://github.com/ml-explore/mlx-lm/pull/1832), 2026-09-09
  *mlxcat:* has it (`MLXCAT_KV_BITS`, `docs/LEVERS.md`). **Verdict: skip.**
- Noise: new-model onboarding and LoRA training fixes.

**waybarrios/vllm-mlx** (Apache-2.0, v0.5.0)

- **The prefix cache silently stopped hitting after the first tool-call turn** —
  [vllm-mlx#714](https://github.com/waybarrios/vllm-mlx/pull/714), 2026-08-22
  The eligibility probe matched raw messages, so after a tool-call message the
  ~20k-token system and tool prompt was re-prefilled every turn, with no error.
  *mlxcat:* checked against `mlxcat-http` with a ~2.8k-token system and tool
  prompt, then an assistant tool call, a tool result and a follow-up. There is no
  silent miss. Llama-3.2-3B reuses the whole previous prompt on each turn (3032,
  then 3069 tokens). Qwen3.5-4B resumes from its 2560-token grid checkpoint and
  re-prefills about 200–300 tokens, as designed. The check did expose a
  different gap: an *exact* repeat of a prompt reused nothing (`cached_tokens` 0),
  because a full match was released and prefilled cold (`Scheduler.swift`,
  `matchedTokenCount == promptTokens.count`).
  **Verdict: skip** for the tool-turn bug. The exact-repeat gap is ported as
  `MLXCAT_PREFIX_FULL_MATCH_REUSE` (swama#123 below).
- **A metadata graph never evaluated leaked Metal handles until a crash after 19 h** —
  [vllm-mlx#708](https://github.com/waybarrios/vllm-mlx/pull/708), 2026-09-30
  `filter`/`extend` rebuilt offsets and padding as lazy ops that were never
  evaluated, until Metal's handle limit (a count, not bytes) was hit.
  *mlxcat:* `BatchKVCache.filter` rebuilds `leftPadding` lazily too
  (`Sources/MLXCat/TrackB/BatchCache.swift:229`). Whether it is forced each step
  is not established. **Verdict: measure first** — a soak test of many
  admit/retire cycles watching the handle count.
- **A bounded KV size that the validator rejected, so generation ran unbounded** —
  [vllm-mlx#688](https://github.com/waybarrios/vllm-mlx/pull/688), 2026-08-26
  **Verdict: skip** — a lesson for tests: assert a configured bound actually trims.
- Noise: Python server internals, API compatibility shims.

**ddalcu/mlx-serve** (its LICENSE file is MIT text, though GitHub does not detect it; v26.10.1)

- **Eviction cleared the registry but not MLX's allocator cache, so loads failed until restart** —
  [mlx-serve#637](https://github.com/ddalcu/mlx-serve/pull/637), 2026-09-29
  About 3.2 GB stayed in MLX's cache after a make-room eviction. Six loads in a
  row failed until the process restarted.
  *mlxcat:* yes, same gap. `EnginePool.unloadLoadedModel` passes the
  still-referenced engine to the loader, whose `Memory.clearCache()` runs before
  the weights are released (`Sources/MLXCat/Pool/EnginePool.swift:545`,
  `Sources/MLXCatNative/NativeModelEngine.swift:910`). The watchdog then
  re-samples `active + cache` without clearing. Measured through the real pool,
  that sum reads the same after eviction as before (Llama-3.2-3B 1723 → 1723 MiB,
  all of it now cached; Qwen3.5-4B 2894 → 2894) and drops to 0 only after the
  next clear.
  **Verdict: port now** — done as `MLXCAT_WATCHDOG_TRIM_AFTER_EVICT`.
- **Decode streams keep ticking during another request's long prefill** —
  [mlx-serve#568](https://github.com/ddalcu/mlx-serve/pull/568), 2026-09-26
  Decode got about 3 % of wall time during a 32k prefill. The fix keeps ticking
  until decode reaches a share of chunk time.
  *mlxcat:* busy prefill uses 2048-token chunks with a decode tick at each
  boundary (`Sources/MLXCat/TrackB/Scheduler.swift`, `busyPrefillStep`).
  **Verdict: measure first** — inter-token latency of a running stream while a
  16k prompt is admitted.
- **SSD-restored prefix rows not credited as owned** —
  [mlx-serve#621](https://github.com/ddalcu/mlx-serve/pull/621), 2026-09-30
  **Verdict: measure first** — a lead for the open SSD-tier prefix entry in
  `docs/KNOWN-FAILURES.md`.
- **Prompt-lookup drafts need exact acceptance, or agents loop** —
  [mlx-serve#614](https://github.com/ddalcu/mlx-serve/pull/614), 2026-09-28
  *mlxcat:* we speculate only when greedy and one row is running
  (`Sources/MLXCat/TrackB/BatchGenerator.swift:957`), so acceptance is exact.
  **Verdict: skip.**
- Noise: QSA, MoE and GDN prefill kernels for models we do not serve (+2.6 to +22.6 %
  on Qwen3.8-Flash-Next), app UI, image generation.

**lmstudio-ai/mlx-engine** (MIT, rolling main)

- **Shrink the prefill step automatically to fit a longer context** —
  [mlx-engine#367](https://github.com/lmstudio-ai/mlx-engine/pull/367), 2026-08-21
  *mlxcat:* fixed 512 idle / 2048 busy chunks, overridable per request.
  **Verdict: measure first.**
- **Gemma 4 image prefill bypassed the chunk cap** —
  [mlx-engine#362](https://github.com/lmstudio-ai/mlx-engine/pull/362), 2026-08-19
  A model-specific path made one 48.7 GB allocation. **Verdict: skip** — a lesson:
  model-specific prefill paths must honour the chunk budget.
- Nothing else relevant since 2026-08-03 (read: all 7 merged PRs).

**ollama/ollama** (MIT, v0.35.1-rc2; MLX backend paths read)

- **Clear the cache when a multi-token round crosses the interval, not only on an exact count** —
  [ollama#18510](https://github.com/ollama/ollama/pull/18510)
  *mlxcat:* immune. The interval counts `step()` calls with `>=`
  (`Sources/MLXCat/TrackB/Scheduler.swift`, `decodeStepsSinceCacheClear`).
  **Verdict: skip.**
- **Cancelled prefills keep their progress; restore points span exactly their edge** —
  [ollama#17901](https://github.com/ollama/ollama/pull/17901)
  Agent clients cancel long prefills, so retries that started from zero never
  finished. **Verdict: measure first** — cancel a chunked hybrid-model admission
  midway and check whether the retry reuses the completed prefix.
- **Bound model loads by system free memory, not only our own accounting** —
  [ollama#18345](https://github.com/ollama/ollama/pull/18345)
  *mlxcat:* the pool uses physical memory and its own accounting
  (`Sources/MLXCat/Pool/MemoryGuard.swift`). **Verdict: measure first** — this
  matters most on iOS.
- **Lazy weight folds, materialised by CPU reads before any GPU graph** —
  [ollama#17998](https://github.com/ollama/ollama/pull/17998)
  Metal command buffers waiting on slow storage were killed mid-load, and MoE
  folds briefly held twice the expert weights. **Verdict: measure first** — load
  from a throttled volume once.

**ggml-org/llama.cpp** (MIT; Metal, speculative and server titles skimmed)

- **Re-sort hidden states before the drafter when a batch is split** —
  [llama.cpp#29019](https://github.com/ggml-org/llama.cpp/pull/29019)
  **Verdict: skip** — not live for us (single-row speculation). It is a design
  constraint for the day speculation goes multi-row.
- **Metal 4 tensor API detection on M5 / A19** —
  [llama.cpp#27461](https://github.com/ggml-org/llama.cpp/pull/27461)
  **Verdict: skip** — MLX handles this inside its kernels (`mlx#3842` above).

**raullenchai/Rapid-MLX** (Apache-2.0 per its LICENSE file, v0.15.3)

- **Cap prompt tokens before any GPU work** —
  [Rapid-MLX#3835](https://github.com/raullenchai/Rapid-MLX/pull/3835)
  A prompt inside the model's limit but beyond the machine's prefill envelope
  crashed an unattended node.
  *mlxcat:* only the model's context window is checked
  (`Sources/MLXCatNative/NativeModelEngine.swift:140`). **Verdict: measure first** —
  it needs a number for the envelope per machine before a cap means anything.
- **Flush the allocator cache after each prefix eviction before re-measuring** —
  [Rapid-MLX#3798](https://github.com/raullenchai/Rapid-MLX/pull/3798)
  **Verdict: port now** — the same fix as mlx-serve#637; ported once.
- **Row-invariant lane matmul for batched decode (opt-in)** —
  [Rapid-MLX#3931](https://github.com/raullenchai/Rapid-MLX/pull/3931)
  It reports 1.77× aggregate at batch 8 on an M4. **Verdict: skip** — a custom
  kernel per chip family; revisit if batch-8 decode becomes the constraint.

**Trans-N-ai/swama** (MIT, v2.4.0)

- **Reuse at most N-1 tokens of a cached prompt** —
  [swama#123](https://github.com/Trans-N-ai/swama/pull/123)
  A longest-common-prefix match is trimmed to `min(match, N-1)`, so even an
  identical prompt keeps its cache and only the last token is recomputed.
  *mlxcat:* no. A full match was released and the prompt prefilled cold.
  **Verdict: port now** — done as `MLXCAT_PREFIX_FULL_MATCH_REUSE`. An exact
  replay of a ~950-token prompt takes 794 → 22 ms on Llama-3.2-3B and
  1154 → 550 ms on Qwen3.5-4B. Greedy output equals the cold run. These are
  provisional numbers taken under load.
- **Log a cache diagnostic on every request, including the miss reason** —
  same PR
  *mlxcat:* prefix debug logs hits and checkpoints behind a flag
  (`Sources/MLXCat/TrackB/Scheduler.swift:1153`); misses carry no reason.
  **Verdict: measure first** — cheap, and it is what would catch a vllm-mlx#714-style miss.
- **FIFO waiters instead of a poll loop for model slots** —
  [swama#164](https://github.com/Trans-N-ai/swama/pull/164)
  **Verdict: skip** — our pool queues load waiters on continuations.

**SharpAI/SwiftLM** (MIT, b795)

- **Count only layers that hold KV in the load-time memory estimate** —
  [SwiftLM#173](https://github.com/SharpAI/SwiftLM/pull/173), 2026-09-05
  A hybrid model was estimated at 16.1 GB of KV against 1.6 GB actual, which
  sent the load down a path that crashed.
  **Verdict: measure first** — check our estimate on Qwen3.5 (hybrid) against
  the measured footprint.
- **Hybrid prompt cache at exact turn boundaries** —
  [SwiftLM#189](https://github.com/SharpAI/SwiftLM/pull/189)
  *mlxcat:* has a more general version (grid checkpoints, on by default since
  `58bed88`). **Verdict: skip.**

### Rest of the watchlist

- **Blaizzy/mlx-vlm** (MIT, v0.7.4): phantom KV from zero-filled rejected tokens
  in ragged speculative rollback ([mlx-vlm#2113](https://github.com/Blaizzy/mlx-vlm/pull/2113),
  2026-08-31). A zeroed key still adds `e^0` to the softmax denominator. Not live
  for us (single-row speculation); **skip**, but carry it into any multi-row
  speculation design. Shared-prefix APC with disk snapshots
  ([#2182](https://github.com/Blaizzy/mlx-vlm/pull/2182)): **skip**, ours is
  equivalent. Much per-model drafter work; noise otherwise.
- **john-rocky/apple-silicon-llm-bench** (MIT): thermal-nominal gating, warm and
  cold reported separately, and no ratios across capture sessions
  ([README](https://github.com/john-rocky/apple-silicon-llm-bench#readme)).
  *mlxcat:* `bench/run.py` has a load guard and settles but no thermal gate or
  session id. **Measure first** — a methodology change for `bench/METHODOLOGY.md`.
  Development has moved to `edge-llm-bench` (now watched).
- **google-ai-edge/LiteRT-LM** (Apache-2.0, v0.17.1; release notes only): Metal
  residency and "optimized local attention" ([v0.17.0](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.17.0)).
  **Skip**: not enough detail to act on.
- **vllm-project/vllm** (Apache-2.0, v0.30.0; release notes only): adaptive draft
  length from an online acceptance estimate ([v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)),
  hybrid-model prefill checkpoints for prefix caching (9–25 % TTFT), queue
  admission caps. **Measure first** on adaptive draft length: we already keep
  speculative acceptance stats.
- Not read this pass: exo, mlx_sharding, mlx-omni-server, mlx-openai-server,
  fastmlx, PicoMLXServer, executorch, mlc-llm, Anemll. They are class
  representatives or out of scope; give them cursors when a pass reads them.

### Proposed ports (not done this pass)

1. **Un-fork mlx-swift** once the RNG isolation failure is explained: re-point the
   mlx-swift-lm fork at upstream mlx-swift 0.32.3, drop `gemv.metal` from the
   precompiled list, and re-run the parity matrix.
2. **A prefix-miss reason in the prefix debug log** (swama#123), as the cheap
   instrument for the vllm-mlx#714 check.
