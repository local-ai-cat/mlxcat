# Known failing gates

**Status 2026-08-23.** Of the three suites that were red on 2026-08-22:
`SlidingWindowBatchIntegrationTests` is **3/3 green** on gpt-oss-20b (the model
the gate was written for), `HybridBatchIntegrationTests` is **2/2 green**, and
`TrackAPrefixCacheTests` remains red — a fixture that cannot find wide-margin
suffixes, not an engine defect. Details below; §1b is resolved, §2 is fixed.

Measured 2026-08-22 on an M4 Pro (48 GiB) and an M5 Max, and again at `40b4cf5`
— the commit before the `MLXServe` → `MLXCat` rename, which is the only commit on
`feat/oss-hygiene-bench` that touches `TrackA`/`TrackB`/`Seam`.

**All three fail identically at both commits, on both machines, with byte-identical
assertion messages and mismatch counts.** They are pre-existing and deterministic,
not environmental and not introduced by the benchmark branch.

They had not been run since 2026-08-12. `swift test` skips them (their env gates
are unset), hosted CI cannot run them at all (no weights, no Metal), and
`scripts/nightly-models.sh` was reporting the wrong summary line — three gates
printed **PASS — 0 tests in 0 suites** over runs that had executed real tests and
failed. Fixed in `c8c76cf`; a gate that runs zero tests is now its own result.

Reproduce:

```bash
R=~/Library/Caches/models/mlx-community
swift build --build-tests
MLXSERVE_TEST_MODEL=$R/Qwen3-0.6B-4bit MLXSERVE_DEBUG_PREFIX_GATE=1 \
  swift test --skip-build --filter TrackAPrefixCacheTests
MLXSERVE_SLIDING_TEST_MODEL=$R/gemma-4-E2B-it-qat-4bit \
  swift test --skip-build --filter SlidingWindowBatchIntegrationTests
MLXSERVE_HYBRID_TEST_MODEL=$R/Qwen3.5-4B-MLX-4bit \
  swift test --skip-build --filter HybridBatchIntegrationTests
```

---

## 1. Batched decode diverges from serial — and it is not the sliding window

**Resolved to a cause on 2026-08-22. Read this section before acting on the raw
failure below.**

The gate is named "beyond window" and was written for gpt-oss-20b, whose sliding
window is 128 — but `scripts/nightly-models.sh` wired
`MLXSERVE_SLIDING_TEST_MODEL` to gemma-4-E2B, whose window is **512**. The gate
generates 160 tokens from a ~15-token prompt, so at ~175 total context the window
was never crossed. For its entire life this has been an ordinary
batched-vs-serial comparison wearing a sliding-window test's name.

Measured:

| probe | result |
|---|---|
| gpt-oss-20b (window 128), same gate | **passes**, genuinely crossing its window |
| gemma-4 width-1 batch vs serial | 0 mismatches |
| gemma-4, 2 ragged rows, 24 steps | 0 mismatches |
| gemma-4, 2 **identical** rows, 64 steps | 27 mismatches each |

So: the sliding-window machinery is sound, the defect is not raggedness, and it
accumulates with generation length — the signature of floating-point noise
amplified by greedy argmax, not a logic error. `GemmaBatchDivergenceProbeTests`
keeps those probes.

What settles it is the logit level, and that gate had never been run here.
`BatchInvarianceTests` compares logits and only asserts token equality where the
top-1/top-2 margin is wide — and it was **hard-pinned to Qwen3-0.6B-4bit**, a
model on neither exclusion list. The exclusion and its evidence had never
intersected. Lifting the pin (`MLXCAT_BATCH_INVARIANCE_MODELS`):

| model | batch 2 / 4 / 8 maxLogitError | mismatched | on the exclusion list? |
|---|---|---|---|
| gemma-4-E2B | 0.69 / 0.69 / 0.95 | 0 | **yes** |
| Qwen3.5-4B | 0.0 / 0.38 / 0.66 | 0 | **yes** |
| Qwen3-0.6B | 0.0 / 1.19 / 1.19 | 0 | no — the pinned reference |

Both excluded families are cleaner than the model the gate was pinned to, well
under the 1.25 tolerance. `MLXCAT_UNSERIALIZE_MODEL_TYPES` and the
`mlxcat-batched` bench arm exist to A/B what lifting them buys on real workloads.

Two follow-ups landed with this: the gate now **fails loudly** when the model's
window is wider than the context it generates, rather than passing vacuously; and
`nightly-models.sh` prefers gpt-oss-20b for the sliding gate and runs
cross-family invariance so the exclusions have to be re-earned.

### The raw failure, for reference

`SlidingWindowBatchIntegrationTests.testSlidingWindowBatchMatchesSerialBeyondWindow`
(gemma-4-E2B), 3 tests / 5 failures:

```
sliding-long-0 mismatch count 89/160 exceeds margin 4; sustained run of 87;
               first mismatches: [70,71,73,74,75,76,77,78,79,80]
sliding-long-1 mismatch count 109/160 exceeds margin 4; sustained run of 83;
               first mismatches: [42,43,44,45,46,47,48,49,50,51]
```

A sustained run of 83–87 consecutive mismatched tokens is not numerical drift.
Past the sliding window the batched path produces different text from the serial
path, and it keeps producing it.

`testSlidingWindowInsertRemoveExtractMidBatch` also fails (`1 != 2`), so
mid-batch row removal on a windowed cache is losing a row.

**This is the evidence behind `usesSerializedDecode`.** `docs/COMPETITIVE.md`
records that five of six benchmark models run with batched decode disabled, and
that mlxcat's TTFT consequently scales 15–86× from c1 to c8 where competitors
scale 4–9×. For the `gemma4` family the exclusion is not conservatism — the gate
that would prove batching correct is red, and this is what it says. Any work on
"unlock batching for gemma-4" starts here, not at the exclusion list.

## 1b. "A row is lost by insert + remove + extract" — RESOLVED: the gates were wrong

Separated out on 2026-08-23; resolved the same day. `next()`'s contract is
**"responses for whatever advanced this step"**, keyed by uid — not "one
response per active row per step". No row is lost: one inserted while a
launched-ahead step is outstanding joins the step that same `next()` call
launches, so its token arrives exactly one call later, and every subsequent
call covers the full batch. The two gates asserted an invariant the generator
never promised; they now assert liveness across two calls (both remaining rows
keep producing, the removed row emits nothing), and
`InsertRemoveExtractProbeTests` pins the one-call-offset cadence explicitly.

What decided it (all in-tree evidence, not preference):

* The failing assertions predate pipelining — the gates landed 2026-07-04
  (`4772388`), pipelining 2026-08-12 (`96255a9`) — so `count == 2` encoded the
  old synchronous cadence, which was incidental, never promised. The gates are
  model-gated and did not run when pipelining landed.
* `Scheduler.step()`, the only production caller, already consumes `next()`
  per-uid with zero coverage assumptions; per-row-per-step was already false
  anyway: `insert` returns admission tokens outside `next()`, and a
  speculative step returns *several* responses for one row.
* Restoring per-step cadence by discarding the launch-ahead on insert/filter
  is **unsound**, not just ugly: `launchStepAheadIfSafe` gates only on
  `canPipelineDecode`, which permits seeded RNG rows (the launched sample
  already advanced their streams — a recompute would draw different numbers)
  and cache layers outside `discardOneStepIsExact` — a rotated
  `RotatingKVCache` ring (i.e. every sliding-window model this gate exists
  for) and any width-≥2 `BatchKVCache`, where `trim(1)` cannot restore the
  pre-step state. The `canPrebuildNextStep` gate documents exactly this.
* The feared "lost token" in a rollback does not exist: a launched-ahead token
  is neither returned nor recorded in `generatedTokenHistory` until
  `emitPendingStep`, so discarding it loses nothing observable — the rollback
  is blocked by cache/RNG restoration, not token loss.

The contract is now documented on `ContinuousBatchGenerator.next()` in
`Sources/MLXCat/TrackB/BatchGenerator.swift`.

## 1c. The memory ceiling is advisory, not enforcing — open

Measured 2026-08-23, gpt-oss-20b at a 16k prompt, M5 Max, with a 24 GiB ceiling
configured and confirmed applied by the server's own startup line:

```
memory ceiling: 24.00GB (override); allocator: memoryLimit 24.00GB, cacheLimit 4.00GB
peak_phys_footprint 30.12 GiB
```

**25% above the configured ceiling.** `7f6cbb6` bounded the MLX allocator's
cache — which was a real fix, the same cell peaked at 35.76 GiB before it — but
`Memory.memoryLimit` governs MLX's caching behaviour, not the process. A
configured ceiling does not cap `ri_phys_footprint`.

On a Mac that is a slow machine. mlxcat also ships **embedded in an iOS app**,
where exceeding the budget is a jetsam kill, so "advisory" is the wrong
guarantee for the platform that matters most.

gpt-oss-20b had no `MEMORY_BUDGETS` entry at all, so nothing was watching the
worst-behaved model we have. It now has one, set at 32 GiB as a regression bar
just above today's measurement — explicitly not a target. The target is at or
under the configured ceiling.

## 1d. Batched `qwen3_5` dies in the SERVER, not the engine — open, narrowed

With the exclusion lifted, the benchmark's batched `qwen3_5` arm fails: width 4
returns a completion carrying no `usage` frame, width 8 kills the server process
silently — nothing in its log, and the runaway guard never fires, so it is a hard
crash rather than a memory kill.

`HybridBatchScaleTests` was written to reproduce that from a test. It does not,
and the ruling-out is the result:

| tried | outcome |
|---|---|
| 4 rows × 96 generated tokens | clean |
| 8 rows × 512 generated tokens | clean |
| + a `SessionPrefixKVStore`, which the server always has and the in-process engine does not | clean |
| + a ~1300-token prompt, so chunked prefill engages and rows span many cache blocks | clean |

So **batched decode over a hybrid cache is not the defect**. Everything the
engine does — merge, extract, chunked prefill, prefix store, `.never` policy at
width 8 — is exercised and correct. The failure lives above it, in what the
server adds: the SSE streaming path, `NativeModelEngine`'s chat/tool/stop
handling, `EnginePool`, or concurrent HTTP admission.

That is where to look next, and it is a much smaller place than "hybrid batching
is broken", which is what the bench failure looked like on its own. The gate is
kept as a regression pin for the layer that has been cleared.

`qwen3_5` covers Qwen3.5-4B **and** Qwen3.8-27B — two of the six benchmark
models, both serialized today at 37 s TTFT under load on an M4 Pro. It is the
largest single win left.

## 2. Hybrid caches cannot be combined mid-batch — FIXED

`BatchLayerCache.extract` built a fresh `KVCacheSimple` and copied the row's
state into it, discarding what the layer actually was. Harmless when every layer
is a `KVCacheSimple`; fatal for a hybrid model, where a layer may be an
`ArraysCache` — and continuous batching extracts and re-merges on every insert
and remove, so the next merge threw. `extract` now copies the concrete cache and
slices that, routing `ArraysCache` through its own `filter(batchIndices:)`.

`testHybridBatchMatchesSerialGreedyTokens` — the `Optional([])` one, and the
reason batched `qwen3_5` returned no tokens at all under real concurrency — now
passes. The other test in the suite fails later, at §1b above.

### The original failure, for reference

`HybridBatchIntegrationTests` (Qwen3.5-4B), 2 tests / 2 failures:

```
BatchLayerCache.swift:223: caught error: "BatchKVCache cannot combine
  KVCacheSimple cache layout with existing ArraysCache layout."
testHybridBatchMatchesSerialGreedyTokens: Optional([]) != Optional([760, 1156])
```

The first is a thrown layout error (XCTest counts it "1 unexpected"); the second
means the batched path returned **no tokens at all**. `qwen3_5` is also on the
scalar-offset serialization list, so the same relationship holds as above.

## 3. The prefix-cache gate cannot run its own check

`TrackAPrefixCacheTests.testHotPrefixCacheReconstructsAndDoesNotLeakBlocks`
(Qwen3-0.6B), 7 tests / 5 failures. The **first** failure is the operative one:

```
:274: failed - expected at least two wide-margin suffixes for M3 prefix-cache gate
M3 prefix: maxLogitError=inf, checkedTokens=0, mismatches=1, sharedBlocks=0
```

Read this carefully before concluding the cache is broken. The gate looks for
branch suffixes where the model's top-two logit margin is wide enough to compare
tokens safely, finds fewer than two, and bails — so `checkedTokens` stays 0,
`maxLogitError` stays at its `inf` initialiser, and `sharedBlockCount` stays 0
because the second request never ran. The four assertions after it are all
downstream of that bail.

So this is not "the prefix cache shares no blocks". It is worse in one specific
way and better in another: the cache may be fine, and **we have no passing
evidence either way** — the one gate that would tell us has been unable to
execute its own comparison. Fixture drift (model, tokenizer, or margin
threshold) is the first thing to check.

The SSD tier variant is machine-dependent, which is its own smell:
M5 Max `byteExact=true, maxLogitError=0.98, checkedTokens=1` (passes its margin
checks), M4 Pro `maxLogitError=1.875, checkedTokens=0`.
