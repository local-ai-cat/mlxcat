# Known failing gates

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

## 2. Hybrid caches cannot be combined mid-batch

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
