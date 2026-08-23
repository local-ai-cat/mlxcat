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

## 1d. Batched `qwen3_5` died in the MODEL, not the server — FIXED 2026-08-23

With the exclusion lifted, the benchmark's batched `qwen3_5` arm failed: width 4
returned a completion carrying no `usage` frame, width 8 killed the server
process — nothing in its log and the runaway guard never fired.

`HybridBatchScaleTests` was written to reproduce that from a test and could not:
8 rows x 512 tokens, a `SessionPrefixKVStore`, a ~1300-token prompt spanning many
cache blocks — all clean. **That ruling-out was sound and the conclusion drawn
from it was wrong.** It cleared the batched CACHE, and this defect was in the
MODEL's positional contract, which no hybrid-cache test touches. The section
then said "the failure lives above the engine, in the SSE streaming path,
`NativeModelEngine`'s chat/tool/stop handling, `EnginePool`, or concurrent HTTP
admission", and it lived in none of them.

**What actually happened**, from the crash report the run left behind
(`mlxcat-http-2026-08-23-022441.ips`, `EXC_BREAKPOINT` / SIGTRAP):

```
Swift runtime failure: precondition failure
specialized Qwen35.callAsFunction(_:cache:state:)      Qwen35.swift:1298
ContinuousBatchGenerator.computeNextTokens(...)        BatchGenerator.swift:567
ContinuousBatchGenerator.advanceStepSynchronously()
Scheduler.step()  →  EngineStreamDemux.pump()
```

Qwen3.5 does not read its decode position off the cache the way every batching
family does. It computes `cache.offset + ropeDeltas[row] + j` from a **scalar**
offset plus an M-RoPE anchor it hands back in `LMOutput.State`, and refuses a
warm cache without it:

```swift
precondition(faCacheOffset(cache ?? []) == 0 || state?[ropeDeltasKey] != nil,
             "Qwen35 cannot continue a warm prompt cache without qwen35.ropeDeltas")
```

`BatchGenerator` dropped that state the moment a second row joined. The comment
there said stateful models were kept out by the allowlist — true, right up until
the benchmark arm lifts the allowlist, which is the arm's entire purpose. Width
4 and width 8 were one trap at two timings, not two symptoms.

Two process notes worth keeping. **A silent death is a crash report, not an
absence of evidence** — `~/Library/Logs/DiagnosticReports` had the answer the
whole time, and reading it took less time than the black-box narrowing did.
And a SIGTRAP is a Swift precondition: the message goes to stderr, which a
server whose stderr nobody captured turns into "nothing in its log".

**The fix** (`Sources/MLXCat/TrackB/BatchPositionalState.swift`) is not to
restore the scalar — a scalar cannot describe two rows. The decode branch
already accepts a per-row vector (`Qwen35.swift:897-918` broadcasts a 0-d delta,
truncates a long one, and indexes `delta[row]` otherwise), so the anchor is kept
per row and reassembled each step with `-leftPadding[row]` folded in: the offset
the model adds is the PADDED batch length, so a short row has to be pulled back
to its own position — the same correction `BatchKVCache.ropeOffset` makes for
every other family. At width 1 with no padding it is exactly the scalar that
shipped before. A row that arrives without an anchor now throws instead of
trapping the process.

**Evidence.** `PositionalStateBatchIntegrationTests`, on the `VLMModelFactory`
path production loads: 4 rows of deliberately different prompt lengths are
token-EXACT against serial, and so are 8 rows at 24 tokens each. Disable the
per-row state and the same test dies on signal 5 with the identical message the
benchmark's crash report carried. That is a stronger result than the 1.25 logit
tolerance every other family is admitted on, so `qwen3_5` moved to
`multimodalOnlyModelTypes` uncapped: text rows batch, image rows keep the batch
to themselves.

Note the near-miss in the earlier evidence: `BatchInvarianceTests` swept
Qwen3.5-4B at widths 1/2/4/8 and reported it clean, because it loads through
`LLMModelFactory` — a **different** `Qwen35Model` with no such precondition.
Production routes `qwen3_5` through `VLMModelFactory`. A gate on a path
production never selects is the same trap that put two wrong memory numbers in
this file the day before.

**Confirmed through the real server**, because the unit tests drive the
in-process engine and the thing that died was a process. `mlxcat-http` at
Qwen3.5-4B, 8 concurrent SSE streams of 96 tokens each, M5 Max:

| | wall | aggregate |
|---|---|---|
| one stream alone | 2.55 s | 37.7 tok/s |
| eight concurrent | **9.9 s** (all eight) | **77.6 tok/s** |
| eight serialized (8 x 2.55) | 20.4 s | 37.7 tok/s |

Every stream returned rc=0, 100 data frames and its `[DONE]`, and the server was
still healthy afterwards — at the width that used to kill it. 2.06x aggregate
throughput, and the last stream finishes in half the time.

The other §1d symptom resolves with it: "width 4 returns a completion carrying no
`usage` frame" was the stream being cut before the frame, not a separate defect.
Asked for properly (`stream_options.include_usage`) the frame arrives.

`qwen3_5` covers Qwen3.5-4B **and** Qwen3.8-27B — two of the six benchmark
models, both previously serialized at ~37 s TTFT under load on an M4 Pro. Bench
pass 07 is the board version of the table above.

## 1e. `gemma4_unified` fails logit invariance at every batched width — capped at 4, still open

`multimodalOnlyModelTypes` holds two entries, `gemma4` and `gemma4_unified`, and
entry is documented as requiring both gates:
`BatchInvarianceTests.testBatchInvarianceAcrossModelFamilies` for logits and
`SchedulerEngineTests.testVLMBatchEqualityAcrossModels` for images. The logit
evidence recorded for the grant is gemma-4-**E2B** (a `gemma4`). gemma-4-**12B**,
the `gemma4_unified` that shares the grant, had never been run through it.

Run 2026-08-23 (M5 Max, `MLXCAT_BATCH_INVARIANCE_MODELS`) with a **width-1
control arm** added to the sweep, so a path difference could be told apart from a
batching defect. Tolerance is 1.25; `margin` is the serial top-1/top-2 gap, and a
token can only flip where the error exceeds it:

| model | batch 1 (control) | batch 2 | batch 4 | batch 8 |
|---|---|---|---|---|
| gemma-4-E2B (`gemma4`) | 0.0 | 0.688 ok | 0.688 ok | 0.953 ok |
| **gemma-4-12B (`gemma4_unified`)** | **0.0** | **1.656** (margin 11.5) | **1.656** (margin 11.5) | **1.740** (margin **1.125**) |
| Qwen3.5-4B | 0.0 | 0.0 ok | 0.375 ok | 0.656 ok |
| Qwen3-Coder-30B (`qwen3_moe`) | 0.0 | 0.938 ok | 2.688 | 2.688 |
| gpt-oss-20b | 0.0 | 1.7e-05 ok | 2.0e-05 ok | 2.9e-05 ok |

**The control settles what this is.** Width 1 is bit-exact — 0.0, every model.
A single row through `StaticBatchGenerator` has nothing to be contaminated by, so
the batched code path's own numerics are not the explanation. The divergence at
width >= 2 is rows contaminating each other, and it is real.

**What was done: `gemma4_unified` is capped at width 4**
(`batchDecodeWidthCeilings`), not reverted. At widths 2 and 4 the margin is ~7x
the error; at width 8 the error EXCEEDS the margin, which is the width where a
flip stops being impossible and starts being luck (`mismatched` is 0 everywhere,
so nothing has flipped yet). The cap removes that width and keeps the one the
concurrency win was measured at — `d630cee`'s c4 TTFT 11,999 -> 870 ms on the iOS
flagship family. It is strictly safer than the previous behaviour, which was
unbounded.

**What is still open.** The cap does not make `gemma4_unified` clean; it bounds
the blast radius of a defect that is still there at widths 2 and 4. Three ways to
close it, and which one is a product call:

1. Find the numerics defect. `gemma4_unified` is a distinct loader path from
   `gemma4`, both are QAT 4-bit, and `gemma4` measures 0.69 where this measures
   1.66 — so "bigger model, higher noise floor" does not explain it. This is the
   only option that keeps the win AND satisfies the tolerance.
2. Drop the ceiling to 1, i.e. `.always`, and lose `d630cee` on 12B.
3. Decide the flat 1.25 tolerance is the wrong bar and replace it with a
   margin-relative one — the repo already reasons that way in
   `TrackAPrefixCacheTests` ("a flip means something: `margin > logitError * 4`").
   That is a change to the standard every family is judged by, so it needs to be
   argued once, in writing, not settled per-family.

## 1f. Batched gemma decoded rows 1..N UNROTATED — an MLX RoPE grid bug — FIXED by reverting the grant 2026-08-24

`gemma4` and `gemma4_unified` were moved to `.multimodalOnly` earlier on
2026-08-23, which is what produced the headline result on the board (gemma-4-E2B
longgen c8 TTFT 39,346 -> 1,793 ms). The gate cited for that grant is
`BatchInvarianceTests.testBatchInvarianceAcrossModelFamilies`.

**That gate loads gemma through the wrong factory.** `LLMModelFactory` maps both
`gemma4` and `gemma4_unified` to `Gemma4Model` (`LLMModelFactory.swift:38-39`);
`VLMModelFactory` maps them to `Gemma4`/`Gemma4Unified`
(`VLMModelFactory.swift:99-100`). `NativeModelLoader` routes both through the
**VLM** factory in production (`vlmModelTypes` contains both), and they are
different attention implementations — the MLXVLM one rotates queries and freshly
cached keys at the scalar `cache?.offset ?? 0` (`Gemma4.swift:872,880,903`) where
its MLXLLM twin uses the per-row-aware `applyRotaryPosition(rope, to:, offset:
cache?.ropeOffset)` (`Gemma4Text.swift:313,331,362`).

This is the **third** time in two days that a gate has been found measuring a
path production never selects — the other two are §1d above and the two wrong
memory numbers corrected the day before.

**Measured on the production factory** (`GemmaRoPEOffsetProbeTests`, greedy,
12 generated tokens, rows that diverge from a serial run of the same input):

| model | width 1 (control) | width 2 | width 4 equal-length | width 4 ragged |
|---|---|---|---|---|
| gemma-4-12B (`gemma4_unified`) | **0/1** | 1/2 | 3/4 | 3/4 |
| gemma-4-E2B (`gemma4`) | **0/1** | 1/2 | 2/4 | 2/4 |

Read the two ends of that table together.

**The width-1 control is clean for both**, so this is not the batched code path's
own numerics and not a harness artifact — a single row through the batch
generator reproduces serial exactly. Rows are contaminating each other.

**Equal-length rows diverge as much as ragged ones**, which REFUTES left padding
as the cause. That matters because the scalar-offset defect above is real and
looks like the obvious culprit: with every row the same length there is no
padding, the scalar offset is correct for every row, and the divergence is
unchanged. So there is a second cause, still unlocated, and fixing the RoPE
offset alone will not close this.

Both hold on real chat-templated prompts, not only on synthetic token ids — the
first version of this probe used synthetic ids, where greedy decoding sits near
ties and a divergence can be a property of the input rather than the engine.

**For scale, this is a gemma problem and not a batching problem.** On the same
production-factory bar, `qwen3_5` is token-EXACT at 4 ragged rows and at width 8
(`PositionalStateBatchIntegrationTests`), and gpt-oss measures logit errors of
~2e-05. Batched decode is exact for our other families; gemma is the outlier.

**What is NOT claimed here:** that batched gemma output is *wrong*. It is
different from serial, which is a weaker statement — every serving engine on
non-invariant kernels is batch-variant, and `mismatched` was 0 in the (LLM-path)
logit sweep. What is claimed is that the evidence the grant rests on does not
describe the shipping path, and that on the shipping path gemma fails the bar
every other family here clears.

### Narrowed 2026-08-24: it is the gemma model, not our batching

Three hypotheses died, each to a control rather than to an argument.

**Left padding — no.** Equal-length rows diverge as much as ragged ones (table
above).

**Pipelined launch-ahead — no.** `MLXCAT_DECODE_PIPELINING=0` leaves the numbers
bit-identical. The probe prints which state it ran in, because a toggle that
silently did nothing looks exactly like a toggle that did.

**The rotating-cache merge — no**, and this is the control that settles it.
gemma at width >= 2 was the only configuration that had ever combined a
rotating-merged `BatchKVCache` with pipelined decode, which made the merge the
obvious suspect. Untested is not the same as guilty, so it was tested:

| model, through the engine | rotating layers merged | vs serial | vs each other |
|---|---|---|---|
| gpt-oss-20b | 12 | 0/4 | **0/4** |
| Qwen3.5-4B | 0 (hybrid) | 0/4 | **0/4** |
| gemma-4-12B | 40 | 3/4 | **3/4** |

gpt-oss merges rotating caches into a `BatchKVCache` and runs the same pipelined
path, and it is exact.

**And the sharpest measurement of all: four IDENTICAL rows disagree with each
other.** Same prompt, same length, same tokens, decoded together — 3 of 4 differ
from row 0 at width 4, 1 of 2 at width 2. Identical input must give identical
output regardless of row position. This is per-row contamination, and
`vsSerial == vsEachOther` in every cell, which reads as row 0 correct and rows
>= 1 wrong.

So the defect is in the gemma model implementation, and since BOTH gemmas fail
it is in the shared `Gemma4TextAttention`/`Gemma4TextBackbone` rather than in
anything specific to `Gemma4` or `Gemma4Unified`. The surfaces neither other
family has are the KV-shared layer tail (`sharedKV` reusing an earlier layer's
`kvState`, with `offset: Int` carried through `intermediates`,
`Gemma4.swift:1292-1305`), the sliding/full interleave, and
`gemma4AdjustAttentionMask`.

### Located 2026-08-24: MLX's RoPE fast path drops the batch axis

It was never a tolerance question. **Batched gemma decode produces UNROTATED
queries and keys for every row but the first.**

`RoPE::eval_gpu` takes a fast path when the input is row-contiguous, the
sequence length is 1, and the offset is a **scalar**
(`mlx/backend/metal/rope.cpp:96`). That branch dispatches

```cpp
grid_dims = MTL::Size(dims_ / 2, N, 1);     // rope.cpp:134-139
```

where `N` covers only the head axes. The batch size is absent. The general
branch has it — `dim2 = B * ((N + n_per_thread - 1) / n_per_thread)` at `:149` —
which is why every other family is fine. The kernel therefore rotates batch row
0, and because the row-contiguous input is donated, rows 1..B-1 come out of the
shared buffer untouched.

Reproduced with no model and no weights, in under a second
(`RoPEBatchGridProbeTests`):

```
ROPEGRID scalar-offset rows-unrotated=[1, 2, 3] max|out-in| per row=[0.0000, 0.0000, 0.0000]
ROPEGRID array-offset  rows-unrotated=[]
```

Bit-identical to the input. No positional information at all.

Gemma's VLM attention is the only code in our stack that reaches that primitive
with a scalar offset on a batched decode tensor
(`Gemma4.swift:872,880,903`). Everything else rotates through
`applyRotaryPosition(..., offset: cache?.ropeOffset)`, and a per-row array
offset forces `single = false`.

**One cause accounts for every observation**, which is what makes it the answer
rather than another candidate:

| observation | why |
|---|---|
| width 1 exact | B=1, the grid covers the whole tensor |
| identical rows disagree with each other | row 0 rotated, rows >= 1 not |
| `vsSerial == vsEachOther` everywhere | row 0 is also the one that matches serial |
| equal-length == ragged | rows >= 1 never have an offset read at all |
| pipelining toggle inert | the defect is inside one primitive |
| prefill clean | T > 1 misses the fast path |
| gpt-oss and qwen3_5 exact | both take the array-offset path |

**What was done.** `gemma4` and `gemma4_unified` are out of
`multimodalOnlyModelTypes` and back to `.always`. This is a correctness floor,
not conservatism, and it costs exactly the headline it bought — gemma-4-E2B
longgen c8 goes back to ~39 s at c8. `MLXCAT_UNSERIALIZE_MODEL_TYPES` still
lifts it for measurement.

### Upstream fixed this on 2026-05-11. We cannot reach the fix.

Checked rather than assumed, and the answer changes the options:

| | |
|---|---|
| upstream fix | `76a977ca` — **"Fix rope single token multiple sequences (#3498)"**, 2026-05-11 |
| `ml-explore/mlx` main today | `dim1 = B * N` in the single branch — the batch axis is there |
| what mlx-swift **0.31.6** vendors | mlx `ce45c525`, **2026-03-12** — two months before the fix |
| what mlx-swift **main** vendors | `ce45c525`. The same commit. The submodule has not moved. |

0.31.6 is the newest mlx-swift release (2026-07-02) and its `main` points at a
March mlx. **So no pin bump reaches the fix** — not a release, not a branch
revision. That was the cheap option and it is not available.

It also means this is not our bug to report: it is filed, fixed, and merged
upstream. What is stale is mlx-swift's submodule, which has not advanced in
roughly five months.

**How to get the win back**, in preference order:
1. **Model-level, in a fork of `mlx-swift-lm`.** Port
   `applyRotaryPosition(rope, to:, offset: cache?.ropeOffset)` into
   `Gemma4TextAttention`, carrying `RoPEOffset` rather than `Int` through the
   shared-KV `intermediates` (`Gemma4.swift:1292-1305`) so E2B's 20-layer shared
   tail inherits per-row anchors. This routes gemma onto the array-offset kernel
   path, dodging the broken grid entirely, AND fixes the separate
   ragged-position defect that staggered admission creates in production — two
   defects, one edit, and it needs nothing from mlx-swift. We do not currently
   carry an `mlx-swift-lm` fork; standing one up is **Phil's call**.
2. **Fork `mlx-swift`** to advance its submodule past `76a977ca`, or cherry-pick
   that commit. Larger blast radius: `app/project.yml` pins mlx-swift,
   mlx-swift-lm and mlxcat as a set that moves together, and 0.31.5+ already
   carries a CI toolchain constraint (`swift-tools 6.3.0`).
3. **Wait** for mlx-swift to advance its submodule. Zero work, no timeline, and
   gemma stays serialized until then.

Worth knowing regardless of which: **any consumer of this mlx-swift doing
scalar-offset batched decode silently corrupts rows >= 1.** It is not specific
to gemma or to us.

The probe pins the BROKEN behaviour on purpose. A test asserting the correct
behaviour would sit permanently red and be ignored; asserting the defect means
it goes red exactly once — the day the grid is fixed — and its message says to
restore gemma and delete it.

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
means the batched path returned **no tokens at all**. `qwen3_5` was also on the
scalar-offset serialization list at the time, so the same relationship held.

Read this alongside §1d: "returns no tokens at all" was a **second** symptom of
`qwen3_5` batching, and it had a **different** cause from the crash — this one
was the cache layout, that one the dropped RoPE anchor. Both are fixed, and both
had to be, which is why fixing this one did not make the benchmark arm work and
the section above spent a day looking in the server.

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
