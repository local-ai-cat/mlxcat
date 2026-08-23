# Parity ledger — what the reference engines do, and what we do

mlxcat exists for one reason no engine below can serve: **all-Swift serving that
runs unchanged on macOS and inside an iPhone app**. That is a reason to write an
engine. It is not a reason to invent mechanisms that already exist and work.

So the references live under `guest/` and get **read**. This file is the standing
record: mechanism by mechanism, what each one does, what we do, and whether the
difference is a gap to close or a choice to defend.

**Rules for this file.** Every claim about a reference carries a
`guest/<repo>/<path>:<line>` citation, and every claim about us carries a
`mlxcat` path. A row moves to `closed` only when the change has landed AND a
gate covers it. A row that is a deliberate difference says why, so nobody
"fixes" it back.

Related: `bench/parity.py --check` scores our measured cells against the best
other engine and fails when we lose ground — the only gate here that can go red
because a competitor is better. `docs/ENGINES.md` says who is on the board.

Engines read, at the commits under `guest/` on 2026-08-23: mlx-lm `2ed2231`,
mlx-serve `d173963`, omlx `2edd26e`, vllm-mlx `0dd1157`, vllm `392d1b4`,
ollama `4b2d529`.

---

## Closed

### Free MLX's buffer cache between prefill chunks
**They:** mlx-lm calls `mx.clear_cache()` at the end of every prefill chunk —
`guest/mlx-lm/mlx_lm/generate.py:451` and `:586`. It also clears every 256 decode
tokens (`:465`).
**We did:** nothing. Each chunk attends over a longer key range than the last, so
no two allocate the same attention-scratch shape, MLX's buffer cache can never
reuse one, and it accumulated every shape for the whole prefill.
**Closed** in `Sources/MLXCat/TrackB/Scheduler.swift` (`advanceAdmission`).
Measured at 16k on an M5 Max, after load → peak:

| model | single-pass | chunked | chunked + clear |
|---|---:|---:|---:|
| Qwen3.8-27B | 53.02 | 39.02 | **18.90** −64% |
| gemma-4-12B | 34.51 | 20.74 | **13.42** −61% |
| gpt-oss-20b | 35.17 | 25.37 | **14.69** −58% |
| Qwen3-Coder-30B-A3B | 24.75 | 40.34 | **19.71** −20% |

Peak went from 1.5–3.7× loaded weights to **1.22–1.33× across all four**.
**Gate:** `MemoryBudgetTests` via `scripts/nightly-models.sh`, one bar per model.
**Deliberate difference:** mlx-lm evaluates synchronously before clearing; we
clear after an `asyncEval`, because `clear_cache` returns only unreferenced
cached buffers. omlx takes a `synchronize()` barrier first
(`guest/omlx/omlx/tests/test_admin_hot_cache_clear.py:95`) but that clears from a
separate executor mid-generation; ours is inline on the issuing thread, like
mlx-lm's.

### The prefill logits tensor
**They:** mlx-lm prefills `y.size - 1` tokens and hands the single remaining
token to the step that samples (`guest/mlx-lm/mlx_lm/generate.py:580-587`, same
shape at `:430-452`), and its prefill chunks evaluate only the cache —
`mx.eval([c.state for c in cache])`. MLX is lazy, so a prefill chunk's logits are
never computed at all; the tensor whose logits are read is always `[1, 1, vocab]`.
**We did:** read `output.logits` from whichever chunk ended the range, so
`[1, chunk, vocab]` — ~268 MB of fp16 for a 512-token chunk at a 262k vocab, and
gigabytes on the single-pass path, all discarded except one row.
**Closed:** the last prompt token is now prefilled alone, so the sampled forward
is `[1, 1, vocab]`. Measured at 16k on an M5 Max, on non-windowed families (the windowed ones do
not take this path): Qwen3.8-27B 19.13 → **18.83** GiB. Small because chunking had already bounded
it; the ordering matches the mechanism (gemma's 262k vocab saves more than the
27B's 248k).
**Capability-gated, and this is the interesting part.** It is safe for mlx-lm
because their remaining token goes through `_step()` — the SAME single-token
decode path every later token uses. Ours goes through the PREFILL call site, and
a rotating (sliding-window) KV cache has separate multi-token and single-token
update paths, so the extra `S == 1` prefill update desynchronises the ring.
`SlidingWindowBatchIntegrationTests` on gpt-oss-20b (window 128) went 3/3 green →
60 of 160 tokens diverging from serial, sustained run of 37, from token 97. Every
non-windowed gate stayed green, which is what localised it, and bisecting this
change alone restored 3/3.

So windowed caches keep the old path and everything else takes the smaller
tensor. `MLXCAT_PREFILL_LAST_TOKEN_ALONE=always|never` forces it either way, so
the trade-off stays measurable — `always` on a windowed model reproduces the
divergence above.

**Gate:** the model-backed gates, because this changes which forward produces the
sampled token — 22 core tests on Qwen3-0.6B including `BatchInvarianceTests`,
`MoEBatchIntegrationTests` token equality at widths 2/4/8 against the real 30B,
and `SlidingWindowBatchIntegrationTests` on gpt-oss. The unit suite was green
throughout and could not see any of it; `PrefillLastTokenAlonePolicyTests` pins
the capability rule itself without weights.

---

## Open — ranked

### 1. TTFT under concurrency — ROOT CAUSE FOUND, largely closed
**The question was wrong.** "Why is mlx-serve's serial prefill sublinear where
ours is linear" has no answer, because **mlx-serve is not sublinear**. The bench
fires a burst and reports the MEAN, so for N served serially at per-unit cost U
the mean is `((N-1)/2)·U + P` — for N=8 the ideal is 4.5×, and their measured
c8/c1 of 5.5× is exactly that line.

Both engines are serial. The difference is **what each queues behind**:

- mlx-serve queues behind a **prefill** (`guest/mlx-serve/src/scheduler.zig:3616`,
  `runPrefill :4628` — batch=1 per slot, like us).
- mlxcat queued behind an entire **generation**, because the benched families ran
  under `SerializationPolicy.always` and `refusesToJoin` refuses admission while
  anything runs (`Sources/MLXCat/TrackB/Scheduler.swift:298`,
  `Sources/MLXCat/TrackB/Request.swift:93`).

So the per-unit cost was `prefill + tokens/decode-rate` instead of `prefill`. That
ratio is the whole gap — no prefill-speed difference is involved. The fit, using
G = prefill + 128/measured-solo-decode, is within **1.3 % at every width**:

| model | c2 pred/meas | c4 pred/meas | c8 pred/meas |
|---|---:|---:|---:|
| gemma-4-E2B (G=1689 ms) | 913 / **891** | 2603 / **2605** | 5981 / **5929** |
| Qwen3.8-27B (G=12316 ms) | 9199 / **9318** | 21515 / **21622** | 46147 / **46199** |

The decode column corroborates it: our per-stream rate stayed flat ~76–79 tok/s at
every width (each request ran ALONE at full speed) while mlx-serve's halved per
doubling, because their streams actually share the machine.

**Interventionally proven, and largely closed.** Lifting `.always` for gemma-4
text rows (`multimodalOnly`) moved gemma-4-E2B longgen c8 TTFT **39,346 → 2,439 ms**
and aggregate 70 → 123 tok/s — now ahead of mlx-serve's 85 — with c1→c8 scaling
falling 195× → 11.2×. `docs/COMPETITIVE.md` § Status 2026-08-23 has the table.

**What remains is the same lever on the families still serialized:**
- ~~`qwen3_5` (Qwen3.5-4B **and** Qwen3.8-27B — two of six benchmark models) is
  `.always` pending the server-layer crash in §1d.~~ **Closed 2026-08-23.** It
  was not a server-layer crash: Qwen3.5 anchors its M-RoPE position in
  `LMOutput.State`, the batch generator dropped it at width ≥2, and the model's
  own precondition trapped the process. The crash report named the frame; the
  black-box narrowing that produced "it must be the server" had ruled out the
  cache, which was never the suspect. Fixed by carrying the anchor per row
  (`BatchPositionalState`), token-exact against serial at 4 ragged rows and at
  width 8. Expected: 27B c8 mean 46.2 s → ~13–15 s — to be confirmed by the next
  bench pass.
- `qwen3_moe` is capped at width 2, so requests 3+ still queue behind generations.
  Raising it needs the width-4/8 numerics fixed (2.69 against a 1.25 tolerance).

**Correction worth keeping.** `SchedulerAdmissionShapeTests.testIdleSchedulerAdmitsEveryWaitingRequestInOneStep`
passing was read as "admission count is not the problem". It checked the right
loop under the WRONG policy — it uses the default, so for a serializing family
requests 2..N never reach the loop it asserts on.
`testASerializingPolicyQueuesBehindWholeGenerations` now pins the real behaviour.

### 2. Busy-prefill chunk width — partly closed
**They:** mlx-lm and omlx use **2048** (`guest/mlx-lm/mlx_lm/generate.py:1509`,
`guest/omlx/omlx/scheduler.py:1304`); mlx-serve uses **8192**
(`guest/mlx-serve/src/generate.zig:34`) with per-model caps —
MoE → 4096, composed-causal head-dim-256 → 2048 (`generate.zig:111`).
**We did:** 512 everywhere.
**Now:** 512 idle, **2048 busy** — because one number was answering two questions
that want opposite answers. Idle prefill has no stream waiting on a tick, so
width is a pure memory question; busy prefill pays a decode tick and an actor
round trip per boundary, so width is a scheduling question. Measured cost of
widening the IDLE path at 16k on an M5 Max, which is why it stayed narrow:

| model | idle 512 | idle 2048 |
|---|---:|---:|
| gemma-4-12B | 13.42 | 15.56 (+16%) |
| Qwen3.8-27B | 18.90 | 22.91 (+21%) |

**Gate:** `SchedulerAdmissionShapeTests.testPrefillChunkWidthMatchesTheReferenceRange`
pins the busy width against the reference range.
**Still open:** mlx-serve's per-model caps (`generate.zig:111`) — the width that
is right for a 4-bit MoE is not the width that is right for a head-dim-256
composed-causal model, and we use one number for all of them. And the TTFT half
of this is unmeasured: the memory cost above is measured, the latency win is not,
because it needs a concurrent bench run.

### 4. Packed multi-request prefill
**They:** mlx-lm right-pads up to `prefill_batch_size = 8` rows into ONE forward
per chunk (`guest/mlx-lm/mlx_lm/generate.py:1143`, forward at `:1160`, default at
`:1508`); vllm-mlx inherits it (`guest/vllm-mlx/vllm_mlx/scheduler.py:21`);
ollama mixes prompt and decode tokens from several sequences in one batch
(`guest/ollama/runner/ollamarunner/runner.go:517`).
**We:** every prefill forward is batch=1 (`Scheduler.swift:456`); batching exists
only for decode (`Sources/MLXCat/TrackB/BatchGenerator.swift:582`).
**Why it is ranked last:** it needs ragged per-row positions at prefill time —
the same scalar `cache.offset` family already documented as broken for ragged
rows (`Scheduler.swift:242`, measured 2026-08-23). And mlx-serve reaches its
curve without it, so (1) and (2) likely capture most of the win.

---

### 5. Model-gated suites take the CONSTRUCTOR DEFAULT for cache capabilities
Not a competitor gap — a gap between our gates and our own product, found while
closing item 3.

`MLXCatEngine`'s default `cacheCapabilities` declares
`usesWindowedKVCache: false`, and the scheduler makes real decisions from that
flag (prefix-cache eligibility; whether the last prompt token may be prefilled
alone). In production `NativeModelLoader.usesWindowedKVCache(configuration:)`
derives it from the checkpoint. **12 of 14 suites that build an engine never
declare it**, so any gate pointed at a windowed model measures a configuration
that never ships.

Caught the hard way: `SlidingWindowBatchIntegrationTests` — the gate whose entire
subject is a rotating cache — was telling the engine it was not windowed, so a
correct capability guard could not engage and the gate kept failing.

Fixed in `SlidingWindowBatchIntegrationTests` and `MemoryBudgetTests`. The latter
moved two published numbers: gemma-4-12B 12.92 → **13.45** GiB and gpt-oss-20b
14.69 → **14.99**, because windowed families correctly decline the
last-token-alone prefill. Of the budgeted models only those two are windowed
(`sliding_window` 1024 and 128); the rest are unaffected.

**Still open:** the other suites have not been audited. Most are pointed at
non-windowed models so the default is accidentally right, which is exactly why
this survived — it is only wrong when it matters.

### 6. Quantized KV cache — we have none, and the plumbing already exists
**They:** mlx-lm converts any cache past `quantized_kv_start` via
`to_quantized(group_size:bits:)` — `maybe_quantize_kv_cache`
(`guest/mlx-lm/mlx_lm/generate.py:299-304`), called per prefill chunk (`:418`,
`:441`) and per decode step (`:558`, `:583`).
**We:** nothing. `grep -rn "toQuantized\|kv_bits" Sources/` is empty — zero call
sites — even though our own dependency ships the receiving end:
`QuantizedKVCache`, `KVCacheSimple.toQuantized(groupSize:bits:)` and
`QuantizedKVCacheProtocol` at
`guest/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:111`, `:424-440`.
**Why it matters:** ~4× KV memory at 4-bit on top of the 1.22–1.32× peaks we now
have. It is the lever that decides what fits at 32k–128k on 48 GiB, and on iOS.
**Known integration cost, stated so it is not underestimated:**
`BatchLayerCache` merge/extract/filter must learn the quantized state layout, the
prefix store must round-trip it, and windowed caches are excluded upstream too.
It also shifts logits, so `BatchInvarianceTests`' 1.25 tolerance needs an
explicit decision rather than a silent pass.

### 7. Per-admission batched-KV copy
**They:** mlx-serve admission is a pointer append into `decoding`
(`guest/mlx-serve/src/scheduler.zig:3650`) — slots own their caches, zero copy.
**We:** `BatchGenerator.insert` does `currentLayer.copyLayer()` — a full
`kvCache.copy()` of every already-resident row — plus `extend` per layer per
insertion (`Sources/MLXCat/TrackB/BatchGenerator.swift:339`, `:353`), then a
blocking `eval` (`:419`). Across an N-burst that is O(N²) KV traffic and N
pipeline drains. Negligible at 230-token prompts, material at 4k/16k.
**Harness gap:** no concurrency bench arm at 4k/16k — `bench/matrix.json`
concurrency tiers are short/longgen only. Add one before claiming a win.

### 8. Decode-loop cache clearing
**They:** mlx-lm also clears every 256 DECODE tokens
(`guest/mlx-lm/mlx_lm/generate.py:465`).
**We:** clear only between prefill chunks. Cheap to copy; bounds buffer growth at
longgen. `MemoryBudgetTests` would need a longer generation than its default 32
tokens to see it.

## Deliberate differences — do not "fix" these

- **Decode-first vs prefill-first ordering.** mlx-lm and omlx decode then prefill;
  we and mlx-serve prefill then decode. Both orders appear among the winners.
- **FCFS with head-of-line blocking for rows needing solitude**
  (`Scheduler.swift:263`). Every reference is FCFS by default — mlx-lm
  (`generate.py:1744`), omlx (`scheduler.py:1298`), vllm
  (`config/scheduler.py:109`). mlx-serve skips a blocked head rather than
  stalling behind it (`scheduler.zig:1982`), which is worth revisiting, but no
  engine reorders by priority.
- **No fused decode+prefill under one token budget** (vllm's model,
  `guest/vllm/vllm/config/scheduler.py:42`). No MLX engine does this — even
  vllm-mlx and omlx declare `max_num_batched_tokens` and never read it. It needs
  ragged-batch attention our stack does not have, and the measured MLX leaders
  get their curves without it.
- **MLX encode thread-overlap.** Three variants measured dead on this repo
  (`exp/encode-thread`): same-stream early enqueue, multi-stream background
  encode, and a dedicated encode thread. Do not re-attempt without an MLX-core
  allocator/encoder change.
