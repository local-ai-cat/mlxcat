# Benchmarking mlxcat on a real iPhone

How iOS rows are produced, why each rule exists, and what the first complete
night measured. Everything here is reproducible from the repo with two
commands; nothing in it depends on a human watching.

MLX does not run on the iOS simulator — it needs a real Apple Silicon GPU. Every
number below comes from physical hardware.

## The two commands

```bash
# Leaderboard rows: the roster, both engine arms, both context tiers.
ios/BenchHost/run-device-bench.sh --device <udid>

# The context ceiling: walk prompt length up until the device kills us.
ios/BenchHost/run-context-ladder.sh --device <udid> --model <hf-id>
```

`xcrun devicectl list devices` gives the udid. The phone must be paired,
trusted, unlocked when the run starts, and on power.

## Why the harness is shaped the way it is

Each rule below is a scar. They are listed with the failure that produced them,
because a rule whose reason is lost gets removed by the next person.

**One `xcodebuild` invocation per phone, whole roster inside one app session.**
A passcode phone auto-locks in the gap between per-model launches, and iOS
SIGKILLs Metal work behind a locked screen. That killed the first device night.

**Per-device `DerivedData` (`.build/dd-<udid>`).** Two phones benching
concurrently otherwise race one build directory.

**Rows are printed to the console as `BENCHROW` *and* attached to the
xcresult.** The attachment is the tidy path; the console copy is the one that
survives a crash, and the driver recovers it. A harness that only reports on
success cannot report the most interesting result.

**`com.apple.developer.kernel.increased-memory-limit` (2026-08-26).** Without
it the host is capped far below the shipping app, so it measures a phone the app
never runs on. See "The ceiling is the experiment" below. The wildcard team
profile cannot carry the capability, hence `-allowProvisioningUpdates`.

**`*.jinja` in the snapshot glob (2026-08-26).** gemma-4 keeps its chat template
in a standalone `chat_template.jinja`, not in `tokenizer_config.json`. Without
it the tokenizer throws `missingChatTemplate` *after* a full weight download and
load — the most expensive possible place to discover a missing 20 KB file.

**Rows self-validate from a measured guard (2026-08-30).** The Mac harness has
a quiet-machine guard (loadavg, free memory); the phone's equivalent used to be
"the operator is the guard" — and across three device nights not one row was
ever promoted, because a human attestation after a multi-hour night is a check
that never returns true. `RunConditionMonitor` now samples thermal state,
battery/power, lock (`isProtectedDataAvailable`), foreground, and Low Power
Mode every 2 s; each row is stamped `valid_for_leaderboard` from the worst
state observed during its own cell, with the evidence in the row's `host`
object, so a mid-night thermal event invalidates exactly the cells it touched.
Before each measurement the harness also waits (bounded, 10 min) for thermal
≤ fair — the device analog of wait-for-quiet; the very first guarded run
opened with a correctly-invalidated `thermal serious` row, which is what
motivated the wait. The one condition the platform cannot expose — a finger on
an unlocked, foregrounded screen — costs no more than compositing a static
view and stays out of the gate.

**Both arms, always.** `mlxcat-inprocess` is the engine; the
`mlx-swift-lm-tokeniterator` arm is the raw `TokenIterator` loop, which is
exactly the legacy `LLMEvaluator` mechanism. The pair answers "new engine vs
what the app used to do" per device. A single arm answers nothing.

## The ceiling is the experiment, not a setting

The app budgets **weights** against the device: `iOSLoadableCeilingBytes` asks
the kernel via `os_proc_available_memory()` rather than trusting a constant.
Nothing budgets the **KV cache** — a model's context window is a static
`generationProfile.contextWindow` from the catalog and never consults memory.

So on a phone the real limit on context is not a number we chose. It is jetsam.
`run-context-ladder.sh` measures where that is, per device and per model, by
walking prompt length up until the process dies. It reports:

- `MAX SURVIVED` — the largest prompt that completed, with peak footprint and
  remaining headroom
- `KILLED AT` — a rung attempted and never finished: **our** memory problem,
  and it moves with the device
- `REFUSED AT` — a rung the model itself rejected: the model's own context
  window, which does not move with the device

That distinction is the whole point. A ladder that only reported "it broke"
would conflate a policy limit with a memory limit.

Because a working ladder ends with its own process being SIGKILLed, the answer
can never live in a return value or an attachment. Every rung prints
`BENCHRUNG attempt` before the work and `BENCHRUNG survived` after it, so the
ceiling is recoverable as "highest rung with a survived line", and an `attempt`
with no `survived` **names** the rung that killed the device.

## Measured: the first complete night (2026-08-26)

Devices: iPhone 17 Pro Max (`iPhone18,2`, 12 GB) and iPhone 16 Pro Max
(`iPhone17,2`, 8 GB), both iOS 27.0.

### Finding 1 — the entitled budget does not scale with RAM

```
iPhone18,2 (12 GB)   ceiling 4.77 GiB   available 5.96 GiB
iPhone17,2 ( 8 GB)   ceiling 4.77 GiB   available 5.96 GiB
```

Identical. iOS caps the entitled per-app budget rather than scaling it with
physical memory, so **the 12 GB phone buys no headroom over the 8 GB phone**,
and a ceiling conclusion on one transfers to the other. `ceiling` is 80% of
`os_proc_available_memory()`, leaving room for the transient peak the sampler
cannot cap.

Before the entitlement both phones reported ceiling 2.60 GiB / available
3.25 GiB, and gemma-4-E2B — 4.1 GB of weights, needing 3.83 GiB resident — was
jetsam-killed at load ~25s in, on both phones, producing zero rows. That is not
bad luck; 3.83 > 2.60 is arithmetic.

### Finding 2 — mlxcat's memory advantage is large on device

Qwen3-1.7B-4bit, `iPhone17,2`, peak physical footprint:

| cell | legacy TokenIterator | mlxcat | |
|---|---|---|---|
| 256 | 2.37 GiB | **1.24 GiB** | **−48%** |
| 4k | 3.04 GiB | **2.41 GiB** | **−21%** |

This is `Memory.clearCache()` between prefill chunks (the 2026-08-23 parity
program) paying off on the platform where the alternative is a process kill.

`lifetime_max_phys_footprint` is a process high-water mark and stays pinned at
the largest cell ever run — it is NOT per-model, and reading it as such
suggests a leak that the per-cell `peak` column disproves.

### Finding 3 — latency, gemma-4-E2B (the model the app ships to phones)

`iPhone18,2`, medians of 3 runs:

| | 256 legacy | 256 mlxcat | 4k legacy | 4k mlxcat |
|---|---|---|---|---|
| TTFT | 169.6 ms | **147.9** | 2235.9 ms | **2079.0** |
| prefill tok/s | 1244 | **1427** | 1802 | **1939** |
| decode tok/s | **24.55** | 23.35 | **20.94** | 19.49 |
| cold first request | 10.2 s | **6.3 s** | 8.7 s | 8.7 s |

mlxcat wins TTFT, prefill and cold start; the legacy arm holds a ~5% decode
edge. Cross-device agreement is tight — gemma 4k peaks 4.41 GiB on
`iPhone17,2` and 4.42 GiB on `iPhone18,2`.

### Finding 4 — the 4k cell runs close to the edge

gemma-4-E2B at 4k peaks **4.41–4.42 GiB against a 4.77 GiB ceiling**: roughly
0.35 GiB of headroom, on both phones. A `MTLCompilerService` XPC failure
("may have crashed, been jetsammed") was observed on `iPhone17,2` during the
run, which is what real memory pressure looks like from inside the process.

This is `KNOWN-FAILURES.md` 1c ("the memory ceiling is advisory, not
enforcing") arriving in practice: the ceiling bounds MLX's allocator, not the
process footprint, and on iOS the enforcement is a kill.

### Finding 5 — the first measured context ceiling

`run-context-ladder.sh`, Qwen3-1.7B-4bit, iPhone 17 Pro Max, mlxcat arm:

| prompt tokens | peak footprint | verdict |
|---:|---:|---|
| 256 | 1.26 GB | survived |
| 1,024 | 1.78 GB | survived |
| 4,096 | 2.53 GB | survived |
| **16,384** | **4.67 GB** | **survived — 0.44 GiB headroom** |
| 65,536 | — | **JETSAM KILL** |

So the practical ceiling for a 1.7B 4-bit model on a 12 GB iPhone is **16k
tokens**, and it is set by memory, not by the model's own context window. The
app's catalog says nothing about this: `generationProfile.contextWindow` is a
static per-model number that never consults the device. A user on this phone can
ask for more context than the phone can survive.

The kill is the result, not a failure of the run: the ladder recovered its own
answer from the `BENCHRUNG` console lines after the process was killed, which is
the whole reason those lines exist.

### Finding 6 — KV quantization is worth 4x the context on a phone

Same device, same model, same ladder; the only difference is
`MLXCAT_KV_BITS=4 MLXCAT_QUANTIZED_KV_START=0`:

| | fp16 (default) | kv4 |
|---|---|---|
| MAX SURVIVED | **16,384** tokens | **65,536** tokens |
| peak at max | 4.67 GB | 5.64 GiB |
| 65,536 rung | **JETSAM KILL** | survived |
| verdict | ceiling found | **ceiling NOT reached — ran out of rungs** |

Qwen3-1.7B is full-attention, so all 28 layers quantize. fp16 KV is
~112 KiB/token; the measured kv4 ratio is **3.56x, not 4x** — affine group-64
stores an fp16 scale and bias per group
(`Sources/MLXCat/TrackB/KVQuantizationPolicy.swift:12-15`).

**This is not free, and it is not on by default.**

- Decode costs **0.80-0.84x**, measured on the Mac.
- Enabling it forces `serializationPolicy = .always` and **disables the prefix
  cache** (`Sources/MLXCat/TrackB/Scheduler.swift:103-109`). Harmless for a
  single-stream ladder; a real turn-2 TTFT loss in a chat session.
- gemma-4-E2B benefits far less: its rotating layers refuse conversion by
  design, as they do in mlx-lm (`guest/mlx-lm/mlx_lm/models/cache.py:551`).

**Read the peak column before celebrating.** At 65,536 the process peaked at
**5.64 GiB against a 4.79 GiB configured ceiling** — 0.85 GiB *over*. It
survived because the ceiling bounds MLX's allocator, not the process, and iOS's
real jetsam threshold sits above our 80%-of-available figure. That is exactly
`KNOWN-FAILURES.md` 1c, and it means kv4 buys context by spending the safety
margin. Shipping it as an automatic policy needs the device-aware KV budget
first, so the failure mode is a structured refusal rather than a kill.

Reproduce:

```bash
# fp16 baseline
ios/BenchHost/run-context-ladder.sh --device <udid> \
  --model mlx-community/Qwen3-1.7B-4bit --rungs 16384,32768,49152,65536
# kv4
TEST_RUNNER_MLXCAT_KV_BITS=4 TEST_RUNNER_MLXCAT_QUANTIZED_KV_START=0 \
  ios/BenchHost/run-context-ladder.sh --device <udid> \
  --model mlx-community/Qwen3-1.7B-4bit --rungs 16384,32768,49152,65536
```

The run prints `BENCHHOST kv-quantization: bits=4 group=64 start=0` next to the
load line. **Check it.** Before 2026-08-26 the producer never passed the policy,
so this exact command would have measured fp16 and called it kv4.

## Promoting rows

Rows arrive invalid by design. To promote:

1. Confirm the conditions held for the rows you are promoting.
2. Flip `valid_for_leaderboard` on those rows, append into `bench/results/`.
3. Re-render: `python3 bench/leaderboard.py && python3 bench/timeline.py`.

The leaderboard is stratified platform → device, so `iPhone17,x` and
`iPhone18,x` land as their own sections. **An iPhone is never compared against
a Mac.**

## Other engines on iOS

llama.cpp (GGUF, different artifacts), Google LiteRT-LM (gemma only), and
MLC-LLM each need their own host app to be benched honestly. Candidates for the
BenchHost pattern later; nothing else is headless-benchable on iOS today.
