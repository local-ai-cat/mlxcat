# Levers

Every optimisation in this engine ships as a **toggle with a measured
trade-off and a documented way out**. Not because flags are tidy, but because
an optimisation you cannot turn off is an optimisation you cannot walk away
from — and several of these have already been walked away from.

This file is the registry. A lever that is not listed here is effectively
un-removable: nobody can exit what nobody can find. Twelve flags existed in
code before this file did, with seven scattered mentions across three other
docs and no single list.

## Rules

1. **Default = today's behaviour.** A new lever ships off. The default path
   stays byte-for-byte what it was, so "turn it off" is always a real answer.
2. **State the trade-off, with a number.** "Faster" is not a trade-off.
   `+1.0% gemma decode, nil on qwen3_5` is.
3. **Additive where possible.** If a lever is a new file plus a branch at one
   call site, removing it is deleting a file. If it rewrites the default path,
   it is not a lever — it is a rewrite wearing a flag.
4. **Name the exit.** Every row below says how to remove it, not just how to
   disable it.
5. **An unmeasured lever is a liability.** If nobody has priced it, it is a
   guess with a config surface. Price it or delete it.

## Registry

| Lever | Default | What it trades | Site |
|---|---|---|---|
| `MLXCAT_SINGLE_ROW_SCALAR_ROPE` | on (`0` disables) | One-row caches report RoPE position as `.scalar`, eligible for MLX's RoPE fast path. **Measured +1.0% gemma decode (86.19 vs 85.31 tok/s), nil on qwen3_5.** | `TrackB/BatchCache.swift:127` |
| `MLXCAT_PREFILL_LAST_TOKEN_ALONE` | `!usesWindowedKVCache` (`always`/`never` force) | Last prompt token prefilled alone so the sampled forward is `[1,1,vocab]` not `[1,chunk,vocab]`. Windowed caches diverge, hence the capability gate. | `TrackB/Scheduler.swift:139` |
| `MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES` | per-model list (`all` overrides) | Chunked vs single-pass prefill. **Measured at 16k: gemma-4-12B −13.8 GiB, Qwen3.8-27B −14.0, gpt-oss-20b −9.8, but Qwen3-Coder-30B-A3B +15.6** — which is why it is a list, not a bool. | `MLXCatNative/NativeModelEngine.swift:1185` |
| `MLXCAT_DECODE_PIPELINING` | on (`0` = fully synchronous decode) | Launch-ahead pipelining in the decode loop. | `TrackB/BatchGenerator.swift:604` |
| `MLXCAT_DECODE_CLEAR_CACHE_STEPS` | interval (`0` disables) | Periodic `clearCache` during decode: memory vs per-step cost. | `TrackB/Scheduler.swift:175` |
| `MLXCAT_IDLE_CLEAR_CACHE` | policy default (`0`/`1` force) | Whether the buffer cache is released when the scheduler goes idle. | `TrackB/Scheduler.swift:176` |
| `MLXCAT_CACHE_LIMIT_BYTES` | unset | MLX buffer-cache ceiling. | `Pool/MemoryGuard.swift:133` |
| `MLXCAT_KV_BITS` | off (`4` or `8`; anything else off) | KV cache quantization: memory vs quality//throughput. | `TrackB/KVQuantizationPolicy.swift:105` |
| `MLXCAT_KV_GROUP_SIZE` | see site | KV quantization group size. | `TrackB/KVQuantizationPolicy.swift:105` |
| `MLXCAT_QUANTIZED_KV_START` | see site | Token threshold before KV quantization engages. | `TrackB/KVQuantizationPolicy.swift:106` |
| `MLXCAT_UNSERIALIZE_MODEL_TYPES` | none | Lifts per-family decode serialization. Widening it is a correctness claim — the families are serialized because they failed batching, not for caution. | `MLXCatNative/NativeModelEngine.swift:793` |
| `MLXCAT_BATCH_INVARIANCE_MODELS` | built-in list | Which models the batch-invariance gate covers. Test surface, not a runtime lever. | `MLXCatNative/NativeModelEngine.swift:1152` |

## Harness levers (measurement, not runtime)

These do not change what the engine does — they change whether a measurement of
it means anything. Added 2026-08-26 after a night in which SIX apparently-red
parity cells turned out to be the instrument, not the engine.

| lever | default | trade-off, with the number | exit |
|---|---|---|---|
| `--model-settle-floor-s` | `10` (60 used for the clean pass) | Unconditional pause after each model before the next loads. WITHOUT it, Qwen3.5-4B `longgen/c1` measured **2040 ms / 509 tok/s** when it ran second in a pass and **1291 ms / 807 tok/s** when it ran first — 58% slower, with peak DOWN 4.47→4.11 GiB. Cost: that many seconds per model boundary. | `0` restores the pre-2026-08-26 back-to-back behaviour. |
| `--model-settle-s` | `30` | Upper bound on the settle, after the floor, while free memory is under `--min-free-pct`. | `0` disables. |
| `--cell-settle-s` | `15` (20 used for the clean pass) | Pause between cells of the SAME model so the engine drains to idle and hands back its buffer cache. Cells share one process (it is per-model, not per-cell). Recovered gemma-4-E2B `longgen/c2` outright (TTFT 433→336 ms) and moved Qwen3.5-4B `longgen/c2` from 0.98x to 1.07x vs best-other. | `0` restores back-to-back cells. |

**Why the defaults are ON despite costing wall-clock.** A pass that is fast and
measures queue position is worth less than a pass that is slow and measures the
engine. On a 12-model sweep the settles add roughly `60s x models +
20s x cells`. If that is ever too expensive for a routine pass, turn them down
and know that cross-pass comparisons stop being safe — do not turn them down and
then compare against a baseline recorded with them on.

**The residual, unfixed.** `bench/parity.py` keys a cell by
`(device, model, tier, cache_mode, concurrency)` and NOT by position in the pass,
so it can still compare a first-position row against a fifth-position one. The
settles reduce the contamination; they do not make position part of the cell
identity. Recording position in the row and refusing cross-position comparisons
is the real fix and is not done.

## Levers that were exited

Kept as evidence that the exit works, and so a measured-negative idea is not
re-proposed from scratch:

* **Fused SiLU product** (`exp/fused-silu`, mlxcat pin) — REJECTED 2026-08-25.
  Qwen3.5 failed the width-8 token-exactness gate (compile() rounds the fused
  product differently for serial `[1,1,d]` vs batched `[8,1,d]`); the Qwen3
  half's apparent +8.8% collapsed to +0.5% under an 8-run confirm (control
  ±1.4%). Branch retained as the record; nothing shipped.
* **Encode-thread overlap** (`exp/encode-thread`) — three hiding strategies all
  measured dead against MLX's C++ allocator/device mutexes. Do not re-attempt
  without an MLX-core change.

## Planned

* `MLXCAT_PREFILL_BATCH` (pkg 115 lever A) — default `1` = today's serial
  admission, byte-for-byte. Values >1 stack that many cold admissions into one
  prefill forward. Exit: set to 1, or delete `TrackB/PrefillStack.swift` plus
  the additive `prepare(rightPadding:)`/`finalize()`/`emptyBatched` members —
  the default path is a separate branch and is not modified.
