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

---

## Open — ranked

### 1. One admission in flight, and prefill stretched across scheduler steps
**They:** mlx-serve admits up to **16 slots per tick**
(`guest/mlx-serve/src/scheduler.zig:3561`, `:1982`) and runs each prefill **to
completion**, injecting a decode tick between admitted slots (`:3622`) and at each
chunk boundary (`guest/mlx-serve/src/generate.zig:1937`). Its comment names our
exact failure: *"without this every slot's first token waits for the LAST slot's
prefill (the TTFT staircase collapse)"* (`scheduler.zig:3617`). omlx admits
multiple per step (`guest/omlx/omlx/scheduler.py:7457`); vllm loops admitting
while a token budget remains (`guest/vllm/vllm/v1/core/sched/scheduler.py:640`).
**We:** exactly one `admissionInProgress` (`Sources/MLXCat/TrackB/Scheduler.swift:49`),
FCFS `waiting.removeFirst()` (`:272`), and mid-prefill we return `nil` (`:486`) so
the step falls through to one decode. Request #2 cannot begin prefill — not even
its first chunk — until #1 finishes.
**Measured cost:** TTFT 727 / 1614 / 3122 / 5883 ms at c1/c2/c4/c8 — linear in N,
recorded in our own `bench/matrix.json:194`. Against mlx-serve at c8 we are 13×
worse while per-stream decode is fine.
**Note:** mlx-serve does **not** pack prefills — it is batch=1 per slot, like us.
Packed prefill is not the prerequisite; multi-admission plus run-to-completion is.

### 2. Busy-prefill chunk width
**They:** mlx-lm and omlx use **2048** (`guest/mlx-lm/mlx_lm/generate.py:1509`,
`guest/omlx/omlx/scheduler.py:1304`); mlx-serve uses **8192**
(`guest/mlx-serve/src/generate.zig:34`) with per-model caps —
MoE → 4096, composed-causal head-dim-256 → 2048 (`generate.zig:111`).
**We:** **512** (`Scheduler.swift:421`). Combined with (1) that is one decode tick
plus one actor round trip per 512 prefill tokens, against mlx-serve's one per
8192 — a 16× difference in scheduling overhead.

### 3. The prefill logits tensor
**They:** mlx-lm prefills `y.size - 1` tokens and lets the single remaining token
produce logits, and evaluates only the cache — `mx.eval([c.state for c in cache])`
(`guest/mlx-lm/mlx_lm/generate.py:579`, `:430`). MLX is lazy, so the prefill
chunks' logits are never computed at all; the logits tensor is always
`[1, 1, vocab]`.
**We:** take `output.logits` from the final chunk (`Scheduler.swift:461`), so it is
`[1, N, vocab]` — up to `[1, L, vocab]` on the single-pass path.

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
