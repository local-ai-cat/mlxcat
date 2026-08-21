# GOAL: mlxcat engine hardening (vLLM-derived)

Autonomous Codex goal session. Port vLLM's proven engine patterns into mlxcat, one
focused+tested commit at a time, looping with full test gates on **Mac + iPhone 16** until
every gate is green. Reference engines are Python (do NOT run them — translate the LOGIC to
Swift): **vLLM** `/Users/timapple/Documents/Guest/vllm` (canonical), **omlx**
`/Users/timapple/Documents/Guest/omlx` (cross-ref).

## Repo & branch
- Engine: `/Users/timapple/Documents/Github/mlxcat-native`, branch `feat/parity-next`
  (= local-ai-cat/mlxcat main; PUBLIC — builds/pushes anywhere). Commit per item; do NOT
  push until the governor integrates (pin bump for the app).
- App (for the iPhone gate): `/Users/timapple/Documents/Github/lac-iostest`, branch
  `feat/mlxcat-migration-gate`.

## Prerequisites (governor lands these BEFORE the loop starts)
1. Tool-call parser registry merged on `feat/parity-next` (`ea486ac` + the `<function_call>`
   follow-up) + app pin bumped. This goal session STARTS from that base and must keep it green.

### Tool-call baseline (verified 2026-07-07, 16-model Mac fleet) — the gate's definition of "green"
The per-model tool-parser registry took tool-call from 8/16 → 12/16. Tool-call passes on EVERY
tool-CAPABLE served model (Qwen3 family, Qwen2.5-Coder, Llama-3.2-3B, gemma-4-E2B/E4B, gpt-oss,
Ornith, Qwen3.6-27B/35B, Qwen3-Coder-30B). The remaining failures are **model-compliance
limitations, NOT engine bugs** (raw outputs captured):
- **Llama-3.2-1B** — regurgitates the tool schema instead of calling.
- **DeepSeek-R1-1.5B / 7B** — reason indefinitely, never emit a call (`finish=length`).
- **Qwen2-VL-2B** — refuses ("I don't have the capability to call the weather tool").
**Gate rule:** the tool-call assertion applies ONLY to tool-capable models; the four above are
EXCLUDED from the tool-call check (re-evaluate only if a better checkpoint ships). Do NOT chase
them — there is no valid call to parse. Every OTHER capability (structured JSON, reasoning, stop,
determinism, greedy) already passes on all 16 models.

## Environment & test harness (already built — use these)
- **Mac fleet gate:** boot `.build/release/mlxcat-http --model-dir ~/Library/Caches/models
  --memory-ceiling-bytes <45G>`; run `scratchpad/correctness_check.py <base_url> <model> <timeout>`
  per served model via `scratchpad/mac_campaign.sh`. Served models (bare leaf ids) span
  Qwen3/2.5/Coder, Llama-3.2, DeepSeek-R1, Qwen2-VL, gemma-4, gpt-oss, Ornith, and the 27/30/35B.
- **iPhone 16 gate:** app test `MigrationGateDeviceParityTests/test_nativeCapabilityCorrectness`
  with env `LOCALAI_MIGRATION_GATE_CORRECTNESS=1 LOCALAI_MLXSERVE_EMBED_NATIVE=1
  LOCALAI_MIGRATION_GATE_MODELS=<model>`; device id 89F5157D-...; `-skipMacroValidation`;
  model must be pre-seeded (download-crash landmine).
- **Discipline (STRICT):** serialize ALL model loads; before each load gate on
  `memory_pressure -Q` > 40% free; one model at a time. The Mac has 128GB; the iPhone ~6GB
  (small/medium models only).
- Codex: `model_reasoning_effort: high`, read-only sandbox for review, cwd = the mlxcat worktree.

## Work items — sequenced (each = one focused, tested commit)

### Quick correctness wins (independent — do FIRST)
**W1. Per-request RNG (LIVE BUG).** Seeded requests call `MLXRandom.seed()` → GLOBAL state
mutation; in a mixed batch the last seeded request skews ALL rows' draws
(`Sources/MLXCat/TrackB/BatchGenerator.swift:191-195`, incl. XTC uniform `Sampling.swift:504`).
vLLM uses a per-request generator. Fix: per-row `MLXRandom.key(seed)` passed explicitly into
categorical/uniform draws; unseeded rows keep global/default.
- DoD: new test — one batch with {seed=A, seed=B, unseeded}: each seeded row is reproducible
  across runs AND independent of the others; unseeded row unaffected. `swift test` green.

**W2. Windowed-KV capability flags (landmine).** `Scheduler.swift:539-543` sniffs cache
semantics by matching type NAME strings ("rotating"/"circular") — same class as the omlx stream
segfault (silent break on rename). vLLM drives this off declared per-layer KV specs. Replace with
an explicit per-model capability flag (from model config/type, not a string match).
- DoD: gemma (windowed) detected via flag not string; existing behavior preserved; a test asserts
  the flag path. (Re-enabling gemma prefix caching is W7, not here.)

**W3. min_tokens + logit_bias logits processors.** Real OpenAI-dialect gaps, ~40 lines each.
vLLM `vllm/v1/sample/logits_processor/builtin.py:119` (LogitBias), `:165` (MinTokens — mask EOS
until min reached). Slot into `TokenSampler.sample`; wire request fields through the dialect.
- DoD: unit tests (min_tokens masks EOS below threshold; logit_bias shifts a token's logit);
  dialect parses `min_tokens`/`logit_bias`.

### Scheduler core (sequenced — W4 is the foundation for W5)
**W4. Token-budget scheduling + chunked prefill (the freeze fix).** Prefill is MONOLITHIC:
`admitWaiting → prepareForInsert → prefillTokenRange` runs the WHOLE prompt before returning
(`Scheduler.swift:158-236, 402-420`), stalling every active decode. vLLM has no phases — each
request's computed-tokens catches up under a per-step token budget + long-prefill threshold
(`vllm/v1/core/sched/scheduler.py:396-530`, design note 398-407). Make prefill resumable per-step
state (mlxcat already chunks by `prefillStepSize` — it just never yields between chunks).
- Handle landmine: `continue`-not-`break` on temporarily-unschedulable running requests
  (scheduler.py:514-530) — FCFS relaxation to avoid head-of-line blocking.
- DoD: a test where a long-prompt admission runs CONCURRENTLY with an active decode → the decode
  keeps producing tokens (no multi-second stall); token-budget respected; outputs unchanged vs
  monolithic for the same inputs (determinism preserved).

**W5. Preemption + resume + incremental block caching.** allocate-fail / watchdog soft-pressure
→ preempt youngest/lowest-priority row: free KV, computed-tokens=0, prepend to waiting; resume
re-hits the prefix cache (`scheduler.py:534-582, _preempt_request:1145-1167`). AND publish full
blocks as they fill INCLUDING generated tokens (`block_pool.py cache_full_blocks`) so preemption
is cheap and assistant turns are reusable (agent multi-turn win) — today only the prompt is cached
(`Scheduler.swift:86-93, 435-452`).
- Handle landmines: (a) stale in-flight output frames after preempt/cancel must be counted +
  DROPPED, not delivered (`async_scheduler.py:52-60`); (b) admission deadlock — no-forward-progress
  reservations (SSD prefix-restore holding memory on I/O) admitted only under (free − in-flight
  reservations) (`scheduler.py:899-905`).
- DoD: under induced memory pressure, a row is preempted then resumes and completes correctly
  (byte-identical to no-preempt run); generated-token cache hit on a second same-prefix turn;
  no stale frame leaks; no deadlock with two concurrent SSD restores.

### Stretch (only if W1–W5 green with time left)
**W6. ngram/suffix speculative decoding** (model-free — fits iOS, biggest throughput win).
`vllm/v1/spec_decode/ngram_proposer.py`, `suffix_decoding.py`; verify via `sample/rejection_sampler.py`.
Prereq W4. DoD: throughput up on a repetitive-output prompt; outputs identical (spec decode is exact).
**W7. Predictive KV sizing** feeding admission (`worker/gpu_worker.py:430-520`,
`kv_cache_utils.py:919`) — KV bytes/token is a deterministic formula from config; makes
`checkAdmission` honest (callers pass 0 today) + re-enables gemma prefix cache (uses W2 flag).
**W8. Off-critical-path grammar masks** (`v1/structured_output/__init__.py:69-78,199-258`) —
compile/fill on a thread pool overlapping the forward pass. **W9. Batched sampling**
(`v1/sample/sampler.py`) — vectorize penalties/top-k/p across the batch (matters at batch ≥4).

## SKIP (do not attempt — no fit for single Apple-GPU, ~6-model, on-device+sidecar)
Distributed (TP/PP/EP/NCCL/DP), disaggregated prefill / KVConnector / remote-KV, PagedAttention
CUDA kernels (no MLX kernel — our dense per-row batch + hash-block storage is the correct
adaptation), CUDA graphs / torch.compile, LoRA scheduling, encoder-cache budgets, Mamba alignment,
all of legacy `vllm/core/` (v0).

## Test GATES (the loop's definition of done — ALL must be green to finish each item AND overall)
1. `swift build` + `swift test` green (existing + the item's new tests).
2. **Mac fleet:** `mac_campaign.sh` — correctness PASS on every served model (no regression from the
   tool-call-green baseline).
3. **iPhone 16:** `test_nativeCapabilityCorrectness` green (native engine) on the fitting models
   (≥ Qwen3-0.6B, a reasoning model, a VLM).
4. **Determinism:** greedy determinism unchanged; W1 must make seeded batches MORE correct, never
   less deterministic.
5. **No regression:** the migration-gate + tool-call fleet stay green.
6. **Perf (W4/W6):** an assertion that chunked prefill keeps concurrent decode progressing / spec
   decode raises throughput — measured, not assumed.

## Loop protocol
One item at a time → implement → `swift test` → Mac fleet (memory-gated) → iPhone gate → commit
(scoped, conventional message) → next. After each risky item (W4/W5), run a Codex self-review pass
("hunt for holes in the fix itself — races, stale frames, deadlock") before moving on. Stop and
surface to the governor if a gate can't go green after two attempts. Do NOT push; the governor
integrates (pin bump + app re-validate).
