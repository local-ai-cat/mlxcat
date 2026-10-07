# Changelog

All notable changes to mlxcat. Format follows [Keep a Changelog](https://keepachangelog.com/);
the project is pre-1.0 and pinned by revision from its host app, so "releases"
are tags on `main`.

## [Unreleased]

## 2026-10-07 — `pin/prefix-reuse-2026-10-07`

Everything on `main` since `pin/streaming-detokenizer-2026-09-08`, plus the
entries carried over from the old Unreleased section below.

### Fixed
- **A tiny `top_p` no longer masks every token.** `applyTopP` kept tokens whose
  ascending cumulative probability exceeded `1 - top_p`; with a tiny `top_p` in
  fp16/bf16 every token went to `-inf` and sampling returned an arbitrary token.
  The most likely token now always survives. Ported from ml-explore/mlx-lm#1912;
  `TopPSamplingTests`.
- **Prefix-lease leak.** A row that finished at admission (`max_tokens` 1, or
  EOS as its first token) never released its prefix lease, so its slot could
  never be fetched or evicted again. `PrefixLeaseBalanceIntegrationTests`.
- `scripts/donor-drift.sh` reads paginated commits with `--slurp` (it crashed on
  a 60-day window) and reports an unfound cursor as unknown, not 0.
- The HTTP SSE response header is built from one literal (Swift 6.4 could not
  type-check the six-term `+` chain).

### Changed
- **Hybrid prefix reuse is on by default** (`MLXCAT_HYBRID_PREFIX_REUSE=0` turns
  it off): hybrid models (Qwen3.5/3.8) resume a follow-up turn from recurrent
  checkpoints placed on a grid, token-identical to a cold run. A follow-up on a
  9K prefix went from 46 s to 3 s.

### Added
- Three levers, all **off by default** (`docs/LEVERS.md`):
  `MLXCAT_ADMISSION_CLEAR_CACHE` (clear MLX's buffer cache when a prefilled row
  joins the decode batch; from mlx-swift-lm#620), `MLXCAT_WATCHDOG_TRIM_AFTER_EVICT`
  (clear again after the memory watchdog evicts a model; from mlx-serve#637),
  `MLXCAT_PREFIX_FULL_MATCH_REUSE` (an exact prompt repeat reuses N-1 tokens
  instead of prefilling cold; the swama#123 rule).
- `docs/watch/` — `LEARNINGS.md`, `CHECKER.md`, `DISCOVERY.md` and per-repo read
  cursors; `donor-drift.sh` reports from the cursors.
- `mlxcat-mtp-probe` — single-stream native-MTP probe with an exactness gate.
- Bench: coding and tool-calling campaigns across engines, the
  `mlxcat-admission-clear` and `mlxcat-full-match` arms, more registered engines.

### Added (carried over from the old Unreleased section)
- `bench/` — same-transport benchmark harness (`run.py`), engine registry,
  model/context matrix, JSONL results and a generated `LEADERBOARD.md`
  (`leaderboard.py`, with a `--check` CI gate). Peak footprint sampled via
  `proc_pid_rusage`; quiet-machine guard; platform axis (macOS / iOS).
- `docs/ENGINES.md` — benchmarked engines + drift watchlist, per platform.
- `scripts/donor-drift.sh` + weekly `donor-drift` workflow — pins vs upstream,
  commits since pin with keyword hits, filed as one issue in this repo.
- `.github/workflows/ci.yml` — build + no-model test suite on GitHub-hosted
  Apple Silicon; leaderboard/derivation gate; shell parse gate.
- `scripts/nightly-models.sh` — every env-gated model suite wired to local
  models (so the gates stop skipping forever), for our own Macs.
- `MemoryBudgetTests` — env-gated 16k prefill+decode peak-footprint budget
  assertion (the pkg-102 ruling: a number measured once in a doc is not a gate).
- `CONTRIBUTING.md`, `SECURITY.md`, issue/PR templates, this changelog.
- `guest/` (gitignored) — the reference/competitor engine clones live with the
  checkout.

### Changed (carried over)
- README rewritten to describe the engine that exists (it still said
  "pre-implementation"); planning/history docs moved under `docs/history/`.

## 2026-08-20 — `a481734`
- Rename the module surface to `MLXCat*` / `mlxcat-http` to match the repo
  (hash seed and `owned_by` wire value intentionally unchanged).

## 2026-08-19 — `40b4cf5`
- Qwen3.8 hybrid caches supported on the native path.

## 2026-08-12/13 — perf batch + pin
- Pipelined continuous-batch decode steps, prebuild-before-wait, singleton
  KV-cache passthrough at width 1, SSE template instead of `JSONSerialization`,
  per-token CoW avoided in the scheduler; mlx-swift-lm pinned to `01472a78`.

## 2026-08-10/11 — parity repairs
- Serial greedy scheduler parity (complete-prompt admission, first token from
  prompt logits, speculation opt-in); request-aware reasoning channels;
  OpenAI error contract alignment; Llama tool-parameter preservation.

## 2026-07 — parity with oMLX (M0–M9)
- Batched decode + scheduler (Track B), tiered prefix KV cache (Track A),
  OpenAI/Anthropic/Responses dialects, grammar-constrained decode (JSON, regex,
  GBNF), tool-call parser registry, MCP, speech (`/v1/audio/transcriptions`
  via WhisperKit), rerank, embeddings, memory watchdog, ngram speculative
  decoding. See `PARITY.md` for the 2026-07-03 parity measurement.
