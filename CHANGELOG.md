# Changelog

All notable changes to mlxcat. Format follows [Keep a Changelog](https://keepachangelog.com/);
the project is pre-1.0 and pinned by revision from its host app, so "releases"
are tags on `main`.

## [Unreleased]

### Added
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

### Changed
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
