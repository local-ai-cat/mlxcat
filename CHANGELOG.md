# Changelog

All notable changes to mlxcat. Format follows [Keep a Changelog](https://keepachangelog.com/);
the project is pre-1.0 and pinned by revision from its host app, so "releases"
are tags on `main`.

## [Unreleased]

### Changed
- **MLX core 0.32.4.** `mlx-swift` moves to `15405f9` and `mlx-swift-lm` to
  `d28ad59` (both atlas-open-sources). The mlx-swift fork keeps strict
  Metal 3 compilation of the attention sources in their source, generated and
  JIT forms. The mlx-swift-lm fork keeps the Gemma 4 per-row RoPE fix.
  `swift-jinja` 2.x is a direct
  dependency for ordered tool declarations. The qualification, Metal 3 gate and
  old-vs-new benchmark are in `docs/inkling/CORE-0.32.4.md`.
- **Precise embedded Metal library on macOS.** The pinned Cmlx builds a
  separate `mlx-swift_CmlxMacMetal.bundle` with `-fno-fast-math`. The native
  loader prefers it to the compatibility bundle. A host app's default Xcode
  Metal settings (fast math) no longer change the shader arithmetic, so the
  embedded engine matches standalone `mlxcat-http` and mlx-lm.
  `scripts/build-metallib.sh` uses the same compiler wrapper.

### Fixed
- **A loopback host binds the loopback address.** `--host 127.0.0.1` (and
  `localhost`, `::1`) used to bind the wildcard address and only filter
  accepted connections by interface; `lsof` showed `*:<port>`. The listener now
  binds the address itself (`requiredLocalEndpoint`); `SidecarLifetimeTests`
  asserts the listening address.
- **`mlxcat-http` follows its launcher out.** A launcher that sets
  `MLXCAT_PARENT_PID` to its own pid is watched with a kqueue `NOTE_EXIT`
  source, and the server exits when it goes, so a crashed or SIGKILLed host app
  no longer leaves an orphan holding the port (Local AI Cat, 2026-11-26: an
  orphan held 11435 for 31 hours). An environment variable, not a flag, so a
  host may set it for any build.
- **Singleton sampling matches batched sampling on MLX 0.32.4.** The new
  categorical sampler takes a different path for a scalar result. Singleton
  rows now pass rank-one logits, so an unseeded request batched with seeded
  ones samples the same tokens.
- **Compiled model closures stay executor-local.** Qwen's compiled decode and
  Inkling's compiled experts use the new compile overload for closures that
  capture model state.
- **Inkling vision rotary frequencies use `metal::precise::pow`.** This
  restores zero-error image fixtures.
- **Qwen3.5 tool declarations render in reference order**, and schema literals
  render stably. The real tool fixtures match every prompt ID
  (`NativeQwenToolSchemaTests`).
- **Incremental `swift test` under Xcode 27.** The test bootstrap
  (`MLXMetalRuntime`) no longer copies its metallib into the signed
  `MLXCatTests.xctest` bundle, which made the next incremental run fail to
  re-sign. The library lives in `.build/mlxcat-metal-runtime/` and reaches MLX
  through its relative `default.metallib` fallback. A `.build` from before
  this change can still hold the old copies in
  `MLXCatTests.xctest/Contents/MacOS`. Delete them once (or build clean): the
  re-sign fails at build time, before any test code could remove them.

### Changed
- **Session prefix parity gate.**
  `PrefixSchedulerIntegrationTests.testSessionPrefixCacheMatchesCacheDisabledForExtendedPrompt`
  no longer demands bit-identical greedy output from cached and cold runs. A
  near-tie flip at step 8 on Qwen3-0.6B comes from kernel choice: the slot's
  last prompt position comes from the decode kernel. The gate now asserts:
  - the reused KV is bit-exact at every prefilled position;
  - the first token disagreement, if any, falls on a near-tie.

  Evidence in `docs/KNOWN-FAILURES.md` §4.
- **`MLXCAT_WATCHDOG_TRIM_AFTER_EVICT` is ready to flip, still off.** Its
  default is one constant, `MemoryWatchdogConfiguration.defaultTrimsAfterEviction`.
  `0`/`false`/`never` now force the lever off explicitly. `MemoryWatchdogTests`
  runs the reclaim ladder with the lever off and on.

### Added
- **Native Inkling 2 Mini engine** (`Sources/MLXCatNative/Inkling/`). The
  native loader serves Inkling 2 Mini checkpoints, NVFP4/MXFP8 and affine
  packs, through the existing embedder and `mlxcat-http`:
  - the 24-layer, 48-visit loop with per-visit attention and convolution
    caches, routed experts, the vision encoder and the publisher's chat
    template ported to Swift;
  - grouped NVFP4 prefill with fused unpacking and expert scaling, a compiled
    singleton expert graph and a mixed-storage vocabulary GEMV, each checked
    for exact output against the unfused path;
  - typed reasoning, text and tool-call framing in `InklingOutputParser`;
  - exact prefill checkpoints reused across turns, and declared affine
    layouts validated before loading.

  `mlxcat-inkling-parity` is the same-Mac reference verifier against
  mlx-vlm (`docs/inkling/PARITY.md`). Plans, parity, decode and benchmark
  records are in `docs/inkling/`.
- **Named text session prefixes persist on SSD**, and rolling text prefix
  snapshots restore at exact boundaries (`SessionPrefixDiskStore`).
- `Qwen3_5Tokenizer` loads through the existing Qwen BPE, so Qwen3.5-family
  survey checkpoints that declare that class no longer fail to load.
- **Cat inference benchmark results, all four models**, in `bench/cat/`
  (M4 Pro, default configuration). Hybrid prefix reuse takes Qwen3.5-4B's 8k
  replay TTFT from 15.1 s to 0.58 s. Non-hybrid exact replays get no reuse with
  the levers off.

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
