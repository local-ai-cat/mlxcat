# GOAL — Finish the omlx-parity work queue

You are a delegated worker under an automated orchestrator (a separate AI model)
that will adversarially verify every milestone. When blocked or facing an
ambiguous design decision, emit `NEEDS_INPUT: <question>` and stop — never guess.
The human operator and the governor are watching this session live.

## Ground rules

- Repo: `/Users/timapple/Documents/Github/mlxcat-native`, branch off
  `feat/omlx-parity` → create `feat/parity-finish`. One commit (or a few
  focused ones) **per milestone**; conventional commit messages.
- Baseline: **201 tests green** (`swift test`) on the current tip. That number
  only goes up. Run `swift test` before your first change to confirm the
  baseline on your machine state.
- After EACH milestone: run `swift test`, report the result + a 3-line summary,
  then **pause for governor review** before starting the next milestone.
- Never push, never open PRs. Exception: M2b explicitly pushes one branch to
  `atlas-open-sources/mlx-swift-lm` (an owned org) — and only that.
- Do not touch the `mlx-swift-lm` pin in Package.swift except as M2b specifies.
- Code style: match the existing sources (see `Sources/MLXCat/TrackB/*` for
  engine idiom, `Sources/MLXCatHTTP/*` for route idiom). No stray comments
  explaining your changes; comments only for non-obvious constraints.
- Tests are the deliverable as much as code — every milestone lists its test
  requirements; weakening/skipping an existing test to go green is forbidden.

## Work queue (strict order)

### 1. M6d-a — MCP stdio timeout (severity: live bug)
A hung MCP server currently hangs the request forever.
- Per-call deadline on every stdio JSON-RPC round-trip (default 30s,
  config-overridable via the existing `--mcp-config` schema — add a `timeoutMs`
  field, per-server).
- On timeout: kill the subprocess, surface a clean tool-error result into the
  chat flow (the model sees "tool failed: timeout", the request completes),
  respawn lazily on next use.
- Tests: fake stdio server that never replies → request completes with the
  error tool result within deadline; server that replies slowly-but-in-time →
  works; respawn-after-kill works.
- Files: `Sources/MLXCatHTTP/MCP.swift` (+ its tests).

### 2. M6d-b — dialect tool-merge
`/v1/chat/completions` merges MCP-config tools into the model's tool list;
`/v1/messages` and `/v1/responses` use only request tools. Apply the same merge
in both dialect routes. Tests: MCP tool visible through each dialect.

### 3. M1b — thinking_budget + Harmony channel split
- `thinking_budget`: request param (chat + both dialects) that caps reasoning
  tokens — implement as a logits processor / sampler hook: once the budget is
  reached inside a `<think>`/reasoning span, force-close the span (inject the
  close-tag token sequence) so generation proceeds to the answer. Follow how
  omlx models it (reference checkout: `/Users/timapple/Documents/Guest/omlx`,
  see `thinking_budget` in its engine) — match its request field name/shape.
- Harmony (gpt-oss) channel split: extend
  `Sources/MLXCatHTTP/ThinkingParser.swift` to parse gpt-oss channel markers
  (`<|channel|>analysis` → `reasoning_content`, `final` → `content`,
  commentary per omlx's mapping) in both stream and non-stream paths.
- Tests: parser unit tests for channel sequences incl. split-across-chunks;
  budget test with a fake tokenizer (budget 5 → reasoning span closed ≤ 5
  reasoning tokens).

### 4. M2b — gemma-4-qat loader fix (in the fork)
The QAT gemma-4 checkpoints (`mlx-community/gemma-4-E2B-it-qat-4bit`, E4B
variant) fail at load in mlx-swift-lm — `k_proj` weight mismatch from the QAT
export layout.
- Work in the fork checkout: `/Users/timapple/Documents/Github/mlx-swift-lm-vlmfix`
  (branch `feat/vlm-batched-ropeoffset` is already pushed; create
  `feat/gemma-qat-loader` from it).
- Reproduce the load failure (models are in `~/Library/Caches/models/`),
  diagnose the sanitize/remap needed in the Gemma loader, fix minimally.
- Verify: both qat models load and greedy-generate coherently via a small
  harness or the mlxcat binary. Push the branch to
  `https://github.com/atlas-open-sources/mlx-swift-lm.git` (remote `github`),
  then update mlxcat's `Package.swift` revision pin to the new SHA and run
  the full `swift test`.
- Execution note: governor model sweep verified both
  `mlx-community/gemma-4-E2B-it-qat-4bit` and the E4B QAT variant load and
  generate coherently on this parity branch. Treat M2b as
  cannot-reproduce/already-fixed after the fork pin; do not chase further.
  Coverage remains through the fork gated QAT load test plus the recorded
  mlxcat greedy smoke for both QAT checkpoints.
- If the mismatch turns out to be quantization-semantics (not naming) and needs
  real dequant work: `NEEDS_INPUT` with your diagnosis instead of hacking.

### 5. M5b — /v1/rerank (RAG prep — promoted)
- omlx is the contract reference (`/Users/timapple/Documents/Guest/omlx`,
  `api/` routes): request `{model, query, documents[], top_n?}` → response
  with per-document `relevance_score` + index, sorted. Match its JSON shape
  exactly (the parity harness will diff it later).
- Implementation: rerank models (Qwen3-Reranker-class) score query+doc pairs.
  Follow how omlx runs them (cross-encoder → score). Wire through the model
  pool as its own model type (mirror how embeddings landed in M5 —
  `Sources/MLXCatHTTPServer/` embeddings wiring is the precedent).
- If MLXEmbedders/mlx-swift-lm lacks a usable cross-encoder path:
  `NEEDS_INPUT` with the options you found rather than silently stubbing.
- Tests: route contract tests (shape, sort, top_n, 404 unknown model, 400
  empty documents); scoring smoke behind the metallib GPU gate like
  `BatchCacheShapeTests`.

### 6. M7b-a — regex constrained decode
- `structured_outputs.type == "regex"` currently 400s. Implement true
  constrained decode: compile the regex to a DFA over UTF-8 bytes/characters,
  walk states as tokens are accepted, mask logits to tokens whose text keeps
  the DFA alive (reuse the `JSONGrammarMatcher` machinery pattern —
  `Sources/MLXCat/TrackB/Grammar/` — incl. the vocabulary index and the
  greedy-only rejection fast path exactly as `JSONGrammar` does).
- Scope: a pragmatic regex subset (literals, classes, `.`, `*+?`, `|`,
  groups, bounded repetition) — reject unsupported constructs with 400 +
  clear message, matching omlx's error shape.
- EOS gated on DFA accept-state, same as JSON's isComplete.
- Tests: mirror `JSONGrammarMatcherTests` structure — matcher unit tests +
  the two GPU sampler tests (invalid argmax → masked pick; valid argmax kept).

### 7. M7b-b (stretch — only if all above are VERIFIED) — GBNF/CFG sampler
Swift port of a llama.cpp-style GBNF grammar sampler for
`structured_outputs.type == "grammar"`. Ask the governor before starting this
one — it may be deferred to its own session.

## Explicitly NOT yours
- M8b/M8c (audio registry + consensus) — needs an app-side design handshake;
  the governor owns it.
- M-sweep (model sweep) — runs on the governor's machine with memory
  discipline; do not launch multi-model harness runs. **Never start more than
  one model server at a time; kill any server you start before exiting a
  milestone** (this machine OOM'd today from exactly that).
- Anything in the Local-AI-Chat app repo.

## Definition of done, per milestone
Code + tests + `swift test` green + summary posted + governor review passed.
The governor runs red→green checks, diffs your tests for tampering, and
independently reviews the diff — write accordingly.
