# GOAL — close the remaining parity gaps (post-17/17 sweep)

You are a delegated worker under an automated orchestrator that verifies your
work. If blocked, emit `NEEDS_INPUT: <question>` and stop — never guess.

Repo: `/Users/timapple/Documents/Github/mlxcat-native`, branch
`feat/omlx-parity` (continue on it). Harness work happens in the worktree
`/Users/timapple/Documents/Github/mlxcat-native-harness` on
`feat/omlx-parity-harness`. omlx reference source:
`/Users/timapple/Documents/Guest/omlx`. NEVER push to any remote. Commit per
milestone, author `76924051+atlascodesai@users.noreply.github.com`. Run gates
per milestone and report before moving on (short report, keep going — no
approval pauses). If your context runs low, commit everything and emit
HANDOFF-READY with a state summary.

HARD RULE: never run multi-model sweeps or load >1 large model concurrently —
this machine OOM'd once. Single-model tests only; the orchestrator runs sweeps.

## M10a — session prefix cache (TOP PRIORITY, biggest win)
Native re-prefills every request; omlx reuses session KV. Build session-level
prefix reuse in the batcher (NO SSD tier yet — that's M10b later):
- Cache slots keyed by an explicit session key (accept BOTH a
  `cache_session` JSON param and `X-Cache-Session` header) + an anonymous
  longest-prefix lane for keyless clients.
- Planner semantics: mirror mlx-lm trim/extend/reset — different model or no
  shared prefix → reset; cache is strict prefix of prompt → extend (prefill
  only the suffix); prompt diverges mid-cache → trim back to common prefix
  then extend. The app's planner
  (`Local-AI-Chat` Packages/LocalAIServing/Sources/LocalAIServing/PromptCachePrefix.swift)
  is the proven Swift reference for the DECISION logic — port the decisions,
  adapt storage to the batch engine's KV cache types.
- Pool citizenship: slot memory is accounted; LRU eviction under pressure;
  slots die with their model's unload. Failure-path: a request that throws
  after mutating a slot must remove/roll back that slot (see the app's
  anonymous-slot desync bug — don't repeat it).
- Concurrency: slots must be safe under the concurrent batcher — a slot in
  use by one request must not be trimmed/extended by another (per-slot lease
  or copy-on-divergence; document the choice).
- Tests: unit (planner decisions, eviction, failure rollback, concurrent
  lease) + an integration proof: same-session second request with extended
  prompt does NOT re-prefill (assert prefill token count or TTFT delta).
- Harness: add a TTFT-delta cell (same session, 2 requests, second TTFT must
  be < 40% of first on a >2k-token prompt, small model).

## M6d-b — MCP SSE/streamable-HTTP client transport
Remote MCP servers over SSE + streamable-HTTP (spec-current transport).
Auth headers from config, reconnect w/ backoff, per-call deadline reuse from
M6d-a. Tests against an in-process mock SSE server (no network).

## M5b — /v1/rerank
Route + backend over the MLXEmbedders infra. omlx's request/response shape is
the contract (read their rerank route). Unit tests with a fake scorer; gated
live test if a reranker model exists locally (check the store; if none, do
NOT download >200MB — leave the live test gated-off with a clear skip).

## M1-cosmetics
(a) `/v1/messages/count_tokens` exact via real chat-template tokenization
(drop `estimated:true`). (b) Streaming reasoning event names byte-match
omlx's on `/v1/responses` (diff their stream, rename ours). Harness cells
updated to assert both.

## Harness additions (worktree, after each engine milestone lands)
1. **Grammar diff cells**: same model/prompt/temp-0 + identical json_schema,
   regex, and GBNF grammar on BOTH servers → outputs validate AND are
   token-identical. Any divergence = mask discrepancy: record cell RED with
   both outputs.
2. **Grammar overhead bench**: decode tok/s grammar-on vs grammar-off per
   server (small + large schema); report the ratio table in report.md.
3. **Config axis**: boot both servers at non-default settings (omlx
   max_concurrent_requests=2 and chunked_prefill=true; native equivalent
   flags if present) and rerun the streaming + error axes.
4. **Audio diff cell**: investigate how omlx resolves STT model ids
   (omlx/api/audio_routes.py) and make the existing skip-cell pass with the
   already-downloaded `mlx-community/whisper-tiny` at
   ~/Library/Caches/models — a config/launch fix, NOT a big download.
5. **Perf fleet cell**: extend the bench axis to run per family on the SMALL
   models only (0.6B dense, 1.7B, gemma-E2B, Qwen3.5-4B hybrid) — one at a
   time, sequential.

## Gate per milestone
swift test (all targets) green · relevant pytest axis green · zero warnings ·
focused commits. Report: milestone, SHAs, test counts, one-line proof.
Order: M10a → harness TTFT cell → M6d-b → M5b → cosmetics → harness 1-5.
