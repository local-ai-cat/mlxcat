# Prefix-cache + MCP fix round — feat/omlx-parity (adversarial review: BLOCKER)

Delegated worker under an automated orchestrator; if blocked emit
`NEEDS_INPUT`. NEVER push. Author `76924051+atlascodesai@users.noreply.github.com`.
Continue on `feat/omlx-parity`. MEMORY GUARD: one model at a time, never boot
mlxcat+omlx together, gate model boots on `memory_pressure -Q` >40%.

## BLOCKER — prefix cache stores POST-GENERATION KV → correctness corruption
`storeFinishedPromptCache` (Scheduler.swift:406) extracts the cache AFTER
generation advanced; BatchGenerator.swift:285 mutates cache with
generated-token context each decode step; BatchLayerCache.swift:246 extracts it
UNTRIMMED. So the stored "prompt" cache actually contains generated-token KV.
On reuse, SessionPrefixKVStore.swift:80 trims only `storedPromptLen - matched`,
leaving generated KV in the prefix → the next request continues from WRONG
context vs a fresh prefill. Enabled by default (NativeModelEngine.swift:32) so
it risks the 17/17 sweep.
Fix: the cache stored for a session/prefix MUST correspond EXACTLY to a token
prefix and contain ONLY that prefix's KV — no generated-token KV. Options:
(a) snapshot+trim the KV to the PROMPT length at the prompt/generation boundary
(before any decode step mutates it), or (b) store the running cache but record
the true KV length and trim to the matched-token boundary on reuse, proving the
retained KV equals a fresh prefill of those exact tokens. Whichever: the stored
KV length MUST equal the token count it is keyed by.

## MAJOR — store() overwrites a LEASED slot (lease invariant broken)
SessionPrefixKVStore.swift:156: `store` replaces `sessionSlots[sessionKey]`
without checking the old slot's `leaseCount`. A concurrent lease-holder's later
release() then no-ops (old slot unindexed). Fix: if the existing slot is leased,
do NOT replace it (skip the store, or keep the leased slot and drop the new one,
or defer) — never orphan a lease.

## MAJOR — MCP "SSE" is endpoint-discovery only, not a real transport
MCP.swift:692 reads the SSE stream only long enough to find an endpoint, then
MCP.swift:603 expects JSON-RPC results in each POST response. Real MCP SSE keeps
the GET event stream OPEN and delivers responses there (correlated by id).
Either implement the real streaming transport (persistent GET stream + response
correlation + overall connect deadline, not just per-request timeouts at
MCP.swift:654/698), OR if that's too large this round, DOWNGRADE the claim
honestly: don't call it SSE transport; document it as "HTTP+SSE-endpoint
handshake only" and file the streaming gap. No pretending.

## MAJOR — vacuous correctness tests → make them real
- PrefixSchedulerIntegrationTests.swift:53/71 asserts only stats. Replace with a
  REUSED-SESSION CORRECTNESS test: same session, request 1 (prompt P) then
  request 2 (prompt P + suffix) must produce output TOKEN-IDENTICAL to the same
  request 2 run with the cache DISABLED (fresh prefill). This is the real guard
  against the BLOCKER above.
- SessionPrefixKVStoreTests.swift:46 "overwrite" test never calls store while
  leased. Make it lease a slot, call store for that session, assert the leased
  slot is preserved and the lease-holder's release still works.

## Harness (worktree feat/omlx-parity-harness)
Add a cell that asserts reused-session output == cache-disabled output
(token-identical) on a small model. This must be RED on the current buggy code
and GREEN after the fix — include that red→green evidence in your report.

## After fixes — REGRESSION GATE (memory-guarded, one model at a time)
Re-run the msweep on 3-4 small models (0.6B, 1.7B, gemma-E2B) with the prefix
cache ENABLED (default) and confirm still PASS + concurrency probe still
token-exact. Report the results.

## Gate
swift test (all targets) green; harness correctness cell red→green shown;
regression sweep PASS; zero warnings; focused commits. Report SHAs + evidence.
