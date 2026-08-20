# GOAL — parity-finish fix round 2 (governor review findings)

You are a delegated worker under an automated governor; the previous session
built `feat/parity-finish` (tip `3317430`, 227 tests green). The governor's
independent review found 1 BLOCKER + 4 MAJOR + 1 MINOR. Fix ALL of them on
`feat/parity-finish`, one focused commit per finding, then run the full
`swift test` and report. `NEEDS_INPUT: <question>` if a ruling below proves
unworkable — never guess. Do not push. Do not start new features.

## Findings + rulings

### 1. BLOCKER — MCP stdio client is unsafe under concurrent execute calls
`MCPStdioClient` keeps a single `activeRequestID`/`pendingRead`
(`Sources/MLXCatHTTP/MCP.swift:493,570`); actor reentrancy across
`await readMatchingResponse` lets a second request overwrite both →
leaked continuation / wrong timeout / lost response.
**Fix:** per-request state — a `[RequestID: PendingRequest]` map keyed by the
JSON-RPC id; the reader loop routes each reply to its pending entry; timeout
cancels only its own entry (and still kills/respawns the process, failing all
in-flight entries with the timeout error at that point, since the transport
died). **Test:** two concurrent execute calls against a fake server that
answers out of order → both complete correctly; one hung + one fast concurrent
→ fast completes, hung times out.

### 2. MAJOR — thinking_budget forced tokens bypass constrained-decode masks
The forced close/trailer token returns before allowedSequences / JSON / regex
masks apply (`Sources/MLXCat/TrackB/Sampling.swift:220`) and matchers are
advanced with tokens they may reject (`BatchGenerator.swift:297`).
**Ruling:** forced injection must VALIDATE against active matchers: if a
grammar/choice constraint is active and the forced token is not accepted by
it, skip injection for that step and sample masked as normal (the budget
keeps trying at subsequent steps; grammar always wins). Matchers must only
ever be advanced with tokens they accept. **Test:** budget + json grammar
active → output stays grammar-valid and matcher state never desyncs.

### 3. MAJOR — Harmony unknown channels leak into user-visible content
`parseHarmonyChannels` routes every non-analysis/final channel to `content`
(`Sources/MLXCatHTTP/ThinkingParser.swift:216`).
**Ruling:** match omlx exactly (check `/Users/timapple/Documents/Guest/omlx`
for its channel mapping). Where omlx has no defined mapping for a channel,
route it to `reasoning_content` (never to `content`) — unknown output must
not become visible assistant text. **Test:** commentary + an invented channel
name, stream and non-stream.

### 4. MAJOR — /v1/rerank bypasses pool lifecycle entirely
`NativeRerankBackend` loads/caches ModelContainers outside the memory
ceiling, queue, unload, and modelBusy accounting
(`Sources/MLXCatHTTPServer/NativeRerankBackend.swift:25`).
**Ruling (pragmatic, pre-pool-citizenship):** cap the rerank cache at ONE
loaded model — loading a different rerank model unloads the previous one
first; check the memory-guard ceiling before load and return 507 (same error
shape as the pool) when it would exceed; document in a comment that full
`audio_stt`-style pool citizenship for auxiliary model classes is a follow-on.
**Test:** second model load evicts the first; ceiling-exceeded → 507.

### 5. MAJOR — regex unsupported constructs silently reinterpreted
Anchors accepted as `.empty` anywhere (`RegexGrammar.swift:283`) so `a$b`
becomes `ab`; unknown escapes become literals (`RegexGrammar.swift:368`) so
`\D` means literal `D`.
**Ruling:** unsupported constructs MUST 400 with a clear message (match
omlx's error shape): reject `^`/`$` anywhere (constrained decode is
implicitly anchored — say so in the error), reject any escape not in the
supported set (`\d \w \s \. \\ \n \t \r` and literal escapes of
metacharacters). **Test:** each rejected construct → 400 + message; the
supported set still works.

### 6. MINOR — fork re-pin is unnecessary and carries non-QAT risk
`1fb53dd` makes Gemma4 K/V projections optional for ALL kv-shared-tail
configs, not just QAT; QAT already loaded on `098cf970` (sweep-verified,
both variants).
**Ruling: revert the `Package.swift` (and Package.resolved) pin to
`098cf970a96c26dca1fb5b036abbf198c0b74ad4`.** Keep the pushed fork branch
parked (do not delete). Keep the gated QAT load test if it passes on the
reverted pin (it should); if it needed 1fb53dd to pass, report that via
NEEDS_INPUT — that would mean the sweep and the test disagree and the
governor must arbitrate.

## Done =
All six fixed + tests per ruling + full `swift test` green (count must not
drop below 227 executed) + a per-finding summary with commit SHAs.
