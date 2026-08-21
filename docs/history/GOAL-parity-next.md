# GOAL — parity next wave (post-merge lanes)

Same rules as docs/GOAL-parity-finish.md (worker under governor, NEEDS_INPUT on
ambiguity, never push, tests-are-the-deliverable, no pauses between lanes —
one review at the end). Branch `feat/parity-next` off the CURRENT
`feat/omlx-parity` tip (the governor has merged parity-finish and the M8b
speech registry into it — your baseline test count is whatever `swift test`
reports at start; it only goes up).

## Lane 1 — M7b-b: GBNF/CFG grammar sampler (approved)
`structured_outputs.type == "grammar"` currently 400s. Port a llama.cpp-style
GBNF grammar sampler in pure Swift:
- Parse GBNF text (llama.cpp grammar format — root rule, alternatives,
  char classes, repetition) into rule tables; reject malformed grammars 400.
- Incremental matcher over generated text with the same shape as
  `JSONGrammarMatcher`/`RegexGrammarMatcher` (Grammar/ directory): shared
  vocabulary index, first-char bucket pruning WITH the incomplete-vs-invalid
  distinction (see the JSON matcher's escape handling — prefix-validity must
  be tri-state), greedy-only rejection fast path, EOS gated on accept.
- Wire through StructuredOutputParser + sampler exactly like regex.
- Tests: mirror JSONGrammarMatcherTests (matcher unit + 2 GPU sampler tests)
  + parse-error 400 tests + one end-to-end constrained decode with a small
  arithmetic-expression grammar.

## Lane 2 — M8b-2: speech pool citizenship (audio_stt)
The speech registry (Sources/MLXCatSpeech, WhisperKit adapter, registry
bridge in MLXCatHTTPServer) is merged. Make speech models pool citizens:
- `GET /v1/models` and `models/status` include speech models with a
  `model_type: "audio_stt"` field (match omlx's status taxonomy —
  reference /Users/timapple/Documents/Guest/omlx engine_pool.py).
- `models/<id>/load` and `/unload` route to the owning adapter's
  loadModel/unloadModel for speech model ids (registry resolves ownership;
  unknown id keeps the existing 404 + list shape).
- Footprint: report the adapter's loadedFootprint in status; document that
  CoreML working-set accounting is approximate.
- Do NOT wire speech into the LLM EnginePool's LRU/ceiling eviction — that
  is a follow-on; keep it read/route-through and say so in a comment.
- Tests: status shape incl. audio_stt entries; load/unload round-trip with a
  fake adapter; 404 unknown.

## Lane 3 — small-gaps batch (omlx parity tail)
1. **Reasoning stream event names**: diff our SSE reasoning event naming vs
   omlx's and match exactly (harness semantic axis is the oracle; check
   omlx's api/ formatters).
2. **count_tokens exactness**: `/v1/messages/count_tokens` currently
   estimates; count via the model's chat template application (tokenizer is
   available through the pool) and return exact counts. Keep the estimate
   fallback only when no model is loadable, and mark it in the response the
   way omlx does (check reference).
3. Tests for both.

## Done =
All 3 lanes + full `swift test` green + per-lane summary with commit SHAs +
whole-diff self-review pass (fix what you find, note it).
