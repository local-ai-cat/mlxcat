# Lazy tool grammar results

Date: 2026-08-28

Branch: `feat/tool-grammar`

Base: `3bf9515`

Feature flag: `MLXCAT_TOOL_GRAMMAR=1` (unset by default)

## Outcome

The Qwen XML tool grammar is implemented and functionally works on the target model, but it does **not** pass the required mask-latency gate. The feature therefore remains off by default and is not performance-proven for sampling paths that need a full vocabulary mask.

The stop condition was real: worst measured mask p99 was **110.348 ms**, versus the required **< 10 ms**.

## What landed

- Engine-side Qwen XML tool grammar selection only when tools are present, `tool_choice` is `auto` or omitted, structured output is absent, and the environment flag is enabled.
- Token-ID lazy arming on `<tool_call>`, with the IDs resolved through the loaded tokenizer rather than hardcoded.
- Incremental `NORMAL -> TOOL -> POST_TOOL` runtime state.
- Exact requested-tool-name constraints.
- Free parameter text with premature `</function>` and `</tool_call>` rejection.
- POST_TOOL whitespace / next-call / EOS allowlist.
- A 4,096-token constrained-region back-out.
- Scheduler integration for initial sampling, continuous batches, preemption/resume, and speculative decoding.
- Speculation remains available in NORMAL, but a speculative batch that encounters the trigger is rejected and replayed through the ordinary one-token path before any body token is accepted.
- Focused matcher, state, preemption, speculation, malformed-payload, and real-vocabulary benchmark coverage.
- `AsyncGrammarMask.swift` is unchanged from the base commit, as required.

Model metadata was verified from the target model files:

- Configured vocabulary size: `151,936`
- `<tool_call>`: token ID `151657`
- `</tool_call>`: token ID `151658`
- EOS / `<|im_end|>`: token ID `151645`

## Gates

### Gate 1 — PASS: tests

Command: `swift test`

- Executed: `483`
- Skipped environment-gated tests: `57`
- Failures: `0`
- Test time: `6.040 s` (`6.073 s` wall reported by XCTest)

The focused coverage includes the original malformed payload with embedded quotes. The matcher cleanly refuses the legacy JSON body after the trigger, while the live smoke below proves that constrained generation can instead complete as parseable Qwen XML and emerge as an OpenAI `tool_calls` response.

### Gate 2 — PASS: pi temperature

An actual pi `before_provider_request` payload was captured using pi 0.74.0 with both `--provider mlxcat` and the explicit Qwen model.

- Payload `temperature`: omitted
- mlxcat effective temperature for an omitted field: `0.0`
- Payload `stream`: `true`

Therefore normal pi traffic uses mlxcat's greedy path. A valid argmax is checked directly; a full mask is only needed when argmax is invalid. This does not waive Gate 4.

### Gate 3 — PASS: release build and live smoke

Release build command: `swift build -c release --product mlxcat-http`

- Result: pass
- Reconciled build time: `20.10 s`

With `MLXCAT_TOOL_GRAMMAR=1`, five independent tool-call turns were sent to the release server on port 11702. Each prompt required a bash command containing embedded quotes and spaces.

| Turn | Finish reason | Tool | Arguments parse with `jq fromjson` | Completion tokens | Generation tok/s |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `tool_calls` | `bash` | yes | 29 | 50.518 |
| 2 | `tool_calls` | `bash` | yes | 31 | 88.290 |
| 3 | `tool_calls` | `bash` | yes | 34 | 88.064 |
| 4 | `tool_calls` | `bash` | yes | 29 | 88.616 |
| 5 | `tool_calls` | `bash` | yes | 30 | 88.296 |

Examples of the returned argument strings included:

- `{"command":"printf '%s\\n' \"hello world\""}`
- `{"command":"find . -name \"*.swift\" -maxdepth 2"}`
- `{"command":"sh -c 'echo \"quoted value with spaces\"'"}`
- `{"command":"python3 -c \"print('embedded quotes and spaces')\""}`

Flag-off isolation was compared against the preserved `3bf9515` binary using the same deterministic no-tools request:

- Both arms: content `ISOLATION`, finish reason `stop`, 19 prompt tokens, 4 completion tokens.
- Normalized deterministic response fields compared byte-for-byte equal.
- Preserved baseline generation: `98.061 tok/s`
- Current build, flag unset: `97.395 tok/s`
- Single-sample delta: `-0.679%`
- Pass rate: `1/1` in both arms

This check found no material divergence and did not trigger the flag-off stop condition. It is a focused isolation check, not a statistically powered throughput study.

### Gate 4 — FAIL: real-vocabulary mask latency

The release test bundle was prebuilt, load and resident processes were checked, then the benchmark was run directly with `xcrun xctest` so compilation did not contaminate the timed arm.

- 1-minute load average at timed-arm start: `3.94` (below the stop threshold of 8)
- LM Studio loaded models: none
- Resident mlxcat model servers: none
- Target configured vocabulary: `151,936`
- Usable decoded benchmark vocabulary: `150,199`
- Samples per prefix: `80`

| Prefix length | p50 ms | p99 ms | Required p99 | Outcome |
| ---: | ---: | ---: | ---: | --- |
| 0 | 8.308 | 8.758 | < 10 | pass |
| 64 | 107.049 | 109.579 | < 10 | **fail** |
| 256 | 107.202 | 110.058 | < 10 | **fail** |
| 1024 | 107.259 | 110.348 | < 10 | **fail** |

Benchmark test result: pass as a measurement harness, `27.427 s`. Performance gate result: **fail**.

The worst p99 is about `11.0x` the allowed ceiling. The synchronous fallback is therefore unsafe in active parameter-body states. Per the brief, work stopped here rather than hiding or explaining away the failed gate.

## Stop conditions

- **Hit:** mask p99 >= 10 ms (`110.348 ms` worst case).
- Not hit: load average > 8 at the start of a timed arm.
- Not hit: two 17 GiB model copies resident during a timed arm. `lms unload --all` reported no loaded LM Studio model, and mlxcat arms ran sequentially.
- Not hit in the focused check: flag-off response/pass-rate/tokens-per-second divergence.

## PROVEN by measurement

- The reconciled Swift suite passes: 483 executed, 0 failures.
- The release `mlxcat-http` product builds.
- The target tokenizer resolves the expected trigger and delimiter IDs.
- Pi omits temperature, which mlxcat resolves to greedy temperature 0.
- Five live flagged-on turns produced `tool_calls`; all five `function.arguments` strings parsed as JSON with `jq`, including embedded quotes and spaces.
- The focused flag-off output matched the preserved baseline on deterministic fields, with a `-0.679%` single-sample generation-rate delta.
- The full-mask implementation fails the required active-body latency ceiling: p99 roughly 109.6–110.3 ms at nonzero prefixes.

## BUILT, but not proven shippable

- Lazy Qwen XML constraint arming, post-tool release, constrained-region budget, preemption recovery, and trigger-safe speculation are implemented and covered by synthetic regressions.
- The feature is **not** performance-safe when sampling requires a full active-body mask.
- The flag-off isolation result is focused rather than statistically exhaustive.
- No claim is made for out-of-scope tool dialects, required/named tool choice, mixed-batch preservation, or streaming tool-call deltas.

Bottom line: **functionally implemented, live behavior demonstrated, performance gate failed, feature remains off by default.**
