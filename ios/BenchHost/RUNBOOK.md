# BenchHost runbook — iOS leaderboard rows on real iPhones

The plan for the first device night (iPhone 16 + iPhone 17), per the 2026-08-24
session with Phil.

## The roster (what runs on a phone, and why)

From `bench/matrix.json` (`ios: true` is the curated flag):

| model | role | why it's on the phone list |
|---|---|---|
| gemma-4-E2B-it-qat-4bit | device floor | **the headline row** — the model the app actually ships to phones |
| Qwen3-1.7B-4bit | small | the low end a phone user realistically picks |
| Qwen3.5-4B-MLX-4bit | mac-floor | the ceiling an 8 GB iPhone fits under the ~6 GB app memory cap |
| Qwen3-0.6B-4bit | smoke | wire-format smoke ONLY — never a perf number (2026-08-19 ruling) |
| Llama-3.2-3B-Instruct-4bit / gemma-4-E4B | small | second-night additions if the first three go clean |

Every model runs BOTH arms via `MLXCatBaselineKit`: **mlxcat-inprocess** and
**mlx-swift-lm-tokeniterator** — the raw `TokenIterator` loop that is exactly the
legacy `LLMEvaluator` mechanism, so "new engine vs what the app used to do" is
answered per device.

Cells: short (256 prompt) and 4k. 16k on a phone is a memory campaign of its
own — added deliberately later, never by default.

## Per device

```bash
xcrun xctrace list devices                     # find the udid; phone must be unlocked + trusted
ios/BenchHost/run-device-bench.sh --device <udid>
```

- First run per model downloads the weights into BenchHost's container
  (plugged in; the E2B is ~3 GB). Later runs reuse them.
- Run conditions are MEASURED, not attested (2026-08-30): the harness
  samples thermal / power / lock / foreground / Low Power Mode every 2 s
  and stamps each row's `valid_for_leaderboard` from its own cell's
  measurements, evidence in the row's `host` object. The gate mirrors the
  Mac methodology: pre-run state gates, in-run state is evidence. A cell
  must START at ≤ fair (a bounded 10-min cool-down before each
  measurement makes that the normal case — the device wait-for-quiet),
  and power / unlocked / foregrounded / no-LPM must hold throughout;
  `critical` anywhere invalidates. Worst thermal state and the fraction
  of samples at ≥ serious ride in `host`, because a phone under
  sustained inference re-enters `serious` within a minute of a cooled
  start (measured 2026-08-30) — an all-samples gate would just mean no
  iPhone row can exist. The only condition the platform cannot see is a
  finger on the screen that never locks or backgrounds the app.
- Valid rows append straight into `bench/results/`; re-render with
  `python3 bench/leaderboard.py && python3 bench/timeline.py`. The
  leaderboard is stratified platform → device, so iPhone17,x and
  iPhone18,x rows land as their own sections — an iPhone is never
  compared against a Mac.

## Other engines on iOS (watchlist, not this runbook)

llama.cpp (GGUF — different artifacts, ⚠︎ class), Google LiteRT-LM (gemma
only), MLC-LLM: each needs its own host app to be benched honestly. Candidates
for the BenchHost pattern later; nothing else is headless-benchable today.
