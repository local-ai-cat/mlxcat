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
- Run conditions for a row to be promotable: phone on power, not hot,
  screen untouched after the test launches, no other apps foregrounded.
- Rows come out `valid_for_leaderboard: false` with the reason stamped —
  a phone has no loadavg guard, so the operator is the guard. Promote by
  flipping the flag on rows whose conditions held, append to
  `bench/results/`, re-render (`python3 bench/leaderboard.py && python3
  bench/timeline.py`). The leaderboard is stratified platform → device, so
  iPhone17,x and iPhone18,x rows land as their own sections — an iPhone is
  never compared against a Mac.

## Other engines on iOS (watchlist, not this runbook)

llama.cpp (GGUF — different artifacts, ⚠︎ class), Google LiteRT-LM (gemma
only), MLC-LLM: each needs its own host app to be benched honestly. Candidates
for the BenchHost pattern later; nothing else is headless-benchable today.
