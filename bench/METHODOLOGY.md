# What we learned from other benchmarks, and what we still owe

Read on 2026-08-22 against the reference clones in `guest/` and the public
projects in `docs/ENGINES.md`. This is the list of things other people got right
that we had wrong or missing, what has been adopted, and what has not.

## Adopted

### The concurrency leg was measuring the wrong thing (oMLX, vLLM)

`omlx/admin/benchmark.py:453-468` does not report tokens ÷ wall for a concurrent
burst. It reports:

* `pp_tps` — total prompt tokens ÷ (**last** request's first token − start)
* decode measured from that same instant, because "generation starts when the
  last request finishes prefill"
* `avg_ttft_ms` — the mean, not one request's

Ours was `Σ completion_tokens ÷ wall`, which with serial prefill mostly reports
admission latency. That is precisely the metric that produced a wrong "+22% from
batching" conclusion on this repo. `concurrency_metrics()` now splits the prefill
wall from the decode wall; `bench/test_run.py` proves the two arrangements are
indistinguishable on wall clock and separate cleanly under the decomposition.

### Closed-loop bursts are not how a server is used (vLLM)

vLLM's serving benchmark drives an **open** loop: requests arrive at a rate with
a *burstiness* factor shaping the gaps (gamma; 1.0 Poisson, <1 burstier). Firing
the whole width at one instant is the worst case for an engine that admits
serially — a real number, but only one of them. `--request-rate` / `--burstiness`
now do the open loop, and `workload.arrival` records which was measured.

### Tails and goodput (vLLM)

A median hides the tail, and the tail is exactly where serial admission hurts.
Added `ttft_mean_ms`, `ttft_p95_ms`, `tpot_mean_ms`, `tpot_p95_ms`, and
`goodput_frac` — the share of concurrent requests meeting an SLA
(`--sla-ttft-ms`, `--sla-tpot-ms`). Throughput bought by making some requests
unusably slow should not read as throughput.

### The ladder stopped at the easy half (oMLX)

oMLX offers 1k/4k/8k/16k/32k/64k/128k/200k. We stopped at 16k, which is where
memory strategy and prefix caching have barely started to matter. The ladder now
reaches 128k; 32k and up are opt-in via `--contexts`, and the runaway guard is
what makes running them on a 48 GiB box survivable.

### A debug build reads exactly like losing (mlx-serve)

`guest/mlx-serve/tests/bench.sh`: "Debug is 2-4x slower = a fake regression."
`reject_debug_build()` now refuses a `/debug/` binary rather than publishing a 3×
handicap on our own engine as a result.

## Not adopted yet — in rough order of what they would change

### 1. An accuracy axis (oMLX)

oMLX runs sixteen evals — MMLU, MMLU-Pro, KMMLU, CMMLU, JMMLU, HellaSwag,
TruthfulQA, ARC-C, WinoGrande, GSM8K, MathQA, HumanEval, MBPP, LiveCodeBench,
BBQ, SafetyBench (`omlx/eval/`, ~2.5k lines) — with the same SSE progress UI as
the perf benchmark. We have **none**. A serving engine can be fast and wrong, and
quantisation, batching and cache-reconstruction bugs all surface as accuracy
drift long before they surface as a crash.

The version of this that is ours specifically: **run one eval at c1 and again at
c8 and assert the score does not move.** We have logit-level batch-invariance
gates, but nothing that says batched decode produces the same *answers* — which
is the actual claim `usesSerializedDecode`'s exclusion lists are making.

### 2. A time series, not a snapshot (mlx-serve)

`guest/mlx-serve/benchmarks.md` is decode tok/s **per release**, and it is how
they can say "+188% on Qwen3.6 27B" with a straight face. Our leaderboard is a
snapshot: newest-wins per cell, no history, so we cannot tell whether a change
helped. Every row already carries `harness.commit` and `engine.version`, so the
data is there — the renderer is not.

Their rule about the harness itself is worth stealing verbatim: a speedup that
spans a harness change "measures the harness as much as the engine."

### 3. Speculative decoding is a missing engine feature, not a missing metric

mlx-serve's table names the speculation mode beside every number — `drafter`,
`mtp`, `pld` — and their Qwen3.6-27B went 24 → 76 tok/s (+188%) largely on MTP
speculative decoding. We do not have it. A chunk of the decode gap we are trying
to close by tuning may simply be this feature.

### 4. Realistic prompt distributions (vLLM)

vLLM samples ShareGPT for real length distributions and can replay a trace. We
build one filler paragraph repeated to a token target, which is a clean
controlled input and an unrealistic one — real traffic is a mixture, and a
mixture is what makes a scheduler's queueing behaviour visible.

### 5. Community submission (oMLX, apple-silicon-llm-bench)

* oMLX uploads to `https://omlx.ai/api/benchmarks` with an `owner_hash` =
  SHA-256(IOPlatformUUID + chip + GPU cores + memory GB) plus a check character —
  a stable pseudonymous device identity with no PII. If we ever take outside
  rows, that is the pattern to copy.
* [apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench)
  takes **one PR per row** (runtime × model × device) and covers iPhone and iPad
  as well as Mac. Our schema was already shaped to mirror it, so cross-submitting
  our iOS rows is mostly a format adapter — and iOS is the platform where we have
  no rows at all and the most to prove.

### 6. Engine-reported prefill duration (oMLX)

We derive `prefill_tps` as `prompt_tokens ÷ ttft`, which folds queueing and
sampling into "prefill". oMLX prefers the engine's own reported `prompt_tps` when
it has one (`benchmark.py:340-351`) and only falls back to TTFT. It also asserts
the queue was empty before a measured request and warns when it was not — a
contamination check we do not have.
