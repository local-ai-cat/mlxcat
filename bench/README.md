# mlxcat bench

A local benchmark suite you can run whenever you fancy, and a leaderboard that
is regenerated from the raw results. Nothing here runs in CI minutes; the
harness runs on our own Macs (and the in-app harness on our own iPhones), and
CI only checks that `LEADERBOARD.md` matches the committed JSONL.

```
bench/
  run.py          orchestrator — launches/attaches engines, measures, writes JSONL
  leaderboard.py  renders LEADERBOARD.md (also `--check`, the CI gate)
  engines.json    engine registry (launch templates / attach URLs / platforms)
  matrix.json     models, context tiers, concurrency widths, named model sets
  results/        committed JSONL, one record per cell (+ .logs/, ignored)
```

## Run it

```bash
swift build -c release --product mlxcat-http                 # once per checkout
python3 bench/run.py --dry-run                               # shows the plan
python3 bench/run.py                                         # mlxcat, default set, short/4k/16k, ×2/4/8
python3 bench/run.py --engines mlxcat,omlx,mlx-lm --model-set flagship --contexts short,4k,16k,32k
python3 bench/run.py --engine-url lmstudio=http://127.0.0.1:1234 --models Qwen3.5-4B-MLX-4bit
python3 bench/leaderboard.py                                 # re-render LEADERBOARD.md
```

Defaults: `--runs 3 --warmup 1` after one discarded cold request; `temperature 0`;
`max_tokens` per tier (128 short/4k, 64 at 16k/32k); models under
`~/Library/Caches/models/mlx-community` (`--model-root` / `MLXCAT_BENCH_MODEL_ROOT`).

### The quiet-machine guard

Throughput on an interactive Mac is load-sensitive — consecutive samples of one
configuration ranged 0.95×–1.31× on 2026-08-12, and a whole overnight matrix on
2026-08-20 was thrown away because the host sat at load 170–970. So `run.py`
refuses to run unless the 1-minute load average is ≤ `--max-load` (8), memory
free ≥ `--min-free-pct` (35 %) and `pmset -g therm` reports no CPU speed limit.
`--allow-loaded` runs anyway and stamps every record `valid_for_leaderboard:false`
(kept for audit, never ranked).

### What is measured, and how

Every engine is driven the same way: one streaming OpenAI `/v1/chat/completions`
request at a time (or N in parallel for the concurrency leg), greedy, the same
prompt built from a fixed filler paragraph repeated to a token target plus a
fixed question. Prompt length is calibrated per model from `usage.prompt_tokens`
of two probe requests so "16k" really is ~16k tokens on that tokenizer.

| metric | definition |
|---|---|
| `ttft_ms` | request start → first visible content/reasoning delta |
| `prefill_tps` | `usage.prompt_tokens ÷ ttft` |
| `decode_tps` | `(completion_tokens − 1) ÷ (end − first token)` |
| `e2e_tps` | `completion_tokens ÷ wall` — **the cross-engine comparison metric.** Unlike `decode_tps` it needs no assumption about streaming granularity, so it is the only decode-ish figure that works for an engine which coalesces tokens into few SSE chunks. oMLX sends ~8 tokens per chunk (measured 2026-08-22: 128 completion tokens, 122 ms inter-chunk, 63 tok/s e2e ⇒ ~16 chunks), so its `decode_tps` is correctly reported as unmeasurable; `PARITY.md` hit the same behaviour in July and reached the same conclusion. |
| `itl_p50_ms` / `itl_p95_ms` | inter-chunk gaps (chunks may carry >1 token — a jitter signal, not a token clock) |
| `aggregate_tps` | Σ completion tokens ÷ wall for N simultaneous streams. **Read this with the tier in mind**: prefill is per-row serial in mlxcat by design, so at a short tier a ~700 ms prefill dominates the wall of a 128-token generation and the number mostly reports *admission latency*, not batched decode. Measured 2026-08-21 (gpt-oss-20b, M4 Pro): TTFT 727/1614/3122/5883 ms at c1/2/4/8 — linear in N — while per-request decode fell only 4.3× across an 8× concurrency rise. The `longgen` tier is where batching can show its worth. |
| `peak_phys_footprint_bytes` | max `ri_phys_footprint` of the engine process sampled every 50 ms during the measured runs |
| `lifetime_max_phys_footprint_bytes` | `ri_lifetime_max_phys_footprint` after the cell (includes load; monotonic per process) |
| `cold_first_request_ms` | wall of the discarded first request (shape compilation / lazy load) |
| `server_tps` | whatever the engine itself reports in `usage`, if anything — advisory |

An engine that does not return `usage.completion_tokens` fails the cell: we do
not estimate tokens from characters.

Each launched engine gets **one process per model** so the footprint is per
model, not per session. Attached engines (`url`) are sampled by the pid found
listening on their port.

### Schema (`mlxcat-bench/1`, one JSON object per line)

```jsonc
{
  "schema": "mlxcat-bench/1",
  "run_id": "3f2a9c1e", "timestamp": "2026-08-21T16:10:00+00:00",
  "platform": "macos",                      // macos | ios | ipados
  "device": {"model": "Mac17,6", "chip": "Apple M5 Max", "memory_bytes": 137438953472, "os": "macOS 26.5", ...},
  "engine": {"name": "mlxcat", "version": "…", "transport": "http", "weights": "…", "notes": null, "pid": 123},
  "model": {"id": "Qwen3.5-4B-MLX-4bit", "offered_as": "…", "family": "qwen3_5", "params": "4B", "quant": "4bit", "role": "mac-floor"},
  "workload": {"context_tier": "4k", "prompt_tokens_target": 4096, "max_tokens": 128, "concurrency": 1, "temperature": 0, "runs": 3, "warmup": 1},
  "metrics": {"ttft_ms": {"median":…, "min":…, "max":…, "n":3, "spread_ratio":…}, "prefill_tps": {…}, "decode_tps": {…}, "e2e_tps": {…},
              "itl_p50_ms": {…}, "itl_p95_ms": {…}, "aggregate_tps": {…}?, "prompt_tokens": 4102, "completion_tokens": 128,
              "cold_first_request_ms": …, "peak_phys_footprint_bytes": …, "lifetime_max_phys_footprint_bytes": …},
  "host": {"loadavg_1m": 2.1, "loadavg_5m": 2.4, "memory_free_pct": 61, "thermal_cpu_speed_limit": 100},
  "valid_for_leaderboard": true, "invalid_reason": null,
  "harness": {"commit": "abc1234", "tag": "", "argv": ["--engines", "mlxcat"]}
}
```

`engine.transport` is `http` for everything this script produces. In-process
producers (the iOS app harness, `mlxcat-bench`) write `in-process` and are
rendered in the same tables — the column is there so a reader can tell.

### Platforms: macOS and iOS are both first-class

mlxcat ships **embedded in an iOS app** as well as on macOS, so the leaderboard is
stratified by platform and device first. What that means in practice:

* **macOS rows** come from `bench/run.py` (this script). Competitors are other
  macOS servers/engines (see `docs/ENGINES.md`).
* **iOS rows** come from an in-app harness on a real iPhone — `run.py` never
  produces them. The producer is Local AI Cat's device test suite
  (`MLXServeEmbeddedDeviceE2ETests` / `Gemma4MultimodalDeviceTests` already print
  `firstToken=` / `tok/s=` markers); emitting this JSONL is the next step, and the
  competitor rows on iPhone are MLX-Swift-in-app (the legacy `LLMEvaluator`
  path), llama.cpp, LiteRT-LM, Apple Foundation Models — the same set the
  neutral `apple-silicon-llm-bench` / `edge-llm-bench` project measures.
* Optimisations are per platform and must be **measured per platform** — a
  change that lifts M5 decode can cost an iPhone its memory budget. A row never
  carries over between platforms.

### Rules for a leaderboard row

1. Measured on a quiet machine (guard not tripped) — or it is not ranked.
2. Same weights for every engine in the comparison, or the row says `⚠︎` (Ollama
   library / GGUF are different artifacts).
3. The smoke model (`Qwen3-0.6B`) is rendered in its own section and is never a
   performance claim. Floor for a claim: `Qwen3.5-4B` (Mac), `gemma-4-E2B` (device).
4. The JSONL is the evidence; `LEADERBOARD.md` is derived. CI fails if they drift.
5. Re-running replaces: the newest valid record per cell wins.

### Other harnesses that still exist (and why)

* `mlxcat-bench` (Swift, in-process) — `swift run mlxcat-bench --model-dir …`:
  kernel-level prefill/decode with explicit GPU sync, no HTTP. Use it to
  separate engine cost from transport cost; its numbers are not comparable to
  the HTTP rows and are not on the leaderboard.
* `Benchmarks/http_parity_bench.py` — the original oMLX-parity probe
  (2026-07-03). Superseded by `run.py`; kept for the `PARITY.md` provenance.
* `scripts/fleet-correctness/` — correctness, not performance: the 16-model
  Mac campaign behind the "tool-call green" gate.
