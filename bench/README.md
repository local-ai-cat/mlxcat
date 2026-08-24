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

### Cold vs warm: measuring the prefix cache

Every request in **cold** mode carries a unique nonce prefix, so no prefix cache can
serve it. That is the fair cross-engine default — without it, mlxcat's tiered prefix KV
cache would answer a repeated benchmark prompt from cache while an engine configured
without one did real prefill, and the leaderboard would present the two as comparable.

But cold measures both engines at their worst: the cache costs (block extraction,
memory, publishing) are paid and none of the benefit is collected. **warm** mode repeats
the same prompt so the cache can hit, priming with the discarded cold request first, and
answers the question the feature exists for — *does turn 2 of a conversation get
cheaper?*

```bash
python3 bench/run.py --cache-modes cold,warm --models Qwen3.5-4B-MLX-4bit
```

Read warm against cold **for the same engine** to value its cache; read engines against
each other **within one cache mode**. `cache_mode` is part of the leaderboard key, so
the two never collide.

### Where the method came from

`METHODOLOGY.md` records what this harness took from oMLX's admin benchmark,
vLLM's serving benchmark and mlx-serve's release table — and, more usefully, what
it still owes them: an accuracy axis, a time series instead of a snapshot, and
realistic prompt distributions.

### Profiles: the cost of a run should be a decision

Measured per-cell medians on an M4 Pro (2026-08-22): short 27 s, 4k 46 s,
longgen 152 s, 16k 201 s. **Qwen3.8-27B alone is 76 of the default matrix's 154
minutes** — its 16k c1 (918 s) and longgen c8 (1224 s) are 36 minutes for two
cells. An iteration loop that includes it is not an iteration loop.

```bash
python3 bench/run.py --profile quick     # ~4 min/engine — enough to see a change
python3 bench/run.py --profile default   # ~2.6 h/engine — what a leaderboard row may come from
python3 bench/run.py --profile full      # overnight, on a machine nobody needs
```

Explicit flags still beat the profile, so `--profile quick --contexts 16k` means
what it says.

### The quiet-machine guard

Throughput on an interactive Mac is load-sensitive — consecutive samples of one
configuration ranged 0.95×–1.31× on 2026-08-12, and a whole overnight matrix on
2026-08-20 was thrown away because the host sat at load 170–970. So `run.py`
refuses to run unless the 1-minute load average is ≤ `--max-load` (8), memory
free ≥ `--min-free-pct` (35 %) and `pmset -g therm` reports no CPU speed limit.
`--allow-loaded` runs anyway and stamps every record `valid_for_leaderboard:false`
(kept for audit, never ranked). `--wait-for-quiet SECONDS` waits instead of
refusing — use it on a host that also runs CI. It replaces the shell poll loops
the campaign passes used to carry, which needed three consecutive quiet readings
and reset to zero on a single blip: on 2026-08-22 that spent 90 minutes on a
CI-busy box and produced no rows at all.

### Safeguards: the run must not be able to kill the machine

On **2026-08-22** vllm-mlx panicked the macOS GPU driver on the M4 Pro worker
(`completeMemory() prepare count underflow` @ IOGPUMemory.cpp:550) about nine
minutes after its first cell timed out. The host rebooted without auto-login and
was unreachable for twelve hours; the three passes queued behind it died with the
shell that was waiting on them, and 160 finished rows sat stranded on the
machine because the suite only synced at the end. None of that was measured
wrongly — it was simply lost. The guards below are what the harness learned, and
they are tested in `bench/test_run.py` rather than trusted.

| guard | flag | default | what it does |
|---|---|---|---|
| **runaway kill — footprint** | `--engine-memory-cap-pct` | 92 % of installed RAM | SIGKILLs an engine whose physical footprint crosses the line. The highest *honest* row we have measured is 90.8 % (mlx-serve, gemma-4-12B, c4 on 48 GiB), so this is a host-survival line, not a memory budget. |
| **runaway kill — swap** | `--swap-growth-kill-gb` | 8 GiB | SIGKILLs the engine once the host has paged out materially more than when the run began. Growth, not absolute: a laptop idles with 20 GiB swapped and the worker idles at zero. |
| **thrash invalidates** | `--swap-growth-invalid-gb` | 2 GiB | Well below the kill line — a swapping host produces junk numbers long before it threatens itself, so those rows are recorded and not ranked. |
| **failure budget** | `--engine-failure-budget` | 3 | Abandons an engine after that many consecutive failed cells, and moves to the next engine. A sick engine gets a short leash. |
| **quarantine** | `--allow-quarantined` | off | An engine that destabilised the *host* is marked in `engines.json` and refused by default. Quarantine is about safety, not score. |
| **resume** | `--resume` | off | Skips cells already recorded for this device **by the same engine binary**, so a host that dies mid-matrix costs the cells it had left. The build identity is not optional: without it, resume reused 220 cells measured by the binary from before the allocator fix, and the run that existed to re-measure them silently did not. |
| **checkpoint sync** | `--sync-after-engine CMD` | none | Runs `CMD` after each engine with `MLXCAT_BENCH_RESULT` set. An engine is the natural checkpoint; results that only exist on one machine are one panic from gone. |

The runaway guard watches an engine for its **whole life**, at 1 Hz, from launch
to shutdown — not only while a cell is being measured. `FootprintSampler` covers
just the measured requests, which is precisely the window vllm-mlx did *not* die
in: it died during launch and calibration. A breach kills the process, marks the
cell invalid with the reason, and abandons the engine rather than relaunching it
into the same wall.

Passes themselves are ordered on disk by `bench/queue.sh`, not chained in a
shell — see `bench/queue/README.md`. A pass that finishes is marked `.done` and
never re-run, a pass that fails **halts** the queue instead of letting dependent
passes measure the wrong binaries, and `queue.sh install` makes login resume the
queue so a reboot costs one pass rather than the campaign.

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
| `pp_tps` / `decode_agg_tps` | concurrency leg only: prompt tokens ÷ the wall until the **last** request's first token, and completion tokens ÷ the wall after it. Read these instead of `aggregate_tps` — see `METHODOLOGY.md`. |
| `ttft_mean_ms` / `ttft_p95_ms` / `tpot_mean_ms` / `tpot_p95_ms` | concurrency leg only: the tail, which is where serial admission actually hurts |
| `goodput_frac` | share of concurrent requests meeting `--sla-ttft-ms` and `--sla-tpot-ms`. Throughput bought by making some requests unusably slow is not throughput. |
| `aggregate_tps` | Σ completion tokens ÷ wall for N simultaneous streams. **Read this with the tier in mind**: prefill is per-row serial in mlxcat by design, so at a short tier a ~700 ms prefill dominates the wall of a 128-token generation and the number mostly reports *admission latency*, not batched decode. Measured 2026-08-21 (gpt-oss-20b, M4 Pro): TTFT 727/1614/3122/5883 ms at c1/2/4/8 — linear in N — while per-request decode fell only 4.3× across an 8× concurrency rise. The `longgen` tier is where batching can show its worth. |
| `peak_phys_footprint_bytes` | max `ri_phys_footprint` of the engine process sampled every 50 ms during the measured runs |
| `lifetime_max_phys_footprint_bytes` | `ri_lifetime_max_phys_footprint` after the cell (includes load; monotonic per process) |
| `cold_first_request_ms` | wall of the discarded first request (shape compilation / lazy load) |
| `server_tps` | whatever the engine itself reports in `usage`, if anything — advisory |
| `finish_reasons` | count per SSE `finish_reason` across the cell's runs (`stop` = EOS, `length` = token budget, `unreported`). At temp 0 on longgen some engines EOS near 520 tokens where others run all 1024 for the same model; a cell mixing the two compares workloads, not engines, and this field makes that visible per row (2026-08-24 concurrency-cliff analysis) |

**Request-body parity (2026-08-24):** the harness sends every engine the SAME body —
no hidden per-engine defaults. Until that date, engines without `extra_request_fields`
(only ours) silently received `{"enable_thinking": false}` while the rivals' explicit
`{}` meant model-default thinking: on hybrid-thinking qwen models our longgen rows
answered (~520 tokens, `stop`) while rivals thought to the full 1024 (`length`) — two
workloads in one cell. mlxcat longgen qwen rows from before this date are therefore
not comparable to rival rows in the same cells; re-measured rows supersede them.

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

`engine.build_id` is the short SHA-256 of the launched binary. mlxcat-http
reports no version on any endpoint we probe, so `engine.version` is null on every
row it produces and the hash is what makes a row attributable to a build — and
what stops `--resume` recognising a cell measured by a different one. (Making the
server report a real version is worth doing; the hash is what the leaderboard
needs either way, since it cannot be forgotten at release time.)

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
