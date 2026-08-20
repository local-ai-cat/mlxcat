# MLXCat Native vs oMLX Parity

Date: 2026-07-03

Model: `Qwen3-0.6B-4bit`

## Methodology

Native in-process benchmark:

```bash
MLXSERVE_TEST_MODEL=/Users/timapple/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit \
  /tmp/mlxcat-m6-bench/mlxcat-bench \
  --runs 5 --warmup 2 --decode-tokens 16 \
  --output /tmp/mlxcat-native-bench-corrected.md
```

Native HTTP server:

```bash
MLXSERVE_TEST_MODEL=/Users/timapple/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit \
  /tmp/mlxcat-http-m6/mlxcat-http \
  --port 18181 --model-id Qwen3-0.6B-4bit --max-concurrent-requests 8
```

oMLX HTTP server:

```bash
/tmp/omlx-sidecar-venv-20260703/bin/omlx serve \
  --model-dir /Users/timapple/Library/Caches/models/mlx-community \
  --port 18080 --base-path /tmp/omlx-m6-http-base \
  --no-cache --log-level warning
```

Shared HTTP client:

```bash
python3 Benchmarks/http_parity_bench.py \
  --native-url http://127.0.0.1:18181 \
  --omlx-url http://127.0.0.1:18080 \
  --model Qwen3-0.6B-4bit \
  --runs 5 --warmup 2 --decode-tokens 16
```

Prompt set: same `/v1/chat/completions` request body and same client for both HTTP servers. The measured long prompt repeated the fixed prefix sentence 26 times and produced 529 prompt tokens over HTTP on both native and oMLX.

## Timing Audit

The native in-process prefill benchmark was audited first. The prefill path already called `eval(output.logits, cache)` and `Stream.gpu.synchronize()` before stopping the timer, so it was not merely timing MLX kernel dispatch. The benchmark now also materializes one output scalar after synchronization to make GPU completion explicit.

Two benchmark issues were fixed:

- Decode TG previously included prefill work in the timed region while dividing only by generated tokens. It now prepares/prefills outside the timed decode loop.
- Prompt token extraction through `model.prepare(..., windowSize:)` could observe only the final prepared chunk. The in-process benchmark now tokenizes directly for the long prefill prompt.

Corrected native in-process result:

| Metric | Value |
| --- | ---: |
| Prompt tokens | 521 |
| Prefill PP | 25026.42 tok/s |
| Decode TG | 374.36 tok/s |
| TTFT | 6.52 ms |

Conclusion: the earlier ~23k tok/s native prefill number was not a dispatch-only measurement bug; with a 521-token prompt and explicit GPU completion it still measured ~25.0k tok/s. The decode metric was mislabeled before this fix.

## A. Engine Parity

This section compares native-over-HTTP vs oMLX-over-HTTP using the same client and request shape. This is the honest same-transport comparison.

| Server | Server PP | Server TG | HTTP TTFT | HTTP output | Prompt Tokens | Content Chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native-http | 18317.84 tok/s | 344.62 tok/s | 11.06 ms | 207.27 tok/s | 529 | 16 |
| omlx-http | 5188.22 tok/s | n/a | 78.99 ms | 152.67 tok/s | 529 | 1 |

oMLX emitted one aggregated content chunk for the 16-token completion and reported `generation_duration: 0.0`, so `/v1/chat/completions` did not expose a valid per-token TG split for oMLX in this run. The old 55k+ oMLX TG derived from `wall - TTFT` was rejected as invalid. The comparable HTTP-observed output rate is completion tokens divided by full request wall time.

Same-transport deltas:

| Metric | Delta |
| --- | ---: |
| Server-reported PP | native 3.53x higher |
| Server TG | not comparable; oMLX `/v1` TG unavailable |
| HTTP TTFT | native 86.00% lower |
| HTTP output throughput | native 1.36x higher |

Concurrency sweep, HTTP-observed completion throughput:

| Concurrency | Native HTTP | oMLX HTTP | Native vs oMLX |
| ---: | ---: | ---: | ---: |
| 1 | 217.55 tok/s | 145.01 tok/s | 1.50x |
| 2 | 284.62 tok/s | 122.46 tok/s | 2.32x |
| 4 | 332.86 tok/s | 127.44 tok/s | 2.61x |
| 8 | 365.26 tok/s | 129.80 tok/s | 2.81x |

## B. Deployment

Native in-process avoids HTTP parsing, socket IO, SSE framing, and server-side chunking behavior. That deployment advantage is real for embedding MLXCat directly, but it is a transport/deployment advantage, not proof that the underlying MLX kernels are faster.

| Path | PP | TG | Notes |
| --- | ---: | ---: | --- |
| Native in-process | 25026.42 tok/s | 374.36 tok/s | Direct Swift call, explicit GPU sync |
| Native HTTP | 18317.84 tok/s | 344.62 tok/s | NWListener `/v1/chat/completions`, token SSE |
| oMLX HTTP | 5188.22 tok/s | n/a | Same client, aggregated content chunk |

## Verdict

Functional parity remains green through M5, and M6 now includes a minimal native `/v1/chat/completions` and `/v1/models` HTTP surface.

Engine parity over the same HTTP transport is not a clean TG parity proof because oMLX does not expose per-token streaming timing for this request through `/v1/chat/completions`. On the comparable same-transport metrics that were observable, native matched prompt-token counts, reported 3.53x higher PP, had 86.00% lower HTTP TTFT, and delivered 1.36x higher end-to-end HTTP output throughput for the long-prompt request. The result is favorable to native on this run, but the verdict should not claim an engine TG win from the invalid oMLX aggregated-stream TG value.
