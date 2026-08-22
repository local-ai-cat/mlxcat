# MLXCat

**Native-Swift LLM serving for Apple Silicon — on the Mac and inside an iPhone app.**
Continuous/batched decode, a tiered prefix KV cache (chain-hash blocks, hot RAM +
cold SSD, restart-survivable), OpenAI / Anthropic / Responses dialects, grammar-
constrained output, tool calling, MCP, speech, rerank and embeddings — all on
[mlx-swift](https://github.com/ml-explore/mlx-swift). No Python runtime, nothing
to install beside the binary or the Swift package.

[![ci](https://github.com/local-ai-cat/mlxcat/actions/workflows/ci.yml/badge.svg)](https://github.com/local-ai-cat/mlxcat/actions/workflows/ci.yml)
[![leaderboard](https://img.shields.io/badge/leaderboard-LEADERBOARD.md-blue)](LEADERBOARD.md)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> mlxcat is the serving engine inside [Local AI Cat](https://github.com/local-ai-cat)
> (macOS + iOS) and runs standalone as `mlxcat-http`. It was ported from
> [oMLX](https://github.com/jundot/omlx)'s serving architecture and
> [mlx-lm](https://github.com/ml-explore/mlx-lm)'s batched-decode mechanism, then
> kept honest against both with differential tests. Not affiliated with either,
> or with Apple.

## What you get

| | |
|---|---|
| **Continuous batching** | N requests decode in one forward pass; rows join and leave mid-flight (Track B: `BatchGenerator`, `BatchCache`, `Scheduler`) |
| **Prefix KV cache** | chain-hashed 256-token blocks, hot RAM + cold SSD tiers, CoW sharing, survives restart (Track A) — "context stays cached mid-conversation" |
| **Models** | everything mlx-swift-lm loads: Qwen3 / 3.5 / 3.6 / 3.8 (incl. hybrid caches), Gemma 4 (sliding-window), Llama 3, DeepSeek-R1 distills, gpt-oss, MoE (Qwen3-Coder-30B-A3B), VLMs via MLXVLM |
| **APIs** | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (Anthropic), `/v1/models`, `/v1/embeddings`, `/v1/rerank`, `/v1/audio/transcriptions`, `/v1/mcp/*`, health/metrics |
| **Generation control** | greedy/temperature/top-p/min-p/XTC, per-request seeded RNG, logit bias, min tokens, stop sequences, reasoning channels (`<think>`, Harmony), thinking budget |
| **Structured output** | JSON-schema, regex and GBNF grammar-constrained decode with masks precomputed off the decode path |
| **Tool calling** | per-model parser registry (Qwen, Llama 3, Gemma, Hermes, `<function_call>`, Harmony), streaming tool deltas |
| **Ops** | memory watchdog + ceiling, admission by KV estimate, preemption/resume, chunked long-prompt admission, ngram speculative decode (opt-in), idle unload |
| **Speech** | `MLXCatSpeech` registry with a WhisperKit backend |

Scope fence (v1): no GPU-resident paged attention (contiguous K/V is reconstructed
from cached blocks, like oMLX — see `PLAN.md` §0), no MTP, no GGUF/llama.cpp.

## Quick start

```bash
git clone https://github.com/local-ai-cat/mlxcat && cd mlxcat
swift build -c release --product mlxcat-http
.build/release/mlxcat-http --model-dir ~/Library/Caches/models/mlx-community --port 11500

curl -s http://127.0.0.1:11500/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen3.5-4B-MLX-4bit", "stream": false,
  "messages": [{"role":"user","content":"Say hi in five words."}]
}'
```

Models are plain `mlx-community` directories (safetensors + `config.json` +
tokenizer). Nothing is downloaded by the server; point `--model-dir` at what you
already have. Useful flags: `--memory-ceiling-bytes`, `--max-concurrent-requests`,
`--pin <model>`, `--idle-timeout`, `--mcp-config`, `--embedding-model-dir`,
`--whisperkit-models-dir`.

### As a Swift package

```swift
.package(url: "https://github.com/local-ai-cat/mlxcat.git", revision: "<sha>")
// products: MLXCat (engine), MLXCatNative (model engine + pool), MLXCatHTTP (wire types),
//           MLXCatSpeech, MLXCatSpeechWhisperKit; executables mlxcat-http, mlxcat-bench, mlxcat-baseline
```

The host app embeds `MLXCatNative` directly (same process, same loaded weights —
no sidecar) on macOS **and iOS**. Pin by revision; the API is pre-1.0.

## We read the competition, on purpose

mlxcat exists because Local AI Cat needs one thing none of the engines below can
give it: **all-Swift serving that runs unchanged on macOS and inside an iPhone
app** — no Python runtime, no sidecar process, no separate iOS story. That
constraint is the reason to write an engine. It is not a reason to invent
mechanisms that already exist and work.

So the reference engines are checked out under `guest/` and **read**, and where
one of them does something better we take the mechanism and say where it came
from. Every such place carries a `guest/<repo>/<path>:<line>` citation in the
comment, so the claim is checkable and so the next person can go read the
original rather than re-deriving it:

| watched | what it is | why we read it |
|---|---|---|
| [`mlx-lm`](https://github.com/ml-explore/mlx-lm) | Apple's Python reference | the semantics of prefill, sampling and cache handling — the closest thing to a spec |
| [`omlx`](https://github.com/jundot/omlx) | Python serving stack | serving architecture, engine pooling, MLX stream isolation |
| `mlx-serve` | Zig server | the fastest single-stream numbers on our board |
| [`vllm`](https://github.com/vllm-project/vllm) / `vllm-mlx` | the batching literature | continuous batching, chunked prefill, admission budgets |
| [`ollama`](https://github.com/ollama/ollama) | the UX bar | model management and API surface expectations |
| [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) | the Swift model layer | what we build on; also the floor we must not fall under |

Two things keep this honest rather than aspirational:

- **`bench/parity.py --check`** scores every mlxcat cell against the best OTHER
  engine measured on the same device, model, context and concurrency, and fails
  when we lose ground. It is the only gate here that can go red because a
  competitor is better — every other one compares mlxcat to its own past, which
  is how a suite stays green while losing.
- **`docs/PARITY-LEDGER.md`** records mechanism by mechanism what each reference
  does, what we do, and whether that is a gap or a deliberate difference — with
  the citations verified in CI so they cannot rot as the clones move.

Not affiliated with any of these projects. Apache-2.0 where we have taken an
approach; see `NOTICE`.

## Performance — the leaderboard

[`LEADERBOARD.md`](LEADERBOARD.md) is generated from raw JSONL in
`bench/results/` by a same-transport harness you can run yourself whenever you
like, against whatever engines you have installed:

```bash
python3 bench/run.py --engines mlxcat,mlx-swift-lm-tokeniterator,omlx --model-set default
python3 bench/leaderboard.py
```

It measures TTFT, prefill and decode tok/s, aggregate throughput under
concurrency, and the engine process's peak physical footprint, stratified by
**platform → device → model → context**. `mlx-swift-lm-tokeniterator` is the raw
mlx-swift-examples loop (what Local AI Cat's legacy engine was) — the floor
mlxcat must not fall under. Rules, schema and the quiet-machine guard:
[`bench/README.md`](bench/README.md). Who is on the board and who is watched:
[`docs/ENGINES.md`](docs/ENGINES.md).

## Tests

```bash
swift test                                      # unit + fixture suites; model gates skip
MLXSERVE_TEST_MODEL=~/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit swift test
scripts/nightly-models.sh                       # every model-gated suite against local models
```

Invariant gates: batch-invariance (batched logits == serial within dtype
tolerance), cache-tier-invariance (prefix hit == fresh prefill), restart survival
(byte-exact safetensors round-trip), serial-greedy scheduler parity, 16k memory
budget. CI runs the no-model suite on GitHub-hosted Apple Silicon; model-gated
suites and benchmarks run on our own Macs and land as commits.

## Layout

```
Sources/MLXCat/                 engine: TrackB (batched decode + scheduler), TrackA (prefix + SSD cache), Seam, Engine
Sources/MLXCatNative/           NativeModelEngine, EnginePool, discovery, dialects, parsers, grammar
Sources/MLXCatHTTP/             wire types shared by server + embedders
Sources/MLXCatHTTPServer/       mlxcat-http (NWListener server, MCP, speech routes)
Sources/MLXCatSpeech*/          speech registry + WhisperKit backend
Sources/MLXCatBench/            in-process kernel-level bench (mlxcat-bench)
Sources/MLXCatBaseline/         in-process TokenIterator + engine baseline rows (mlxcat-baseline)
Tests/MLXCatTests/              ~60 suites; Fixtures/ golden JSON; Support/ resolvers + tolerances
bench/                          harness, engine registry, matrix, results → LEADERBOARD.md
scripts/                        donor-drift.sh, nightly-models.sh, fleet-correctness/
docs/                           ENGINES.md, MCP-TRANSPORTS.md, M8B-SPEECH-DESIGN.md, reference/, history/
PLAN.md · PARITY.md             the original executable plan and the 2026-07-03 parity measurement
guest/                          (gitignored) reference + competitor engine clones
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — both platforms, invariant gates, quiet
machines, conventional commits. Security reports: [SECURITY.md](SECURITY.md).
Changes: [CHANGELOG.md](CHANGELOG.md).

## License

[Apache 2.0](LICENSE). A port of Apache-2.0 oMLX; see [NOTICE](NOTICE) for
attributions (oMLX, vllm-mlx, mlx-lm, mlx-engine).
