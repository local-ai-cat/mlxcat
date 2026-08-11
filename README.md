# MLXServe

**Native-Swift LLM serving for Apple Silicon** — continuous/batched decode and a
tiered prefix KV cache (chain-hash blocks, hot RAM + cold SSD, restart-survivable),
built on [mlx-swift](https://github.com/ml-explore/mlx-swift). No Python runtime.

> Status: **pre-implementation.** This repo currently holds the implementation plan
> (`PLAN.md`) and reference material (`docs/reference/`). Code is built milestone by
> milestone against `swift test` invariant gates — see `PLAN.md`.

## Why

Local-first Swift apps that run MLX in-process today decode one sequence at a time and
recompute the full prompt every turn. MLXServe adds the two serving capabilities that
make on-device models practical for real agentic/coding work:

1. **Batched decode** — decode N concurrent sequences in one forward pass.
2. **Tiered prefix KV cache** — chain-hash the prompt into blocks, reuse the longest
   cached prefix across turns (hot RAM → cold SSD, survives restart). "Context stays
   cached mid-conversation" — the Claude-Code-stays-fast property.

It plugs into a host app behind a generation protocol (e.g. `LocalAITextGenerating`)
as a native engine, sharing the already-loaded weights — no second model process.

## Design

MLXServe is a native-Swift re-implementation following **mlx-lm's mechanism**
(the batched forward + ragged mask) and **oMLX's serving architecture** (scheduler,
block cache, hot/cold tiers). It does **not** use GPU-resident paged attention — like
oMLX, it reconstructs contiguous K/V from cached blocks and uses standard
`scaledDotProductAttention`, so there is no custom Metal kernel. See `PLAN.md` §0.

Scope v1: full-attention models only. Sliding-window/rotating caches, MTP/speculative
decode, VLM/OCR/embeddings, and GGUF/llama.cpp are explicitly out of scope (`PLAN.md` §1).

## Layout

```
PLAN.md                 the executable implementation plan (milestones + gates)
docs/reference/         mechanism extracts + pointers to local reference clones
Sources/MLXServe/       TrackB (batched decode + scheduler), TrackA (prefix + SSD cache)
Tests/MLXServeTests/    invariant gates (batch-invariance, cache-tier-invariance) + golden fixtures
```

## Building

Pass a local MLX model directory explicitly when running the executables:

```bash
MLXSERVE_MODEL_DIR=/path/to/mlx-model swift run mlxserve-http
MLXSERVE_MODEL_DIR=/path/to/mlx-model swift run mlxserve-bench
```

The package pins the `mlx-swift-lm` fork used by this branch at commit
`ac22e979e8223fde195e5930962bf6d0ed9f3081`.

## License

[Apache 2.0](LICENSE). A port of Apache-2.0 oMLX; see [NOTICE](NOTICE) for attributions.
Not affiliated with oMLX, mlx-lm, mlx-engine, or Apple's MLX.
