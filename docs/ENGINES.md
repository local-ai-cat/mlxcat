---
living:
  cadence: weekly
  sources:
    - scripts/donor-drift.sh
    - docs/watch/cursors.json
  checker: docs/watch/CHECKER.md
  edit: open
  owner: mlxcat
  last_verified: 2026-10-02
---

# Engines: what mlxcat is benchmarked against, and what we watch

Two lists. **Benchmarked** engines have (or are slated for) rows on
`LEADERBOARD.md` via `bench/run.py`. **Watchlist** engines are tracked by
`scripts/donor-drift.sh` (pins, releases, commit subjects since our pin) because
we port ideas from them, depend on them, or expect to compare against them.

Philosophy (ruled 2026-08-19): **all-in on MLX, port to MLX when needed** — no
foreign engine is ever bundled. Python projects are references, never runtime
dependencies. Everything below is a reference or a competitor, not a dependency,
except the SwiftPM pins in `Package.swift`.

## Platform matters

mlxcat ships twice: as a **macOS server/engine** and **embedded in an iOS app**.
A competitor that only exists on macOS cannot have an iPhone row, and iPhone
numbers come from in-app harnesses (see `bench/README.md` § Platforms). The
`platforms` column below is the honest comparison set per platform.

## Benchmarked (macOS rows via `bench/run.py`)

| engine | what it is | lang | weights | platforms | harness status |
|---|---|---|---|---|---|
| **mlxcat** | this repo — native Swift continuous batching + tiered prefix KV cache on mlx-swift | Swift | mlx-community | macOS, iOS | launcher (`mlxcat-http`) |
| **mlxcat-defaults** | the same binary with **no memory ceiling** — the fair-fight arm. Competitors all run at MLX's defaults (~54 GiB cache limit on a 48 GiB Mac); `mlxcat` runs at 24 GiB with a 4 GiB cache slice, which is a product choice, not a comparison baseline. Read the pair as an A/B of what the guard costs. | Swift | mlx-community | macOS, iOS | launcher (`mlxcat-http`) |
| [oMLX](https://github.com/jundot/omlx) | the Python reference mlxcat was ported from; continuous batching, paged SSD KV cache, menu-bar server | Python (mlx-lm) | mlx-community | macOS | launcher (`omlx serve`) |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) `server` | Apple's reference server; single-sequence | Python | mlx-community | macOS | launcher (`python -m mlx_lm server`) |
| [vllm-mlx](https://github.com/waybarrios/vllm-mlx) | continuous batching, paged + prefix KV, SSD tier, OpenAI + Anthropic APIs, MCP | Python | mlx-community | macOS | launcher (verify CLI) |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | ddalcu's Zig core + mlx-c — the other non-Python native MLX server, and our closest architectural rival; also the source of the `docs/gotchas/` mining in pkg 100 | Zig | macOS, iOS (static lib) | launcher (`mlx-serve --serve`) |
| [Ollama](https://ollama.com/blog/mlx) | MLX backend on Apple Silicon by default since v0.30 (2026-05-13); uses M5 neural accelerators | Go + MLX | **ollama library (⚠︎ different artifacts)** | macOS | attach (`OLLAMA_HOST=127.0.0.1:11435`) |
| [LM Studio](https://github.com/lmstudio-ai/mlx-engine) | mlx-engine; the largest consumer MLX surface | Python/Electron | mlx-community (if pointed at them) | macOS | attach (`lms server start`) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | the GGUF/Metal baseline everybody quotes | C++ | **GGUF (⚠︎)** | macOS, iOS | attach |
| Local AI Cat legacy `LLMEvaluator` | our own pre-mlxcat in-app path (mlx-swift-lm `TokenIterator`) — the regression floor | Swift | mlx-community | macOS, iOS | attach (Local API, engine forced legacy) |

## Watchlist (drift-tracked; candidates for rows or ports)

| project | why we watch it | lang | platforms | status |
|---|---|---|---|---|
| [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | **dependency** — model loaders, KV caches; pinned by revision in `Package.swift` | Swift | both | pinned |
| [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) | **dependency** — the framework; tag cadence lags `mlx` core | Swift/C++ | both | pinned |
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | core; kernel changes land here first (e.g. M5 neural-accelerator paths) | C++ | both | tracked |
| [argmaxinc/WhisperKit](https://github.com/argmaxinc/argmax-oss-swift) | **dependency** — `MLXCatSpeechWhisperKit`. The repo is now `argmaxinc/argmax-oss-swift`; the old URL redirects, so the pin still resolves | Swift | both | pinned |
| [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | **dependency** via the `atlas-open-sources` fork — tokenizers/Jinja | Swift | both | pinned (fork) |
| [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine) | disk-chunked KV (PR #326), continuous batching; 2026-08-19 "fix high memory during gemma4 image prefill" is directly relevant to our 16k memory work | Python | macOS | reference |
| [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | checkpoint authority for our VLM weights; DFlash/EAGLE-3/MTP spec-decode reference | Python | macOS | reference |
| [Trans-N-ai/swama](https://github.com/Trans-N-ai/swama) | native **Swift** MLX engine + macOS app + iOS app — the closest sibling to mlxcat; candidate for both platforms' rows | Swift | macOS, iOS | attach when installed |
| [SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM) | native Swift server; SSD streaming for 100B+ MoE, TurboQuant KV, iOS support | Swift | macOS, iOS | attach when installed |
| [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) | "Ollama replacement", prompt caching, claims 2–4× over Ollama-llama.cpp (pre-MLX Ollama) | Python | macOS | attach when installed |
| [madroidmaq/mlx-omni-server](https://github.com/madroidmaq/mlx-omni-server) · [cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server) · [arcee-ai/fastmlx](https://github.com/arcee-ai/fastmlx) · [PicoMLX/PicoMLXServer](https://github.com/PicoMLX/PicoMLXServer) | OpenAI-compatible MLX servers; thin wrappers over mlx-lm — one row (`mlx-lm`) represents the class | Python | macOS | class represented |
| [exo-explore/exo](https://github.com/exo-explore/exo) · [mzbac/mlx_sharding](https://github.com/mzbac/mlx_sharding) | distributed/sharded MLX inference — out of scope for a single-device leaderboard, watched for the cluster idea | Python | macOS | watch |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | scheduler design source (token-budget scheduling, chunked prefill, per-request RNG) — `docs/planning/GOAL-mlxserve-engine-hardening.md` | Python | — | reference |
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) (lib) | the batched-decode mechanism mlxcat's Track B follows (`BatchKVCache`, ragged mask) | Python | — | reference |
| DFlash 2 ([inco.ai](https://inco.ai/blog/dflash2)) | parallel spec-decode with released drafters for Qwen3.8-27B / Muse-Glimmer-30B; claims 2.7–3.4× decode — the drafters' licence is unverified, but two open MLX implementations are now watched below (Apache-2.0 and MIT) | — | macOS | ceiling measurement pending |
| [google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) | wins Gemma-4 E2B on iPhone 17 Pro in neutral benchmarks (native `.litertlm` INT4-QAT on Metal) — the iOS competitor for Gemma | C++ | iOS, macOS | iOS comparison set |
| Apple Foundation Models | the ~3B system model; the "why not just use Apple's" baseline on iOS 26+ | Swift API | iOS, macOS | iOS comparison set |
| Apple Core AI | iOS 27 / macOS 27 runtime, successor to Core ML; runs exported `.aimodel` LLMs on GPU or Neural Engine. The neutral benchmark reports Qwen3-8B 4-bit at 94 tok/s on an M4 Max GPU vs MLX 90 under one protocol | Swift API | iOS, macOS | iOS comparison set |
| [pytorch/executorch](https://github.com/pytorch/executorch) · [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) · [Anemll/Anemll](https://github.com/Anemll/Anemll) (ANE) | other on-device runtimes in the neutral iPhone benchmark set | C++ | iOS | iOS comparison set |
| [john-rocky/apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench) | the neutral Mac + iPhone + iPad benchmark (MLX Swift, llama.cpp, CoreML, LiteRT-LM, ExecuTorch, ANEMLL, Foundation Models); MIT; JSONL → `LEADERBOARD.md` with CI consistency checks — our schema mirrors its shape so rows can be cross-submitted. Since 2026-08 it mirrors `edge-llm-bench` (next row) | Swift | both | methodology reference |
| [john-rocky/edge-llm-bench](https://github.com/john-rocky/edge-llm-bench) | where the neutral benchmark is developed now: macOS / iOS / Android, adds Apple Core AI and Cactus arms, thermal gating and session-keyed comparisons; MIT | Swift | both | methodology reference (added 2026-10-02) |
| [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) | vLLM's scheduler and paged block manager on MLX, with a paged variable-length Metal attention kernel and M5 tensor-unit prefill — the live data point for the GPU-resident paged attention we did not pursue; Apache-2.0 | Python | macOS | reference (added 2026-10-02) |
| [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | DFlash block-diffusion speculative decoding on stock MLX; tape-replay rollback for gated-delta recurrent state; reports 3.0–3.7× decode on Qwen3.5-4B; Apache-2.0 | Python | macOS | spec-decode reference (added 2026-10-02) |
| [ARahim3/mlx-dspark](https://github.com/ARahim3/mlx-dspark) | DSpark and DFlash drafters under one lossless verify loop, measured on an M4 Pro (Gemma-4 12B, Qwen3.8-27B, Qwen3-8B; short and 16k–32k contexts); MIT | Python | macOS | spec-decode reference (added 2026-10-02) |
| [drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) | Swift + Metal runtime that runs Gemma 4 26B-A4B in ~2 GB by streaming routed experts from SSD; publishes 103 measured experiments; Apache-2.0 | Swift | macOS | memory reference (added 2026-10-02) |

### Not pursued

* Adopting the Zig core (mlx-serve), ds4, or any non-MLX engine as a runtime — ruled out.
* GGUF beyond the existing `LLMEvaluator` escape hatch in the app.
* GPU-resident paged attention: mlx-lm #610/#629 were closed upstream; mlxcat (like oMLX) reconstructs contiguous K/V from cached blocks.

## Where we actually stand

`docs/COMPETITIVE.md` reads the committed rows and says so plainly: fourth of
four on a cell-by-cell geometric mean, competitive at c1 and collapsing under
concurrency — because `usesSerializedDecode` has batched decode switched off for
five of the six benchmark models. Read it before optimising anything.

## Keeping this honest

* This file is a **living document** (the `living:` block at the top): a weekly
  pass follows [`docs/watch/CHECKER.md`](watch/CHECKER.md), appends what the
  watched repos learned to [`docs/watch/LEARNINGS.md`](watch/LEARNINGS.md), and
  advances the per-repo read cursors in
  [`docs/watch/cursors.json`](watch/cursors.json). A monthly discovery pass
  ([`docs/watch/DISCOVERY.md`](watch/DISCOVERY.md)) adds and retires repos.
* `scripts/donor-drift.sh` prints, for every pinned dependency and every
  watchlist repo, the newest release and the commits since our pin or since the
  repo's read cursor (with keyword hits: cache, prefill, memory, batch, rope,
  gemma, qwen, kv, stream). The `donor-drift.yml` workflow runs it weekly and
  upserts an issue in this repo.
* A watchlist entry becomes a benchmarked engine by adding a block to
  `bench/engines.json` — a launch template or an attach URL — nothing else.
* Local clones of the reference engines live in `guest/` (gitignored), so the
  whole reference shelf travels with the repo checkout.

Sources for the 2026-08 survey: [Ollama MLX blog](https://ollama.com/blog/mlx),
[Ollama v0.30 stable MLX](https://runaihome.com/blog/ollama-v030-mlx-stable-upgrade-2026/),
[MLX vs llama.cpp on Apple Silicon](https://yage.ai/share/mlx-apple-silicon-en-20260331.html),
[Local LLM on iPhone: which runtime is fastest](https://rockyshikoku.medium.com/local-llm-on-iphone-which-runtime-is-actually-fastest-58096685481e),
[awesome-mlx](https://github.com/raullenchai/awesome-mlx) (a fork, quiet since
2026-04; its maintained upstream is
[antranapp/awesome-mlx](https://github.com/antranapp/awesome-mlx)),
[vllm-mlx](https://github.com/waybarrios/vllm-mlx),
[apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench).
