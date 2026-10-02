# Discovery: finding repos worth watching

The watchlist in [`docs/ENGINES.md`](../ENGINES.md) only stays useful if new
projects can get onto it and dead ones can leave. This file records each
discovery pass: which methods were tried, what each turned up, how much of it was
noise, and which candidates were added or declined. The point of keeping the
method notes is to learn which searches are worth repeating.

## Entry test

A candidate joins the watchlist only if it passes all three:

1. **Alive.** A commit on its default branch in the last 90 days.
2. **Measurable and new to us.** It does something we can measure (a speed,
   memory or correctness claim with numbers, or a methodology we can apply) that
   mlxcat does not already do.
3. **Readable licence.** A `LICENSE` file we can read. Code is only ever copied
   from Apache-2.0-compatible licences; for everything else we learn the
   mechanism and write our own.

At most five additions per pass. Everything else is recorded as declined with one
line, so the next pass does not re-evaluate it from scratch.

---

## Pass 2026-10-02

Window: repos pushed since 2026-07-04 (90 days).

### Methods and what they turned up

| # | Method | Raw hits | Relevant to an inference engine | Noise | Worth repeating? |
|---|---|---:|---:|---|---|
| 1 | Links, credits and dependencies in watched repos' recent PRs and READMEs (collected while reading, not tallied separately) | roughly 15 named projects | about 6 | low: almost every mention is another engine or kernel source | **Yes.** Highest signal per minute. oMLX and mlx-serve credit each other's kernels by PR number, and that chain led to the oMLX kernel PRs. |
| 2a | Forks of `mlx-swift` and `mlx-swift-lm`, sorted by stars, pushed in window | 24 | 1 (`PrismML-Eng/mlx-swift`, a low-bit kernel fork 31 commits ahead on branch `prism`) | very high: nearly all forks are personal mirrors with 0–4 stars | Rarely. Check only the top two or three forks by stars. |
| 2b | Public dependents of `mlx-swift-lm` (GitHub dependency graph) | 0 | 0 | n/a | **No.** The dependency graph does not index SwiftPM, so it reports zero dependents. |
| 2c | Code search: `"ml-explore/mlx-swift-lm"` in `Package.swift` | 352 files / 96 repos, 25 pushed in window | 4 (OmniInfer, AnyLanguageModel, LocalLLMClient, Atomic-Chat) | high: mostly apps that embed the library (voice, notes, chat UIs) | Yes, as the replacement for 2b, but expect ~85 % apps. |
| 3 | The curated list we cite, `raullenchai/awesome-mlx` | 0 since 2026-04-11 | 0 | n/a | **Switch source.** That repo is a fork whose last push was 2026-04-11. Its upstream, [`antranapp/awesome-mlx`](https://github.com/antranapp/awesome-mlx), is maintained: 16 links added since 2026-08-01, of which 2 are inference-relevant (MTPLX, already a bench arm; claude-code-local, a wrapper). |
| 4 | GitHub search: `topic:mlx`, `mlx inference server`, `mlx-swift`, `topic:apple-silicon llm`, `speculative decoding mlx`, `kv cache mlx`; pushed ≥ 2026-07-04; star floor 100 (30 for the two narrow queries, 50 for `mlx-swift`, 200 for the broad topic) | 79 unique | ~14 | moderate: voice/TTS studios, image/video generators and agent shells top the star-sorted list | **Yes.** It found three of the five additions. The narrow keyword queries (`speculative decoding mlx`, `kv cache mlx`) had the best signal-to-noise. |
| 5 | The engine list of the neutral benchmark `john-rocky/apple-silicon-llm-bench` | 6 runtimes + 2 new arms (Apple Core AI, Cactus) | 3 | none | **Yes.** It also showed that the benchmark's active development has moved to [`john-rocky/edge-llm-bench`](https://github.com/john-rocky/edge-llm-bench); the repo we watch is now a mirror. |

**What we learned about the methods.** Credits and links in repos we already
watch (method 1) and narrow keyword searches (method 4) found almost everything
useful. Star-sorted topic searches surface popular apps, not engines. GitHub's
dependency graph is useless for SwiftPM, and code search is the substitute. A
curated list must be checked for being a stale fork before it is trusted.

### Added to the watchlist (5)

| repo | licence | last push | why it passes |
|---|---|---|---|
| [john-rocky/edge-llm-bench](https://github.com/john-rocky/edge-llm-bench) | MIT | 2026-10-01 | The active successor of the neutral benchmark we already watch (`apple-silicon-llm-bench` now mirrors it). It adds Apple Core AI and Cactus arms, thermal gating and session-keyed comparisons. Measurable: its rows are the cross-engine numbers our iOS claims are judged against. |
| [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) | Apache-2.0 | 2026-10-02 | vLLM's scheduler and paged block manager on MLX, with a paged variable-length Metal attention kernel (v0.2.0 reported 83× TTFT, 3.6× throughput over v0.1.0) and M5 tensor-unit prefill for MHA/GQA/MQA. GPU-resident paged attention is the path [`ENGINES.md`](../ENGINES.md) records as not pursued; this is the live data point for whether that ruling still holds. |
| [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | Apache-2.0 | 2026-08-20 (15 commits since 2026-07-04) | A DFlash block-diffusion speculative decoder on stock MLX. It reports 3.0–3.7× decode on Qwen3.5-4B (one of our matrix models) with 86–88 % acceptance on an M5 Max. It brings tape-replay rollback for gated-delta recurrent state, which is the hard part of speculating on hybrid models. Apache-2.0 also settles the "licence unverified" note on the DFlash row. |
| [ARahim3/mlx-dspark](https://github.com/ARahim3/mlx-dspark) | MIT | 2026-10-01 | Two lossless EAGLE-family drafters (DSpark and DFlash) under one verify loop, measured on an **M4 Pro** with medians of 3. That is the same chip class as one of our bench machines, so its speedups are directly comparable. It covers Gemma-4 12B, Qwen3.8-27B and Qwen3-8B, and has long-context (16k–32k) numbers. |
| [drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) | Apache-2.0 | 2026-09-27 | A Swift + Metal runtime that runs Gemma 4 26B-A4B in a ~2 GB budget by keeping the shared core and KV resident and streaming only the routed experts from SSD. It publishes an inventory of 103 measured experiments (kernels, caching, I/O, prefill, decode). It is Swift, and memory-bound, which is the iOS problem. |

### Considered and declined

| repo | why not |
|---|---|
| [youssofal/MTPLX](https://github.com/youssofal/MTPLX) | Already a bench arm (`mtplx` in `bench/engines.json`); passes the test, so the next pass should add it to the drift list. Held back only by the cap of five. |
| [PrismML-Eng/mlx-swift](https://github.com/PrismML-Eng/mlx-swift) (branch `prism`) | Low-bit (ternary) kernel fork of a dependency, MIT. A kernel source, not an engine; watch through its upstream PRs if they land. |
| [carloslfu/slotstream](https://github.com/carloslfu/slotstream) | SSD weight streaming in Swift, MIT. Overlaps turbo-fieldfare and SwiftLM, which already cover the idea. |
| [cactus-compute/cactus](https://github.com/cactus-compute/cactus) | Licence not detected by SPDX; not read this pass. Its numbers reach us through edge-llm-bench. |
| [osaurus-ai/osaurus](https://github.com/osaurus-ai/osaurus) | Agent harness and app; inference is delegated. |
| [jjang-ai/vmlx](https://github.com/jjang-ai/vmlx) | A mixed-precision quant format and its runtime; outside what mlxcat serves. |
| [omnimind-ai/OmniInfer](https://github.com/omnimind-ai/OmniInfer) | Rust edge-inference stack, not MLX. |
| [mudler/vllm.cpp](https://github.com/mudler/vllm.cpp) | C++ continuous-batching engine, not MLX; vllm and vllm-metal cover the scheduler ideas. |
| [huggingface/AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel) · [tattn/LocalLLMClient](https://github.com/tattn/LocalLLMClient) | API facades over mlx-swift-lm; no engine-level mechanism of their own. |
| [AtomicBot-ai/Atomic-Chat](https://github.com/AtomicBot-ai/Atomic-Chat) | App plus engine; licence not detected by SPDX. |
| [nicedreamzapp/claude-code-local](https://github.com/nicedreamzapp/claude-code-local) | API-compatibility server over existing engines. |
| [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) · [Blaizzy/mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift) · [soniqo/speech-swift](https://github.com/soniqo/speech-swift) | Speech/TTS; revisit if the speech target grows beyond WhisperKit. |
| [Epistates/pmetal](https://github.com/Epistates/pmetal) | Rust framework; licence not detected by SPDX. |
| [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) | Examples; changes reach us through mlx-swift-lm. |
| Voice, image, video and agent apps in the star-sorted results | Not inference engines. |

### Not a repo, but in the comparison set now

**Apple Core AI** (iOS 27 / macOS 27, the successor to Core ML) appears as an arm
in the neutral benchmark. It reports Qwen3-8B 4-bit at 94 tok/s on an M4 Max GPU
against MLX's 90 under the same protocol. It belongs beside Apple Foundation
Models in the iOS comparison set.
