# MLX core 0.32.4 qualification

The dependency change compares the same native mlxcat checkout and checkpoints
with MLX 0.31.1 and 0.32.4. The motivation is the affine projection result in
[Affine projection parity](AFFINE-PARITY.md). The reference for new affine
fixture runs is mlx-lm 0.31.3 on MLX 0.32.4.

## Metal 3

The existing gate exposed two failures in the unpatched core. Metal 3.0 rejected the
C++17 `if constexpr` in the shared constant loop. Once the loop compiled,
Metal 3.1 rejected more C++17 constructs in the attention translation unit
and its headers.

- The compile-time loop uses SFINAE base and recursive overloads. Its negative
  control still requires instantiation at index two.
- Attention uses ordinary branches on compile-time constants. The
  score-exchange assertion still checks its two-way shape. The shared complex
  layout and multiplication assertions have messages, and the complex limit
  templates use C++14 static constexpr declarations.
- The preamble generator keeps headers rooted in the MLX source tree.
  This retains the previous protection against embedding toolchain-specific
  paths in generated sources. Both offline Metal and JIT sources were
  regenerated from the fixed core. The JIT gate references the new shared
  cross-entropy read-width constant the same way it references the other
  shared constants.
- The standalone builder and XCTest fallback compile all generated Metal
  translation units. This includes `gather_mm_offsets`, newly called by the
  grouped quantized path. GEMV is provided through JIT on this core.

All 24 checks passed with the existing Metal language and deployment targets
and `-Werror`. The controls ran on Metal 3.0, 3.1 and 3.2 on macOS and iPhone.
The full attention translation unit compiled as source, generated Metal and
JIT input on Metal 3.1 and 3.2.
[MLX source change](https://github.com/atlas-open-sources/mlx/commit/163374f0e20c80fc323ef7ba09bb2ca314617be1),
[mlx-swift regeneration and gate](https://github.com/atlas-open-sources/mlx-swift/commit/603209460c5fed217427846dc9e40bdab55d3f5f).

## Runtime and model qualification

The new Swift compile API distinguishes Sendable functions from closures that
capture model state. Qwen's compiled decode and Inkling's compiled experts
use the explicitly executor-local overload.

MLX 0.32.4's categorical sampler uses inverse CDF for a one-row batch and
Gumbel-max for a scalar result. Under the same global seed, that made an
unseeded singleton request emit different tokens when batched with seeded
requests. Singleton sampling now gives MLX the same rank-one logits in either
case. The existing mixed-seed regression test passes.
[Core categorical dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/random.cpp#L412-L428).

Both Inkling packs initially reproduced every greedy token and all text
logits. The image fixture reached a maximum logit delta of 0.6686373 for
each pack. In a vision trace, pixels, patch embeddings and the first three
vision layers matched. Layer 3's input and unrotated queries and keys
matched. Its first mismatch came from rotary frequencies computed by
Metal's default power function. Identical FP32 exponents produced a
7.45e-9 frequency difference, which moved a rotated key by one BF16 step
(0.001953125). The explicit `metal::precise::pow` control reproduced the
reference frequencies, angles, sine, cosine and rotated keys exactly. It
compiled in Metal 3.1 for macOS and iPhone and restored a zero-error image
fixture on the first pack with the same LayerNorm AIR.

Both full Inkling fixture runs on the final code matched 760/760 greedy IDs
and 960/960 sampled logits with zero maximum error. XCTest reported 571 cases
with zero failures and 68 opt-in skips. All 10 Swift Testing cases passed.

## Affine model fixture results

All three packs matched the Python layer-0 input norm and the QKV, Z, B
and A projections for the coding prompt. The first remaining trace boundary
with a difference was the gated delta attention output. The fused and
separate Swift projection controls produced the same maximum delta there.

| Model | Greedy IDs on reference input IDs | Sampled logits | Max logit delta (gate 0.25) | Layer-0 gated delta output max |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B-4bit | 307/307 | 808 | 0.75 | 0.125 |
| Kestrel-4B-Coder-4bit | 304/330 | 856 | 0.625 | 0.015625 |
| DaisyRose-27B-4bit | 293/307 | 808 | 0.75 | 0.125 |

Each matched 10 of 11 rendered prompts. The tool fixture first differed
at input token 20 because of JSON key order. The model arithmetic
comparison used the reference IDs for that case. All three exceeded the
logit tolerance. Kestrel and DaisyRose therefore remain outside the
catalogue gate. Kestrel's measured Whisker T2 result is 0/3.

## Core A/B benchmark

The existing `bench/run.py` short-context arm uses width one, three timed cold
samples, one warm-up and the harness's quiet-machine and memory checks. The
greedy check uses the same fixed coding prompt, temperature zero and a
64-token cap on both cores and compares emitted IDs and stop reasons.

Both release arms ran under the exclusive lease on the M4 Pro. All 18
performance cells had three valid samples. The table shows old → new.
TTFT and decode are medians. Peak is the server process physical footprint.

| Checkpoint | Decode tok/s | TTFT ms | Peak GiB | Fixed-prompt greedy match | Relative changes |
| --- | ---: | ---: | ---: | --- | --- |
| DaisyRose-27B-4bit | 13.18 → 13.50 | 2485 → 2409 | 17.83 → 17.83 | 23 common, 64 → 64 IDs | decode +2.4%, TTFT -3.1%, peak 0.0% |
| Inkling-2-Mini-MLX-4-8bit | 22.45 → 22.49 | 1302 → 1309 | 22.37 → 22.39 | 64/64 exact | decode +0.2%, TTFT +0.6%, peak +0.1% |
| Inkling-2-Mini-MLX-M5Pro-48gb | 22.43 → 22.44 | 1309 → 1296 | 22.37 → 22.38 | 64/64 exact | decode 0.0%, TTFT -1.0%, peak +0.1% |
| Kestrel-4B-Coder-4bit | 68.85 → 68.99 | 338 → 330 | 4.25 → 4.26 | 64/64 exact | decode +0.2%, TTFT -2.6%, peak +0.3% |
| MTPLX-Bare-Speed-text | 27.10 → 27.18 | 1282 → 1271 | 17.17 → 17.22 | 64/64 exact | decode +0.3%, TTFT -0.9%, peak +0.3% |
| Qwen3-1.7B-4bit | 171.58 → 166.33 | 126 → 124 | 2.23 → 2.30 | 64/64 exact | decode -3.1%, TTFT -2.0%, peak +3.2% |
| Qwen3.6-35B-A3B-4bit | 65.05 → 68.97 | 385 → 342 | 20.29 → 20.35 | 31 common, 64 → 64 IDs | decode +6.0%, TTFT -11.2%, peak +0.3% |
| Qwen3.8-27B-4bit | 13.31 → 13.37 | 2451 → 2414 | 17.83 → 17.77 | 64/64 exact | decode +0.4%, TTFT -1.5%, peak -0.3% |
| VisualInference-Inkling-2-Mini-MLX-4bit | 25.66 → 25.88 | 1266 → 1247 | 17.96 → 17.95 | 50 common, 64 → 64 IDs | decode +0.9%, TTFT -1.5%, peak 0.0% |

Qwen3-1.7B had the only measured decode slowdown (3.1%) and the largest
relative peak increase (3.2%, 2.23 → 2.30 GiB). The Inkling 4/8-bit pack's
TTFT rose from 1,302 to 1,309 ms (0.6%). Qwen3.6-35B's decode improved 6.0%
and TTFT fell 11.2%. Six fixed greedy sequences matched all 64 tokens and
their stop reasons. DaisyRose, Qwen3.6-35B and VisualInference Inkling first
changed at zero-based output positions 23, 31 and 50, respectively.

The sweep covers the nine complete chat checkpoints found for this run,
including the archived Qwen3.6-35B MLX pack and MTPLX text weights. Ten
additional benchmark registry entries lacked complete local checkpoints:
Qwen3-0.6B, Llama 3.2 3B, Gemma 4 E2B/E4B/12B, Qwen3.5-4B,
Qwen2.5-Coder-7B, GPT-OSS-20B, Qwen3.6-27B and Qwen3-Coder-30B.

The 28 benchmark harness tests passed. The leaderboard and regression
snapshot checks passed. Raw measurement receipts are local.
