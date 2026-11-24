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

## Core A/B benchmark

The existing `bench/run.py` short-context arm uses width one, three timed cold
samples, one warm-up and the harness's quiet-machine and memory checks. The
greedy check uses the same fixed coding prompt, temperature zero and a
64-token cap on both cores and compares emitted IDs and stop reasons.
