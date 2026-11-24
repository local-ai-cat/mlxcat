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

## Landing qualification follow-up

### Fixed-prompt greedy differences

The same coding prompt was scored with Python on MLX 0.32.4. mlx-lm 0.31.3
ran the Qwen checkpoints and mlx-vlm 0.8.0 ran the affine Inkling pack.
The original HTTP request supplied a top-level reasoning field, so the
server used the template defaults. The native processors and Python
produced identical input IDs with those defaults on all three packs.
The reference followed the shared greedy prefix through the changed
position in every case.

| Model | Zero-based output position | Old core's common prefix with Python | New core's common prefix with Python | Old token and logit | New token and logit | Python margin |
| --- | ---: | ---: | ---: | --- | --- | --- |
| DaisyRose-27B-4bit | 23 | 64/64 | 23/64 | 4222 at 27.625 | 13 at 27.500 | 0.125 for old |
| Qwen3.6-35B-A3B-4bit | 31 | 64/64 | 31/64 | 1132 at 27.875 | 264 at 27.750 | 0.125 for old |
| VisualInference Inkling | 50 | 50/64 | 64/64 | 170248 at 27.62344 | 196980 at 27.65937 | 0.03593 for new |

Both Qwen flips are one BF16 logit step. In all three cases the two
candidates are Python's top two tokens at the changed position.

### Layer-zero gated delta

A 25-row coding prompt supplied real checkpoint tensors to the standalone
recurrence probe. With identical Q, K, V, gamma and beta, the original
Swift recurrence had low-order differences in a few BF16 output elements
and in the FP32 state. Its compensated K-state lane sum differs from
mlx-lm's ordinary sum. The largest state delta was 3.81e-6. The core's
separate chunked GDN primitive also differed on the same tensors.
[Original Swift recurrence and beta conversion](https://github.com/atlas-open-sources/mlx-swift-lm/blob/ed4f3eabf1022e4fc88e0cc8f5022ae6de67694c/Libraries/MLXLMCommon/GatedDelta.swift#L56-L88),
[Python recurrence](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/gated_delta.py#L66-L89).

Feeding the exact reference recurrence output and Z into Swift isolated a
larger difference at the output gate. The old Swift gate used BF16 SiLU and
multiplication. Python promotes the operands, multiplies in FP32, and casts
once. The old gate reached 0.015625 on Qwen3.8 and Kestrel and 0.03125 on
DaisyRose. The output projection was exact when fed the Python gate output.
[Original Swift gated RMSNorm](https://github.com/atlas-open-sources/mlx-swift-lm/blob/ed4f3eabf1022e4fc88e0cc8f5022ae6de67694c/Libraries/MLXVLM/Models/Qwen35.swift#L322-L327),
[Python gated RMSNorm](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3_next.py#L58-L78).

On Kestrel the fused prefill projection also changed 18 of 204,800 QKV
elements by up to 0.03125. Concatenating the output columns changes the
split-K tile count. Independent projections matched the Python operands.
The VLM route now uses independent projections for prefill, FP32 gate
arithmetic, and the reference recurrence. The compensated recurrence
remains available to other callers. Single-step decode keeps the fused
projection.
[Scoped mlx-swift-lm change](https://github.com/atlas-open-sources/mlx-swift-lm/commit/3123a08).

Use `Benchmarks/gated_delta_reference.py CHECKPOINT FIXTURE OUTPUT` with
mlx-lm 0.31.3 and MLX 0.32.4. Run the
`GatedDeltaProjectionProbeTests` filter with `MLXCAT_GDN_INPUT` set to the
capture and `MLXCAT_GDN_MODEL` set to the original pack.

The regression probe on the shipped pin matched all captured recurrence
outputs, FP32 recurrent-state elements and layer-zero attention outputs.
In the coding trace, complete decoder layers 0–2 matched on all three
models. The first remaining delta is at decoder layer 3, the first
full-attention block.

| Model | Recurrence output exact | FP32 state exact | Layer-zero attention exact |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B-4bit | 153,600/153,600 | 786,432/786,432 | 128,000/128,000 |
| Kestrel-4B-Coder-4bit | 102,400/102,400 | 524,288/524,288 | 64,000/64,000 |
| DaisyRose-27B-4bit | 153,600/153,600 | 786,432/786,432 | 128,000/128,000 |

| Model | Full greedy, original candidate → GDN fix | End-to-end max logit delta, original candidate → GDN fix | First remaining coding-trace delta (layer 3) |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B-4bit | 307/307 → 307/307 | 0.75 → 0.75 | 0.00390625 |
| Kestrel-4B-Coder-4bit | 304/330 → 256/330 | 0.625 → 0.5 | 0.0078125 |
| DaisyRose-27B-4bit | 293/307 → 293/307 | 0.75 → 0.75 | 0.25 |

Kestrel's literal fixture became exact, and its reasoning and short-chat
fixtures diverged earlier. All three still exceed the 0.25 end-to-end
tolerance, and the tool prompt key-order difference remains.

### Qwen3-1.7B repeat and allocation trace

Three paired release cells ran under the same lease, with three timed
samples per cell and old/new order reversed in the second pair.

| Pair | Decode tok/s, old → new | TTFT ms, old → new | Peak MiB, old → new | Extra peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 167.36 → 166.24 | 125.49 → 123.43 | 2,275.97 → 2,349.85 | 73.88 |
| 2 | 166.37 → 166.32 | 126.32 → 124.95 | 2,278.25 → 2,342.11 | 63.86 |
| 3 | 167.35 → 166.60 | 125.45 → 129.16 | 2,277.17 → 2,349.83 | 72.66 |

The original 3.1% decode drop did not recur. The paired drops were
0.03–0.67%, and the within-cell decode ranges overlapped in every pair.

Peak memory increased in all three pairs. A process-footprint control
showed equal startup footprints and a lower new-core peak for the
nine-token warm-up. A 735-token prompt with one output token reproduced
a 78.47 MiB increase, localizing it to prefill. A Metal encoder trace
captured the new `affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true` followed
by `steel_gemm_splitk_accum_bfloat16_float32`. For a 219-row projection with
K=2,048 and N=1,024, the split kernel bound a 1,802,240-byte partial-sum
buffer. Across 56 split-K launches the trace observed 38 distinct
partial-sum buffer identities totaling 65.31 MiB. The old QMM goes
directly to the output. The buffer total describes the trace, while
the process peak also covers other working allocations.
[Old core dispatch](https://github.com/atlas-open-sources/mlx/blob/5186c44bf7d10c07a4e4fc7838da881c6c031c33/mlx/backend/metal/quantized.cpp#L1315-L1339),
[New split-K allocation and reducer](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L1180-L1277).
