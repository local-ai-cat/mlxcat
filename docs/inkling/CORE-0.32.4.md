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
candidates are Python's top two tokens at the changed position. The
release rerun after the gated delta fix retained the same three
match lengths against Python.

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
fixtures diverged earlier. Python's margin at the new reasoning flip
was 0.125, and the short-chat candidates tied at 19.875. All three still
exceed the 0.25 end-to-end tolerance, and the tool prompt key-order
difference remains.

On this pin, both Inkling packs repeated their 760/760 greedy and
960/960 sampled-logit matches at zero maximum error. The Swift suite
reported 572 XCTest cases with zero failures and 69 opt-in skips.
All 10 Swift Testing cases passed.

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

### Layer-three full attention

The Kestrel regression after the gated-delta correction was concentrated in
two fixtures. The score sums the common greedy prefix of each fixture. A
single early miss excludes the rest of that fixture from the count even
though the harness continues on the Python prefix and compares the logits.

| Kestrel fixture | Before GDN correction | After GDN correction | With fused RoPE |
|---|---:|---:|---:|
| Literal | 1/19 | 19/19 | 19/19 |
| Reasoning | 56/64 | 10/64 | 56/64 |
| Short chat | 23/23 | 3/23 | 3/23 |
| Total across 11 fixtures | 304/330 | 256/330 | 302/330 |

At reasoning index 10, Python scored its winner (15060) at 19.75 and
mlxcat's winner (13) at 19.625. At short-chat index 3, Python scored both
353 and 10252 at 19.875 and selected 353. These are zero-based output
indices. Reducing the overall max logit error did not guarantee preserving
these local rankings. The earlier GDN error had moved the logits enough
for the legacy rotation path to pick the Python token at both positions.
With GDN corrected, blocks 0–2 matched on both teacher-forced steps and
the first nonzero layer boundary was block 3 (0.001953125 for reasoning,
0.0009765625 for short chat). Replacing only the rotation zeroed the
reasoning boundary and restored the greedy prefix through index 55.
Short chat still differed at block 3 by 0.00048828125 and selected the
other tied candidate.

The real-weight layer-3 probe held the Python input fixed, replayed each
operator independently and compared BF16 outputs. Q/K/V projections,
Q/K RMS normalization, KV writes, output gating and output projection were
exact on their respective reference operands. Legacy M-RoPE first changed
the normalized queries and keys. Its largest query delta was 0.03125 on
each pack in the code prefill. Swift calculated the angles with reciprocal
powers, cast cos/sin to BF16, and multiplied and added separate BF16
tensors. mlx-lm's default text RoPE called the core's fused
half-split rotation, using `exp2`, FP32 trig and FP32 rotation before the
final BF16 write. The scalar and per-row offset APIs run that same
rotation. [Original Swift Q/K normalization and full attention](https://github.com/atlas-open-sources/mlx-swift-lm/blob/3123a08d910ae80f399845d5172c1e6c570608dc/Libraries/MLXVLM/Models/Qwen35.swift#L379-L430), [Original Swift rotary arithmetic](https://github.com/atlas-open-sources/mlx-swift-lm/blob/3123a08d910ae80f399845d5172c1e6c570608dc/Libraries/MLXVLM/Models/Qwen35.swift#L232-L309), [Python Q/K and rotary path](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3_next.py#L113-L158), [Python default RoPE](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/rope_utils.py#L235-L251), [Metal rotation](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/kernels/rope.metal#L20-L65).

Contiguous positions now take the fused default path, using the start of
each row across prefill, decode and windowed continuation. Image and video
spans retain explicit temporal, height and width coordinates. The code
prefill is now exact through layer 3 on all three packs. This initial
comparison used the default package-bundled metallib in XCTest.

| Model | GDN fix greedy | Fused RoPE greedy | Sampled logits | Max delta before → after | First remaining code-prefill boundary |
|---|---:|---:|---:|---:|---|
| Qwen3.8-27B-4bit | 307/307 | 307/307 | 808 | 0.75 → 0.5 | Layer 46, 0.25 |
| Kestrel-4B-Coder-4bit | 256/330 | 302/330 | 856 | 0.5 → 0.375 | All 32 blocks exact |
| DaisyRose-27B-4bit | 293/307 | 307/307 | 808 | 0.75 → 0.625 | Layer 7, 0.00390625 |

These counts use the same Python token IDs in both engines. The separate
tool-schema key-order difference still changes the rendered prompt on all
three packs. With that fixture excluded, the numerical maxima are 0.375,
0.375 and 0.5, respectively, above the 0.25 gate.

With identical rotated Q/K and BF16 V, the causal descriptor and explicit
boolean mask produced identical output for the code prefill and both
captured 1,024-row chunks on all three packs. The Kestrel short-chat decode
probe still found one different output out of 4,096 (0.0009765625) with
both the original and contiguous Python operands. The separate gate and
output projection replays were exact. Swift and Python both pass ordinary
BF16 KV caches to the core. Metal uses the vector kernel for at most eight
query rows, with float accumulators. The captured single-row case takes
the one-pass vector path. [Swift SDPA and KV dispatch](https://github.com/atlas-open-sources/mlx-swift-lm/blob/3123a08d910ae80f399845d5172c1e6c570608dc/Libraries/MLXLMCommon/AttentionUtils.swift#L46-L111), [Python SDPA dispatch](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py#L108-L137), [Metal SDPA dispatch](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/scaled_dot_product_attention.cpp#L1507-L1581), [Vector precision](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/kernels/sdpa_vector.h#L50-L160).

Reproduce with `Benchmarks/full_attention_reference.py CHECKPOINT FIXTURES
OUTPUT_DIRECTORY` in the mlx-lm reference environment. Run the
`FullAttentionProjectionProbeTests` filter with `MLXCAT_ATTN_MODEL` set to
the same checkpoint and `MLXCAT_ATTN_CAPTURE` set to the output directory.
The probe also accepts a full-attention layer index through `--layer` in
Python and `MLXCAT_ATTN_LAYER` in Swift.

The compiled Metal library explains additional differences after the Swift
path uses the same RoPE operator as Python. With the default bundled
library, both scalar and array offsets reproduce the 1,024-row residual
even when Q/K are bit-exact and the head/sequence layout is preserved. For example,
Kestrel's second chunk changes 12 of 4,194,304 rotated query values, with
a maximum delta of 0.0078125. DaisyRose's first code-prefill difference
after layer 3 is one rotated query value at layer 7, also 0.0078125.

Explicitly loading each of the two stock mlxcat artifacts isolated
the build difference. The package resource compiled by Swift Build
uses `-fmetal-math-mode=fast` and `-fmetal-math-fp32-functions=fast`.
The standalone builder compiles with `-fno-fast-math`. Both stock
artifacts use the same source, compiler 32023.921 and macOS 14
deployment target. The standalone artifact zeroed the captured
long-window RoPE differences and the single-row SDPA difference.
The embedded artifact reproduced both. [Standalone builder](../../scripts/build-metallib.sh#L29), [Metal RoPE dispatch](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/rope.cpp#L44-L120).

The Python wheel metallib and a local macOS 26.2 rebuild also zeroed the
captured differences. A precise sine/cosine trial compiled for macOS 14
matched the Python readings, including operators outside RoPE. The
stock standalone artifact gave the same result without that kernel
patch. The retained model change uses the existing fused operator and
the supported deployment target.

Before aligning the prefill schedule, the standalone fixture run on
macOS 14 matched all greedy IDs on the reference prompts. All sampled logits outside the 8,225-token
fixture were exact, including the tool case when its input IDs were
held fixed.

| Model | Standalone greedy | Sampled logits | Max delta | Numerical gate ≤ 0.25 |
|---|---:|---:|---:|---|
| Qwen3.8-27B-4bit | 307/307 | 808 | 0.375 | Above gate |
| Kestrel-4B-Coder-4bit | 330/330 | 856 | 0.25 | Within gate |
| DaisyRose-27B-4bit | 307/307 | 808 | 0.375 | Above gate |

The macOS 26.2 and precise sine/cosine controls produced the same
fixture results. A minimal Swift Build shader package kept its fast-math
flags when given an `XCODE_XCCONFIG_FILE` override. Closing the embedded
gap requires control over package shader compilation.

The 1,024-row prefill also differs before full attention with the bundled
library. The first nonzero boundary is layer 1 on all three packs. The
reference metallib restores that boundary and the next two in the
captured first chunks on Qwen3.8 and Kestrel. In the separate Qwen3.8
short-prompt trace, layer 46 still differs in the end-to-end run, but
replaying the captured layer's affine projections, recurrent update,
attention output and MLP individually gives exact outputs. That trace
localizes a boundary without assigning the difference to one of those
operators.

Both official Inkling packs remain exact at 760/760 greedy IDs and
960/960 sampled logits each, including the image fixture. The pinned
Swift suite ran 573 XCTest cases with 70 opt-in skips and zero failures,
and all 10 Swift Testing cases passed. The opt-in Kestrel ragged-width-4
and prefix tests passed. The width-8 stress test had one divergent row
before and two after the RoPE change, so width-8 token parity remains
unqualified for this pack. The unchanged strict Metal 3 matrix passed
all 24 checks. The benchmark harness passed 28 tests and its parity and
leaderboard checks passed.

The standalone fixture run can be reproduced by pointing
`MLXCAT_SURVEY_METALLIB` at the library generated by
`scripts/build-metallib.sh`. The full-attention probe takes the same
override through `MLXCAT_ATTN_METALLIB`. Use a separate process for
each artifact because MLX opens the device library once per process.

### Match long-prefill shapes

The sampled-logit difference on the standalone build came from comparing
different prefill schedules. The Python fixture reserved the final token
and used eight 1,024-row windows and a 32-row remainder. The Swift
fixture passed the same ceiling but inherited balanced chunking, which
used eight 914-row windows and a 912-row remainder. Both then used a
one-row forward for the reserved token. Quantized prefill can round
differently when the number of rows in an individual forward changes.

Replaying the fixed schedule with the standalone metallib matched every
layer's row fingerprints on the 8,225-token Qwen3.8 and DaisyRose
prompts. The benchmark now records its chunking mode and the Swift
fixture reproduces it. Existing fixtures encode the Python fixed
windows by default. Use `Benchmarks/survey_reference.py --chunking balanced`
to regenerate the production-default schedule. [Swift prefill strategy](https://github.com/atlas-open-sources/mlx-swift-lm/blob/b984a96fa056a837b1c7b7f79615d4203e6ab191/Libraries/MLXLMCommon/PrefillParameters.swift#L40-L104).

| Model | Fixed windows greedy | Fixed windows sampled logits | Balanced windows greedy | Balanced windows sampled logits |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-4bit | 307/307 | 808/808 exact | 307/307 | 808/808 exact |
| Kestrel-4B-Coder-4bit | 330/330 | 856/856 exact | 330/330 | 856/856 exact |
| DaisyRose-27B-4bit | 307/307 | 808/808 exact | 307/307 | 808/808 exact |

The six standalone runs all had zero maximum logit delta. The sole
failed assertion in each opt-in fixture run was the pre-existing
tool-schema key order, first differing at prompt token 20. These
numerical results use the Python prompt IDs for that case. The two
official Inkling packs each retained exact greedy IDs and sampled
logits. The standard Swift suite stayed green after the fixture change.

### Embedded macOS Metal library

The macOS host links `MLXCatNative` in-process. Its original app
layout resolves `Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`.
The host build's shader command used `-fmetal-math-mode=fast` and
`-fmetal-math-fp32-functions=fast` under Swift Build's package resource
rule. The sidecar has a separate executable directory and colocated shader
lookup.

The pinned macOS Cmlx dependency now generates a separate
`mlx-swift_CmlxMacMetal.bundle` through a package build tool. The tool
and the standalone script call the same compiler wrapper, which uses
`-fno-fast-math` and the macOS 14 deployment target. The native loader
checks the precise resource before the compatibility bundle. A colocated
standalone library or explicit MLX override still has priority. The
precise resource is a macOS-only target dependency.
[Package dependency](https://github.com/atlas-open-sources/mlx-swift/blob/15405f9642a6da7877c9964a688c807a641c100e/Package.swift#L270-L275),
[Shared Metal builder](https://github.com/atlas-open-sources/mlx-swift/blob/15405f9642a6da7877c9964a688c807a641c100e/tools/build-macos-metallib.sh#L1-L20),
[Native resource lookup](https://github.com/atlas-open-sources/mlx/blob/8025c1b83254a5a2a88905faa06fd5e1691f578a/mlx/backend/metal/device.cpp#L212-L251).

For Qwen3.5 tool declarations, the Swift dictionary-to-Jinja adapter sorted
every object lexically before `tojson`. The native tokenizer now passes an
ordered Jinja object for the standard OpenAI declaration fields. Extra keys
keep deterministic ordering because the request dictionary has already
discarded caller insertion order. The unit test renders a complete weather
declaration. The three real tool fixtures each match all 281 prompt IDs.
[Tokenizer template context](https://github.com/atlas-open-sources/swift-transformers/blob/089cb3f02a1718b2943c7e7c4553876cd51a75d1/Sources/Tokenizers/Tokenizer.swift#L790-L808).
