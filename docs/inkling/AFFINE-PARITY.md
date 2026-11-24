# Affine projection parity on M4

The Swift build embeds MLX 0.31.1. With identical activation and weight tensors,
it reproduced the Python MLX 0.31.1 projection outputs exactly. The same
operation changed when run against Python MLX 0.32.2 (the survey baseline) and
0.32.4 (the current comparison). Matching the reference requires the corresponding
Metal arithmetic and dispatch. The evidence below covers the three tested packs.

## Qwen3.8-27B-4bit fixture gate

Reference was mlx-lm 0.31.3 on MLX 0.32.4. The eleven text cases used the survey
fixture protocol, including 1,024-token prefill chunks and the final-token split
for the 8,225-token prompt. When rendering differed, the arithmetic comparison
used the Python prompt IDs. Sampled logits were evaluated along the reference
sequence.

| Check | Result |
| --- | --- |
| Rendered prompt IDs | 10/11 exact. Tool JSON key order changed the prompt from token 20. |
| Greedy IDs on identical prompts | 307/307 exact |
| Sampled logits | 808 compared, maximum absolute delta 0.75 (gate 0.25) |
| 25-token code fixture | 49/49 greedy, 80 logits, maximum delta 0.25 |
| 8,225-token fixture | 6/6 greedy, 48 logits, maximum delta 0.75 |
| First arithmetic difference on the code fixture | Layer 0 `in_proj_z`, maximum delta 0.015625. Embedding, input RMSNorm, and QKV were exact. |

## Standalone real-weight probe

Each model supplied its code fixture's 25 normalized embedding rows and the
actual layer-0 packed weights, scales, and quantization biases. Every backend
read the same safetensors file. The probes used transposed weights, affine 4-bit
packing, group size 64, uint32 packed weights, and BF16 activations, scales and
biases. No tokenizer, cache, recurrent attention, or model iteration runs during
the standalone evaluation.

| First differing projection | Core 0.31.1 vs Swift | Core 0.32.2 vs Swift, 25 rows | Core 0.32.4 vs Swift, 25 rows | Core 0.32.4 vs Swift, 1 row |
| --- | ---: | ---: | ---: | ---: |
| Kestrel-4B-Coder QKV | exact | 72,602/204,800 changed, max 0.125 | 18/204,800 changed, max 0.03125 | 4,842/8,192 changed, max 0.0625 |
| DaisyRose-27B Z | exact | 46,667/153,600 changed, max 0.125 | 33/153,600 changed, max 0.03125 | 3,953/6,144 changed, max 0.0625 |
| Qwen3.8-27B Z | exact | 47,066/153,600 changed, max 0.125 | 23/153,600 changed, max 0.015625 | 3,974/6,144 changed, max 0.0625 |

For reproduction, generate the text receipt with
`Benchmarks/survey_reference.py`. Use `Benchmarks/affine_projection_probe.py capture
CHECKPOINT RECEIPT INPUT` once. Use the script's `evaluate INPUT OUTPUT` command in
each pinned Python environment. The native probe is the
`AffineProjectionProbeTests` filter after installing the debug metallib. Set
`MLXCAT_AFFINE_INPUT` to the same input tensor file and `MLXCAT_AFFINE_OUTPUT` to
the desired output file. Compare matching keys after conversion to FP32.
The Metal pipeline traces and tensor receipts are local.

## Metal dispatch and rounding

The first differing Swift calls are QKV and Z in
[`MLXVLM/Models/Qwen35.swift:576–577`](https://github.com/atlas-open-sources/mlx-swift-lm/blob/dcfa3aa157b7a7972bde0048eaa3ab5ab834ed37/Libraries/MLXVLM/Models/Qwen35.swift#L571-L579).
Both model implementations pass the same affine operands to MLX. Swift's
`QuantizedLinear.callAsFunction` forwards `weight`, `scales`, `biases`, group
size, bits and mode in
[`Source/MLXNN/Quantized.swift:367`](https://github.com/atlas-open-sources/mlx-swift/blob/ea227ae6323128b1b68310bb9424002b0bb0ee64/Source/MLXNN/Quantized.swift#L367).
Python mlx-lm calls QKV and Z separately in
[`mlx_lm/models/qwen3_5.py:143`](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3_5.py#L143)
through `QuantizedLinear.__call__` in
[`python/mlx/nn/layers/quantized.py:272`](https://github.com/ml-explore/mlx/blob/v0.32.4/python/mlx/nn/layers/quantized.py#L272).
The old-core match covered all 72 saved output matrices across both projections,
six row counts and two dtypes. The observed difference does not require
different bias tensors, group size, scales, or Swift projection fusion.

- **25 rows.** In the pin, the threshold leads to regular QMM and a single
  output from the full K reduction. The newer dispatch can split K for a
  non-batched transposed QMM, depending on the output tile count and whether K
  divides into aligned partitions. The 0.32.2 branch stored each split partial
  in the activation dtype before the reduction. The 0.32.4 branch stores
  FP32 partials, sums them, then casts to BF16. The changed partial precision
  explains why the survey baseline had many more rounded outputs than 0.32.4.
  [Pinned dispatch](https://github.com/atlas-open-sources/mlx/blob/5186c44bf7d10c07a4e4fc7838da881c6c031c33/mlx/backend/metal/quantized.cpp#L1315-L1339),
  [0.32.2 temporaries and reduction](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp#L1166-L1216),
  [0.32.4 temporaries and reduction](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L1180-L1277).
- **One row.** The older QMV implementation sums a four-element BF16 group
  before accumulating the bias contribution. The newer 4-bit loader promotes
  each term before that addition. At one row 0.32.2 agreed exactly with
  the pin for these projections, while 0.32.4 differed as shown above.
  [Pinned loader](https://github.com/atlas-open-sources/mlx/blob/5186c44bf7d10c07a4e4fc7838da881c6c031c33/mlx/backend/metal/kernels/quantized.h#L60-L69),
  [0.32.4 loader](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/quantized.h#L77-L84).

Metal pipeline IDs confirmed `affine_qmm_t` at 25 rows on 0.31.1 and
`affine_qmm_t_splitk` on both newer versions for all three projections.
0.32.2 followed it with `col_reduce_small_1_reduce_sumbfloat16`; 0.32.4
followed it with `steel_gemm_splitk_accum_bfloat16_float32`. On the M4 Pro
at eight rows the pin used `affine_qmv_fast` and both new cores used
`affine_qmv_wide`. This branch is architecture-sensitive.
[0.32.4 QMV dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L1840-L1866).

Two controls isolate the split-K contribution for the 25-row fixtures.

- Duplicating the operands along the batch dimension forces the new core's
  ordinary batched QMM. Its first result is exact against the pin's single
  QMM across all three projections.
- On the pin, splitting the matching 4-bit weights, BF16 scales and biases,
  and input evenly into two aligned K partitions, doing two QMMs, then
  summing their BF16 outputs reproduced all 512,000 elements of the 0.32.2
  result exactly. That construction still differs from 0.32.4. Using the
  old QMV for each row or casting all operands to FP32 before the old QMM
  also failed to reproduce 0.32.4 (maximum delta 0.125 across these probes).

The results call for the 0.32.4 Metal behavior or a backport of its kernels
and dispatch. A core move needs a separate parity and performance run.

## Tokenizer alias

DaisyRose now loads from the original pack through the native tokenizer
loader. A native HTTP completion from that pack returned 200 and emitted
eight tokens. Only its `Qwen3_5Tokenizer` class name is normalized in memory to
`Qwen2Tokenizer`. The loader reads the original vocabulary and chat
template and keeps strict class validation. The tokenizer tests exercise
the Qwen BPE merge and template and rejection of an unknown class.

Both Inkling checkpoints passed their full reference gate on this build.
Each matched 760/760 greedy IDs and 960/960 sampled logits with zero
absolute logit error. XCTest reported 571 cases with zero failures and
68 opt-in skips. All 10 Swift Testing cases passed.
