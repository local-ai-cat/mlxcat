# Inkling 2 Mini parity

Both verified checkpoint builds passed the same-Mac Python fixture gate at
`a2059f1` and again with fused NVFP4 unpacking at `45d1209`. Coding results
are recorded in [BENCH.md](BENCH.md).

## Acceptance

- Verify every shard listed in the checkpoint index before loading it. Check the safetensors header, file size, indexed tensor names and total payload size. Verify the SHA-256 manifest when the checkpoint supplies one.
- Run mlx-vlm in the shared uv venv and the native path under the same machine lease, one model at a time. Record the build revisions, checkpoint, seed, prompt token IDs, prefill chunk size and decoded token IDs.
- Require equality for each of the first 64 greedy token IDs, or through EOS if it arrives earlier. Compare native FP32 logits at the reference fixture's token indices with a maximum absolute error of 0.25 logits. Report actual maximum and median errors, and the first failing prompt and position.
- Compare text and image token preparation separately. For tools and reasoning, also verify the streamed HTTP events and the assembled non-streamed response.
- Compare solo versus batched generation and cold versus warm prefix-cache continuation separately from reference parity.

## Current checkpoint verification

Both indexes report four shards and 19,525,880,832 tensor payload bytes. Before the first full-weight Swift run, the Baytizzel checkpoint passed its file sizes, tensor index and SHA-256 manifest checks.

## Local test entry point

Build `mlxcat-inkling-parity` and stage the Metal library with
`scripts/build-metallib.sh .build/debug`. Set
`MLXCAT_INKLING_VERIFIED_MODEL` and `MLXCAT_INKLING_REFERENCE` and run
`.build/debug/mlxcat-inkling-parity` under the machine's exclusive lease.
`InklingParityTests.testGreedyTokensAndReferenceLogits` invokes the same
standalone verifier when the environment is set. The verifier checks the
formatted input token IDs and uses the fixture's attention chunk size,
including the reference image for the image
case. It checks the exported top-k scores at all available positions
and stops comparing a case when the greedy token IDs diverge. The fixture script
allows at least 64 generated tokens per case.

## First full-weight run

The M4 verified the baytizzel checkpoint's four indexed shards and all entries
in its SHA-256 manifest. The lane generated the reference locally with
mlx-vlm `111ac668` and MLX 0.32.4. Swift used the pinned mlxcat dependencies
at branch commit `98c17fd`. All 12 native prompt token sequences matched.
The 0.25-logit gate failed. Scores below are from the first run, before
correcting the native image row order.

| Case | Matching greedy prefix | Maximum absolute top-k logit error | Median |
| --- | ---: | ---: | ---: |
| Code | 39 / 64 | 0.457 | 0.145 |
| Fact | 40 / 64 | 1.200 | 0.179 |
| Image | 61 / 61 | 1.648 | 0.137 |
| Literal | 55 / 64 | 1.038 | 0.250 |
| Long context | 64 / 64 | 3.421 | 0.429 |
| Multi-turn | 29 / 64 | 2.127 | 0.179 |
| Reasoning | 22 / 64 | 0.758 | 0.058 |
| Short chat | 64 / 64 | 1.500 | 0.335 |
| Structured | 64 / 64 | 3.615 | 0.143 |
| System | 30 / 64 | 1.946 | 0.156 |
| Tool call | 59 / 59 | 1.446 | 0.154 |
| Unicode | 64 / 64 | 0.830 | 0.128 |

A pixel inspection confirmed the first native run reversed image rows during
Core Image conversion. The corrected image path needs a fresh fixture run.
The text discrepancies still need investigation. The pinned Swift FP32
NVFP4 fallback and the mlx-vlm fused outer-scale path use different kernel
interfaces. A newer MLX comparison is in preparation.

The native release `mlxcat-http` served the verified Baytizzel model over
loopback. A cold 24-token greeting request streamed 32 reasoning tokens and
all 32 token IDs matched the Python fixture. Server TTFT, including load,
was 6.668 s. Excluding the first token, 31 tokens in the 1.727 s decode
interval give 17.95 tokens/s for this request. Its cold TTFT uses a different
measurement boundary from the warmed Python fixture.

The image row-order regression now passes with an asymmetric bitmap. The
full-weight image comparison remains pending after that correction.

## Netprojects follow-up

The netprojects checkpoint passed the four-shard size and index verification.
The Python lane independently verified the shard hashes and produced the same
input IDs, generated IDs and exported logits on both builds for all 12 cases.

With the row-order correction, the native netprojects image case matched all
61 generated IDs. Its maximum absolute top-k logit error was 0.608 and its
median was 0.084. Two scores exceeded the 0.25 gate. The release server also
returned a correct image answer and decoded a structured `get_weather`
invocation with the city argument `Lisbon`.

An isolated candidate used mlx-swift `00afc1d` with its newer MLX core.
On the code case, enabling the fused outer scale in that build gave the same
results as its FP32 fallback. Both matched 39 greedy tokens, with maximum
logit error 2.326. This probe did not justify changing the shipping pin.

## Netprojects run at `60122de`

The Swift residual attention now multiplies by the reciprocal square root
of the hidden width, in the same order as Python. The code-prompt prefill
trace matched every element entering visits 0–44. The first difference was
in physical layer 20's MXFP8 expert projections. The largest gate-projection
difference was 0.0078125. The final three visit inputs differed.

The complete run matched every fixture's formatted input token IDs. Two
cases matched the complete greedy continuation through EOS, and each case
had logits outside the 0.25 tolerance. Values are absolute differences over
the captured reference top-k entries through the first token mismatch.

| Case | Matching greedy prefix | Maximum | Median |
| --- | ---: | ---: | ---: |
| Code | 39 | 3.405 | 0.212 |
| Fact | 23 | 1.459 | 0.221 |
| Image | 61 (EOS) | 0.456 | 0.101 |
| Literal | 55 | 4.176 | 0.291 |
| Long context | 34 | 4.124 | 0.551 |
| Multi-turn | 29 | 1.300 | 0.213 |
| Reasoning | 18 | 0.723 | 0.055 |
| Short chat | 41 | 1.775 | 0.305 |
| Structured | 38 | 2.137 | 0.213 |
| System | 30 | 2.399 | 0.174 |
| Tool call | 59 (EOS) | 1.818 | 0.262 |
| Unicode | 7 | 0.979 | 0.169 |

## MLX 0.32.4 comparison

The isolated Swift probe with MLX 0.32.4 and the reference residual scale
matched all 64 code tokens and every sampled logit exactly. The result was
identical with BF16 or promoted FP32 input to the MXFP8 operator.

The pinned MLX 0.31.1 core uses a 512-element alignment test to select its
fast quantized gather vector kernel. MLX 0.32.4 derives the alignment from
the bit width, which selects that kernel at 256 elements for MXFP8. Inkling's
gate and up projections have a 2,304-element input, so the cores dispatch
different reduction kernels. The newer core also fails the current Metal 3
compile gate. The pinned core remains the branch default while the narrower
MXFP8 compatibility path is evaluated.

The full 0.32.4 probe matched all 12 greedy continuations through 64 tokens or
EOS. The 11 text cases had zero error in their sampled logits. The image case
matched all 61 tokens and had a maximum logit error of 2.481.

## Pinned core and vision arithmetic

The pinned-core standalone A/B matched all 64 code tokens in both arms.
Its maximum sampled logit error fell from 0.0725 to zero with zero-filled
MXFP8 columns. The release HTTP path also matched all 64 code tokens at
19.17 decode tokens/s. The separate XCTest process still diverged on code
with the same input IDs, including when fed a fresh array of fixture IDs.

The vision port matches mlx-vlm's fused linear bias, spatial-frequency
arithmetic and horizontal-first position interpolation. In the intermediate
trace the native pixels and every vision layer, including the adapter
output, matched the Python arrays element for element.

With those changes the pinned-core Swift probe matched all 12 greedy
continuations, including the 8,225-token prompt. Ten cases had zero sampled
logit error, including the image. Long context reached 3.471 maximum error
and the tool call reached 1.018. Both prompts reach the grouped matrix
dispatch during prefill. An unpadded MXFP8 control produced the same errors.
Disabling fused NVFP4 tensor scaling on MLX 0.32.4 also reproduced the tool
call error. Scaling the unpacked NVFP4 weights before BF16 rounding
eliminated both errors on the pinned core.

## Packaged verifier at `a2059f1`

The standalone verifier and `mlxcat-http` share the native model and
processor loader. The verifier checks the rendered prompt, then compares
greedy token IDs and the exported top-k logit samples. Standalone Metal
initialization resolved the earlier XCTest-host discrepancy.

| Checkpoint | Prompt IDs | Greedy output | Sampled logits | Maximum absolute logit error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Netprojects MLX 4/8-bit | 12/12 cases | 760/760 tokens | 960/960 exact | 0.0 | Pass |
| Baytizzel MLX M5Pro 48 GB | 12/12 cases | 760/760 tokens | 960/960 exact | 0.0 | Pass |

The opt-in XCTest wrapper also passed the code case. The ordinary mlxcat suite
reported 557 XCTest cases with 66 opt-in skips and zero failures, plus seven
passing Swift Testing cases.

The same 24 fixture cases passed again at `45d1209`, with identical input
IDs, greedy output IDs and sampled logits in every case. The ordinary suite
reported the same passing counts.

The NVFP4 unpacker and both custom attention and position kernels passed
compile-only Metal 3.1 checks with `-Werror` for macOS 15.2 and iOS 18.2.

## Prefill and vocabulary follow-up

The packaged Swift verifier ran against the same local mlx-vlm receipts at
`406dd58`. Greedy token IDs matched throughout each arm. All 960 sampled
FP32 scores per arm were identical to the reference, within the existing
0.25 absolute tolerance. Each arm also matched all 12 rendered prompt
token sequences.

| Checkpoint | Arm | Greedy match | Maximum sampled logit error |
| --- | --- | ---: | ---: |
| netprojects | Control | 760/760 | 0 |
| baytizzel | Control | 760/760 | 0 |
| netprojects | Packed grouped prefill | 760/760 | 0 |
| netprojects | Mixed-storage vocabulary | 760/760 | 0 |

The HTTP continuation A/B separately matched the greedy token IDs on both
turns while the warm arm restored 3,072 tokens of attention and convolution
state. The standalone 256-token vocabulary A/B also matched every token.

### Speculative verification gate

The native probe forked each fixture's populated cache, then compared
an eight-token verification panel with eight ordinary one-token forwards
on the same inputs. The 96 top-token decisions agreed. The maximum
full-vocabulary logit difference was 9.253425, and every panel exceeded
the 0.25 absolute tolerance. The current native panel therefore stays
disabled for Inkling speculation. MTPLX's row-stable attention, expert
and state-commit path is a separate implementation.

The updated package suite executed 560 XCTest cases with 66 opt-in
skips and zero failures. Seven Swift Testing cases also passed.
Both new Metal kernels compiled with warnings treated as errors for
macOS 14, macOS 15.2, and iOS 18.2 using Metal 3.1.

With packed prefill and mixed-storage vocabulary both enabled by default,
the verifier again matched 760/760 greedy tokens, 960/960 sampled scores
and 12/12 rendered prompts on each original checkpoint. Maximum sampled
logit error was zero on both. The combined-default release built
successfully and its package suite passed all 560 XCTest cases with
66 opt-in skips, plus seven Swift Testing cases.
