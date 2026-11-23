# Inkling 2 Mini parity

The first full-weight Baytizzel run failed the token and logit gate. Whisker results are pending.

## Acceptance

- Verify every shard listed in the checkpoint index before loading it. Check the safetensors header, file size, indexed tensor names and total payload size. Verify the SHA-256 manifest when the checkpoint supplies one.
- Run mlx-vlm in the shared uv venv and the native path under the same machine lease, one model at a time. Record the build revisions, checkpoint, seed, prompt token IDs, prefill chunk size and decoded token IDs.
- Require equality for each of the first 64 greedy token IDs, or through EOS if it arrives earlier. Compare native FP32 logits at the reference fixture's token indices with a provisional maximum absolute error of 0.25 logits. Report actual maximum and median errors, and the first failing prompt and position. This initial tolerance is an acceptance target and has not been measured.
- Compare text and image token preparation separately. For tools and reasoning, also verify the streamed HTTP events and the assembled non-streamed response.
- Compare solo versus batched generation and cold versus warm prefix-cache continuation separately from reference parity.

## Current checkpoint verification

Both indexes report four shards and 19,525,880,832 tensor payload bytes. Before the first full-weight Swift run, the Baytizzel checkpoint passed its file sizes, tensor index and SHA-256 manifest checks.

## Local test entry point

`InklingParityTests.testGreedyTokensAndReferenceLogits` accepts
`MLXCAT_INKLING_VERIFIED_MODEL` and `MLXCAT_INKLING_REFERENCE`. Run it under the
machine's exclusive lease after verification. It uses the fixture's expanded
token IDs and attention chunk size, including the reference image for the image
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
