# Inkling 2 Mini parity

Status: pending. The native decoder visit schedule has passed its isolated Swift unit tests. No same-weight generation, logit comparison, or Whisker result has been recorded.

## Acceptance

- Verify every shard listed in the checkpoint index before loading it. Check the safetensors header, file size, indexed tensor names and total payload size. Verify the SHA-256 manifest when the checkpoint supplies one.
- Run mlx-vlm in the shared uv venv and the native path under the same machine lease, one model at a time. Record the build revisions, checkpoint, seed, prompt token IDs, prefill chunk size and decoded token IDs.
- Require equality for each of the first 64 greedy token IDs, or through EOS if it arrives earlier. Compare native FP32 logits at the reference fixture's token indices with a provisional maximum absolute error of 0.25 logits. Report actual maximum and median errors, and the first failing prompt and position. This initial tolerance is an acceptance target and has not been measured.
- Compare text and image token preparation separately. For tools and reasoning, also verify the streamed HTTP events and the assembled non-streamed response.
- Compare solo versus batched generation and cold versus warm prefix-cache continuation separately from reference parity.

## Current checkpoint verification

Both indexes report four shards and 19,525,880,832 tensor payload bytes. At the first verification on the M4, all four indexed shards for each build were still missing. No weights were loaded.

## Local test entry point

`InklingParityTests.testGreedyTokensAndReferenceLogits` accepts
`MLXCAT_INKLING_VERIFIED_MODEL` and `MLXCAT_INKLING_REFERENCE`. Run it under the
machine's exclusive lease after verification. It uses the fixture's expanded
token IDs and attention chunk size, including the reference image for the image
case. It checks the exported top-k scores at all available positions
and stops comparing a case when the greedy token IDs diverge. The fixture script
allows at least 64 generated tokens per case.
