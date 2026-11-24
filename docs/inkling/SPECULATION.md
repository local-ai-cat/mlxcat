# Inkling greedy speculation

## The first difference

The eight-row prefill-style verifier and eight singleton forwards started
from copies of the same populated cache and received the same forced token
IDs. The first divergence on the 4/8-bit coding fixture was row zero of
visit zero, at the key projection in `InklingDecoderAttention.swift:118`.
Embedding and input normalization were bit-identical. The key projection
differed by `1.9073486e-6`. The query projection differed by as much as
`0.0078125` at the first visit. The pinned MLX Metal matmul dispatch in
`backend/metal/matmul.cpp:1270–1295` uses GEMV for the singleton and GEMM for
the panel. The coding fixture's final full-vocabulary error was 2.2311864.
The earlier full gate in [PARITY.md](PARITY.md) measured 9.253425.

Replacing each attention projection with its singleton counterpart restored
the first visit's projection bits. The next observed divergence was the
MoE output, including row six at the first visit (`7.6293945e-6`).
The router goes through FP32 `matmul` in `InklingMath.swift:33–39`, and the
expert gather starts sorting at 64 routes in `InklingExperts.swift:146–162`.
The packed NVFP4 GEMM has a further density condition in the projection.
Eight tokens at top eight give 64 routes, below that kernel's density
threshold on both release packs. The first observed drift therefore
cannot come from its BF16 tile rounding.

After restoring the singleton expert path, the first residual divergence
on row zero was SDPA at visit six (`0.00012207031`). The mask's logical
position is `query + keyCount - queryLength - key`
(`InklingAttention.swift:67`), with future and out-of-window positions
masked. The panel still shares a K/V span with its zero-value sink at the
end of that span. Singleton decode puts the sink immediately after each
row's visible keys. The pinned SDPA backend also selects its two-pass
reduction topology using query length
(`backend/metal/scaled_dot_product_attention.cpp:440–484`). Masking the
same logical tokens therefore does not guarantee identical accumulation.

A partial commit must preserve the completed prefix's attention and
convolution histories. `InklingAttentionState.append` keeps the last
`window - 1` keys and values for the next step. The convolution append
keeps `taps - 1` rows. Once an entire panel has advanced both histories,
rejected rows may have displaced data needed at the accepted boundary.
The lossless probe snapshots each target cache after advancing that
row with singleton math.

## Greedy acceptance

The first anchor comes from ordinary target prefill. The native drafter
reads the released `mtp.safetensors` and follows the three DSpark
attention blocks and their Markov recurrence. It takes pre-visit
features at visits 5, 23 and 41, and the final normalized feature at
48. See MTPLX `model.py:719–757`, `draft.py:76–125,223–270`, and
`runtime.py:576–618`. The probe projects the Markov correction in
hidden space before the vocabulary projection. This changes proposal
arithmetic order. Every proposal is held to the ordinary target's
greedy decision.

Starting from the committed cache, the probe constructs the target
forwards on a fork and reads their singleton argmaxes together. It
always commits the anchor. It accepts a proposal only when it equals
the corresponding target argmax, stops at the first rejection, and
keeps the snapshot after the anchor and matching proposals. The next
anchor is the exact target argmax at that accepted boundary. Thus the
committed cache and the emitted prefix advance by ordinary greedy
steps. The drafter only observes features from committed rows.
