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

## Control results

The singleton-arithmetic panel trial matched all 96 full-vocabulary
rows per release checkpoint bit for bit, along with every compared
final KV and convolution array and cache offset. Two additional
eight-row trials on the 4/8-bit release crossed offsets 1,024 and 8,192
from populated prefixes. Both matched all logits and cache state.
The batches that kept packed dense GEMV and unsorted vector expert
gathers while invoking SDPA per row also matched the code and long
context panels on both packs.

The strict DSpark control matched 760/760 greedy IDs on each release
pack. The timed pairs used the 4/8-bit release, 128 output tokens and
three DSpark proposals per round under an exclusive machine lease.
Rates exclude target and drafter prefill and include proposals,
verification and state commits.

| Input | Plain greedy | DSpark with singleton checks | Identical output IDs | Accepted proposals |
| --- | ---: | ---: | ---: | ---: |
| 32 tokens | 20.75–20.96 tok/s | 9.81–9.84 tok/s | 128/128 | 66/183 |
| 4,630 tokens | 18.04–18.05 tok/s | 8.04–8.24 tok/s | 128/128 | 68/179 |

The strict verifier groups the host readback and still submits each
target row's singleton GPU operations. It leaves native serving on
ordinary decode. In the standalone fixture tool, set
`MLXCAT_INKLING_PROBE_DRAFT=3` to reproduce the control, or use `0` for
plain timing. `MLXCAT_INKLING_PROFILE_PROMPT_FILE` supplies a benchmark
prompt and `MLXCAT_INKLING_DRAFT_STEPS` sets its output budget. Without
these two overrides, the tool compares the emitted IDs against the
selected fixture. The two verified draft archives had the same
SHA-256, matching the checksum shipped with the M5 Pro pack.
Raw benchmark and trace receipts are local and unpublished.

## Packed draft-round experiment

The packed verifier used batched dense GEMV with weights on the left
and token column vectors on the right. This avoids the pinned Metal
matmul's row-batch collapse (`backend/metal/matmul.cpp:1255–1265`).
It kept unsorted singleton-width gathered expert projections and used
an ordinary SDPA call for each row with that row's exact visible span
and sink position. The first commit strategy repaired the accepted
prefix after a rejection by running those rows again. The second saved
each row's K/V state and sliced its four convolution histories out of
the original convolution contexts. It reused the saved state at the
accepted boundary.

The scratch experiment ran the same two prompts under an exclusive
lease. Each packed arm matched 128/128 greedy IDs against the ordinary
control. Retaining the original row state improved on rerunning the
accepted prefix, but remained slower than the ordinary target.

| Input | Plain greedy | Packed with prefix rerun | Packed with row snapshots |
| --- | ---: | ---: | ---: |
| 32 tokens | 21.25 tok/s | 11.91 tok/s | 16.00 tok/s |
| 4,630 tokens | 18.10 tok/s | 9.72 tok/s | 12.26 tok/s |

The packed snapshot arm matched 760/760 greedy IDs on each release pack.
On the coding fixture for each pack, a separate checked run also compared
every drafted-row logit and each row's KV, convolution and offset
snapshot against ordinary singleton execution. Those checks passed.
This packed path stays in the scratch experiment because it lost both
throughput comparisons. The opt-in strict probe remains reproducible
from the branch.

The clean branch rerun passed the Swift suite (567 XCTest cases with
66 opt-in skips, plus 10 Swift Testing cases) and 28 benchmark
safeguard tests. The leaderboard and benchmark parity checks passed.
