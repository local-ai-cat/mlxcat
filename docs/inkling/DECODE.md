# Inkling autoregressive decode on M4 Pro

The earlier warm MTPLX run measured 41.54 tokens/s on this M4 Pro with speculation disabled. A new run with Metal command counting measured 42.18 tokens/s by the runtime clock and 41.85 tokens/s between token callbacks, with a 4,630-token cached prompt. All 128 greedy IDs match the prior warm run on the verified MTPLX Bare Speed checkpoint. The mlxcat profile uses the verified netprojects NVFP4/MXFP8 release. The Bare Speed pack also uses 4-bit affine attention projections, embeddings and vocabulary. The models therefore move different amounts of weight data per token and the comparison does not establish how much speed a same-checkpoint port can recover.

## What one token does

| Difference | mlxcat | MTPLX 2.13.2 at `3890be556160af48887d5bfbf5304c4ad5b9417f` |
| --- | --- | --- |
| Runtime submission | The Swift probe builds a fresh lazy graph and submits it with `asyncEval` before reading the greedy ID. The pinned MLX runtime commits as it hits its operation or memory thresholds. | `mtplx/models/inkling/fast/engine_site.py:251–298` records Metal launches and checks the replay bitwise before using the native executor. `native/engine/executor.mm:1272–1310` encodes the recorded passes into one command buffer per chunk and signals completion through a semaphore. |
| Routed expert execution | `Sources/MLXCatNative/Inkling/InklingExperts.swift:120–148` selects the top eight of 128 experts and runs separate gathered quantized up, gate and down GEMVs per visit. Routing weights and the output reduction use FP32. | `mtplx/models/inkling/fast/experts.py:1437–1521` picks the own-kernel path for a singleton. It fuses input normalization and the router, performs top-eight selection and both gate/up dot products in one stage, then dispatches down projection. `experts.py:1523+` combines the weighted expert output, short convolution and residual. Its native NVFP4 path quantizes activations, which changes the arithmetic relative to mlxcat’s release reference. |
| Loop cache | `Sources/MLXCatNative/Inkling/InklingDecoderAttention.swift:15–24,91–108` appends chronological keys and values, slices the local window and appends a zero sink for attention. The per-visit cache also holds four short convolution histories. | `mtplx/models/inkling/fast/state.py:131–296` uses fixed local banks and growable global storage. It writes accepted KV rows and all four convolution histories together. `mtplx/models/inkling/fast/decoder.py:172–188` evaluates the replay outputs before committing those rows. |

For 128 warm native code-fixture steps, mean graph construction was 4.63 ms, time inside `asyncEval` was 38.55 ms and token readback took 5.25 ms. These are host intervals. GPU work can begin while `asyncEval` is still walking and submitting the graph. An eight-step diagnostic that forced evaluation at five boundaries across all 48 visits assigned 28.05 ms/token to attention and KV, 21.88 to MoE, 13.94 to the controller and input norm, 13.07 to attention convolution and MLP input, 7.07 to MLP convolution, and 4.25 to the final residual and vocabulary. The diagnostic inserted barriers and took 88.26 ms/token before sampling, so the stage figures show where work accumulates and cannot be summed to estimate production latency.

| Runtime and mode | Measured tokens after warm-up | Metal command-buffer commits/token | Public Metal wait calls/token |
| --- | ---: | ---: | ---: |
| mlxcat, default | 128 | 467.33 | 0 |
| mlxcat, 200 ops and 500 MB per buffer | 128 | 96.73 | 0 |
| mlxcat, MTPLX prompt at 4,630 tokens | 128 | 398.05 | 0 |
| MTPLX, ordinary decoding | 127 | 9.11 | 0 |

On the same 4,630-token prompt text, the direct native probe decoded at 16.99 tokens/s. Graph construction took 5.00 ms, submission 47.90 ms and readback 5.95 ms per token. This measures the native model without HTTP's graph overlap. The two packs produce different token sequences.

The Metal count wraps `commit`, `waitUntilCompleted`, and `waitUntilScheduled` on the concrete Metal command-buffer class in the process. It includes uploads and cache work committed within the marked interval. A second capture also wrapped Metal shared-event waits and main-thread dispatch semaphore waits. mlxcat recorded exactly one shared-event wait per token on a worker thread (5.34 ms total per token), with no main-thread semaphore waits. MTPLX recorded 2.06 shared-event waits (0.57 ms total) and 10.11 main-thread semaphore calls (21.58 ms total) per token. Counts include calls that return immediately.

The native probe reads one greedy ID and pinned MLX waits through its event path (`transforms.cpp:296–325`, `backend/metal/event.cpp:23–28`). MTPLX also waits on native engine semaphore completion (`executor.mm:1682–1720`), evaluates the single-token logits and cache deltas (`fast/decoder.py:172–188`), and evaluates the greedy reduction and finite-check together once before reading both scalars (`mtplx/generation.py:7992–8022`).

## Isolated exact-greedy trials

| Change | Baseline tok/s | Trial tok/s | Greedy |
| --- | --- | --- | --- |
| Increase the pinned MLX command-buffer budget | 20.19–20.37 | 20.59–20.66 | 128/128 in each trial |
| Fuse the gate activation after the separate quantized GEMVs | 20.19–20.37 | 20.49–20.58 | 128/128 in each trial |
| Fuse controller pointwise math after the existing projections | 19.92–20.39 | 19.92–20.36 | 128/128 in each trial |
| Append the KV row and the sink in a single concatenation per tensor | 20.22–20.32 | 20.24–20.30 | 128/128 in each trial |
| Reserve global KV rows at a short context | 20.23–20.34 | 20.22–20.28 | 128/128 in each trial |
| Reserve global KV rows at an 8,225-token context | 16.34–16.35 | 17.74–17.86 | 128/128 in each trial |

The buffer budget removed about 79% of Metal commits and moved time from submission into final readback. The gain was under 2.4% and the knobs apply to the entire MLX process. The gate trial covered just the pointwise activation with separate GEMVs, so it tests one small part of MTPLX’s fused path. Production retains the prior buffer budget, gate path and sink concatenation for appends.

The global KV capacity trial gained about 8.9% at an 8,225-token prompt. The change uses amortized growth from MTPLX's global bank design. From offset 8,192, singleton decode writes the new row into a buffer that grows in 256-row increments and uses the following untouched row as the sink. The published prefix cache keeps only the active rows. Attention reads chronological keys and values. Local visits and short contexts use the existing append path.

## Adopted HTTP decode change

The scheduler now builds the next eligible singleton Inkling graph while the GPU finishes the pending token. Building on shallow copies lets the scheduler discard a step at EOS and export a snapshot at the last emitted token. The approach follows the overlap used by mlx-swift-lm's `TokenIterator` and retains the Swift scheduler's existing sampling eligibility checks.

The release HTTP A/B used temperature 0, no seed, a 32-token coding prompt and a 256-token output. After adding the pending-cache snapshot safeguard, repeated controls ran at 20.18–20.19 tokens/s and repeated copy-prebuild runs at 21.93–22.03 tokens/s, about 8.9% faster by the paired medians. Both arms matched all 256 output IDs. Each cached follow-up reused 3,072 tokens and matched the control output. Physical peak on the coding request was 21.32–21.46 GiB in controls and 21.75 GiB with the optimization. The comparison used `MLXCAT_INKLING_COPY_PREBUILD=0` for the controls.

An 8,225-token HTTP A/B enabled graph overlap in both arms and varied only `MLXCAT_INKLING_RESERVED_GLOBAL_KV`. The runs finished at natural EOS after 85 tokens. Controls ran at 17.05–17.36 tokens/s and peaked at 24.66–24.70 GiB. Reserved KV ran at 18.32–18.54 tokens/s and peaked at 22.94 GiB. Every generated ID matched across all four runs.

The reference probe with reserved KV enabled matched 760/760 greedy IDs and all 960 sampled logits for each of the netprojects and baytizzel checkpoints. The maximum observed absolute logit error was 0.0, including each 8,225-token fixture.

The raw benchmark receipts were retained locally and are unpublished. Each measured run held the exclusive machine lease.

## Paired NVFP4 projection trial

One shared gathered matrix-vector multiply projected the same routed inputs
through packed up and gate weights. It kept the pinned core's FP32
accumulation and applied the original per-projection scales and BF16
rounding separately. This tested a larger part of MTPLX's gate/up fusion
without adopting its activation quantization.

On the short code fixture, controls ran at 20.24–20.33 tok/s and the paired
projection ran at 20.58–20.60 tok/s. Each run matched all 128 greedy IDs.
The trial retained another packed copy for decode alongside the prefill
weights. A gain of about 1.5% did not justify that memory cost. The trial is
recorded in `overnight-paired-expert-ab` and was removed from the engine.

## Same-checkpoint MTPLX ablation

The MTPLX Bare Speed ablation used the same verified weights, 4,630 cached
prompt tokens and 128 generated tokens for every arm. All five arms
produced identical greedy output.

| MTPLX autoregressive arm | Decode tok/s |
| --- | ---: |
| Default, bookended controls | 41.20, 41.23 |
| Metal replay off, same fast stages | 29.24 |
| Expert stage E off, replay and stages A/BC/D enabled | 33.12 |
| Entire fast decoder off | 23.57 |

`mtplx/engine/runtime.py:57–68` controls replay availability.
`mtplx/models/inkling/fast/decoder.py:38–56,107–117` selects the fast route
and stage set. `fast/step.py:637–655,677–687` binds the expert stage or
its reference implementation. The last arm changes both execution and
cache route. These cuts are independent comparisons with the default and
their deltas cannot be added.

Replay was the largest isolated contributor on this pack. Changes limited
to Swift graph construction or to MLX buffer size do not reproduce this
execution path. MTPLX records launch bindings through its compute-encoder
hooks (`native/engine/recorder.mm:1–39`), then its executor reconstructs
command buffers with the recorded barriers and binds live inputs. The
existing mlxcat schedule still evaluates an MLX graph for each token.
Receipts are in `overnight-mtplx-ab`.

## Compiled singleton expert graph

mlxcat now keeps a compiled MLX graph for the routed expert stage at the
singleton decode shape. It compiles the existing router, gathered projections,
activation and reduction, preserving the checkpoint's accumulation and
rounding. This reuses graph work at a fixed shape and leaves Metal submission
to MLX.

The direct probe improved from 20.16–20.31 to 21.02–21.06 tok/s, with all 128
greedy IDs matching. With graph overlap enabled in every HTTP arm, the
32-token prompt and 256-token completion improved from 21.96–22.06 to
22.71–22.76 tok/s, about 3.3%. All 256 IDs and the cached follow-up matched.
TTFT was 0.33 s in each arm. Peak physical footprint ranged from 21.70 to
21.81 GiB across the four arms. Both original checkpoints matched 760/760
greedy IDs and 960/960 sampled logits per checkpoint with a maximum error
of 0.0 in the weighted verifier.

Set `MLXCAT_INKLING_COMPILED_EXPERTS=0` for the comparison path. Receipts
are in `overnight-compiled-expert-ab`, `overnight-compiled-http` and
`overnight-compiled-Inkling-*.txt`.

On the same 4,630-token prompt used by MTPLX, an HTTP pair improved from
18.56 to 19.05 tok/s with all 128 IDs matching between the native arms.
Peak was 24.32–24.33 GiB. This is a matched prompt and output budget across
runtimes, with different checkpoint layouts and with MTPLX's prompt restored
from cache. The turn's prefill is excluded from decode throughput. The
remaining gap is still roughly 2.2×. Receipt: `overnight-matched-http`.
