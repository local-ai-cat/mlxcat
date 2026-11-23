# Inkling 2 Mini and Qwen3.8 coding benchmark

The latest complete paired results below are from release build `a2059f1`.
The fused unpack release is under verification. The transport stub is
excluded from the measurements below.

## Protocol

Run one model at a time on the M4 Pro under the machine's exclusive lease.
Use the same release build of `mlxcat-http` for both models. Run each of
T1–T6 three times from the same frozen fixture commit `ba6b87d`, with a fresh
checkout and Whisker thread per run. Use a frozen Whisker 0.1.0 binary in auto-edit mode,
greedy sampling with seed 42, a shared advertised 32,768-token context limit,
and at most 30 model rounds. Evaluate acceptance independently of the answer
the model reports.

Wall time covers the entire purr turn. Report median per-turn server TTFT for
each run, and divide generated tokens after the first token of each turn by
the summed server decode time. Use the peak sampled server physical footprint
from `proc_pid_rusage` at 50 ms intervals. Count completed, unsuccessful and
declined tool calls from the purr event log, and count model rounds from server
usage events. Summarize repeat medians with ranges and task acceptance counts.

| Model | T1–T6 acceptance | Wall | Decode tok/s | TTFT | Peak footprint | Tool calls | Failed tool calls | Rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Inkling 2 Mini MLX 4/8-bit | Pending | | | | | | | |
| Qwen3.8-27B-4bit | Pending | | | | | | | |

## KV memory at the advertised context limit

The Mini schedule contains eight global attention visits and 40 sliding
visits. Eight global BF16 K/V pairs at 8 heads × 128 dimensions consume
32 GiB at 1,048,576 tokens. The 1,024-token sliding windows add approximately
160 MiB. The target shards contain 19.53 GB of tensor data. Together these
exceed the M4's 48 GiB before temporary memory, system memory and allocator
overhead. This is a size calculation. Measurements and optimisation results
are pending.

## Relative attention mask

Measured on the M4 Pro under the exclusive lease, with warm Metal pipelines
and 20 synchronous evaluations per arm. A/B/B/A used one synthetic BF16
global-attention visit with 128 queries, 8,192 keys, 24 heads and relative
extent 1,024. The peak column is incremental MLX allocation above the retained
inputs. These are operation measurements.

| Mask | Mean time across the two arms | Incremental peak |
| --- | ---: | ---: |
| Composed matmul, gather and concatenate | 4.90 ms | 112.05 MiB |
| Fused Metal mask and sink | 2.53 ms | 48.02 MiB |

The fused mask follows mlx-vlm's shape-generic Metal band kernel in
`mlx_vlm/models/inkling/language.py`. It also writes the query-dependent sink
in the output allocation. The CPU implementation and the strided-input
differential test retain the composed mask. GPU inference uses the fused
kernel. End-to-end decode and TTFT impact remain to be measured.

## Vocabulary projection

On the same M4 Pro under the exclusive lease, a warmed synthetic single-token
FP32 projection used Mini's 201,024 × 2,304 vocabulary matrix. Each A/B/B/A
arm evaluated 12 steps.

| Projection | Mean time across the two arms |
| --- | ---: |
| Convert BF16 weights on every step | 21.94 ms |
| Reuse the converted FP32 weights | 8.40 ms |

This follows the decode projection cache in mlx-vlm
`models/inkling2/language.py`. The prototype probe kept the BF16 matrix alive
in both arms and peaked near 2.59 GiB. The Swift model stores the converted FP32
projection and releases the raw checkpoint dictionary after load. The
retained weight representation is about 0.86 GiB larger than BF16. Model-level
resident memory and decode speed remain to be measured.

## Native HTTP functional smoke

On the verified netprojects checkpoint, a 24-token greeting finished in 82
generated tokens and streamed a friendly answer. Timings below came from
`mlxcat-http` usage. Decode rate excludes the first generated token.

| Request | TTFT | Decode tok/s |
| --- | ---: | ---: |
| Cold greeting including model load | 4.870 s | 18.23 |
| Warm greeting | 0.287 s | 18.95 |
| Weather tool invocation | 1.353 s | 18.72 |
| Reference image | 1.334 s | 18.76 |

The first Whisker smoke hit the server's default 16-token completion cap
midway through a tool frame. The coding benchmark proxy sets an explicit
2,048-token cap and supplies the same six local file tools to both models.
The assessment harness runs acceptance tests outside the model loop.

## First completed Whisker repeat

Both models passed the independent T1 Swift tests with release build
`70747f1`. T1 fixes an off-by-one in the counting implementation. These
measurements describe one repeat per model.

| Model | Wall | Decode tok/s | Median TTFT | Peak footprint | Tool calls | Failed tool calls | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling 2 Mini MLX 4/8-bit | 126.16 s | 16.80 | 7.06 s | 23.44 GiB | 11 | 1 | 7 |
| Qwen3.8-27B-4bit | 214.68 s | 12.15 | 6.90 s | 21.68 GiB | 11 | 0 | 6 |

## Three-repeat T1 comparison at `60122de`

Both models passed all three independent T1 checks. Values show the median
and observed range. The later repeated run includes a warm server between
repeats and uses a fresh checkout and Whisker thread for every repeat.

| Model | Passed | Wall s | Decode tok/s | TTFT s | Peak GiB | Tool calls | Failed calls | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling 2 Mini MLX 4/8-bit | 3/3 | 204.60 [144.93–290.19] | 15.55 [15.28–16.09] | 8.45 [7.61–10.38] | 23.80 [23.75–24.82] | 14 [12–15] | 3 [2–3] | 11 [8–11] |
| Qwen3.8-27B-4bit | 3/3 | 124.29 [117.00–135.64] | 13.21 [13.13–13.32] | 5.38 [5.25–5.73] | 20.51 [19.73–20.52] | 10 [10–10] | 0 [0–0] | 6 [6–6] |

## Three-repeat T1 comparison on the verified release

Both models passed all T1 repeats on release build `a2059f1`. Values show
the median and observed range. The remaining task cells are running.

| Model | Passed | Wall s | Decode tok/s | TTFT s | Peak GiB | Tool calls | Failed calls | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling 2 Mini MLX 4/8-bit | 3/3 | 412.67 [225.02–434.76] | 16.09 [16.07–16.92] | 21.64 [19.43–22.01] | 26.96 [26.14–26.97] | 14 [11–15] | 2 [2–3] | 9 [8–10] |
| Qwen3.8-27B-4bit | 3/3 | 140.42 [129.90–143.54] | 13.10 [13.06–13.39] | 5.55 [5.39–6.56] | 20.48 [19.75–20.52] | 11 [10–11] | 0 [0–0] | 6 [6–6] |

## Grouped NVFP4 prefill cost

Run the 111-token tool fixture over loopback in A/B/B/A order. Each arm starts
a fresh release server and completes a short greeting before timing the tool
request. TTFT and decode rate come from server usage. Peak physical footprint
is sampled during the measured request. Values are means across the two arms.

| Release | Warm TTFT | Decode tok/s | Peak footprint |
| --- | ---: | ---: | ---: |
| `6b7edf1` with FP32 postscale | 0.579 s | 19.10 | 19.92 GiB |
| `a2059f1` with reference grouped scaling | 5.455 s | 19.13 | 22.46 GiB |

The pinned-core compatibility path adds prefill time and temporary memory
when the routes select grouped matrix multiplication. Both arms emitted the
59-token tool continuation. The native fixture verifier measured maximum
logit error of 1.018 before the grouped fix and zero after it.

## Fused NVFP4 unpacking

Compare the two grouped implementations in a standalone Swift probe. Each
arm warms Metal with a short prompt, then evaluates first-token logits for the
111-token tool fixture. A/B/B/A used the same verified checkpoint and cache
state. Peak is MLX allocation after resetting its peak counter for each arm.
Both implementations matched the reference's 59 greedy tokens and all
sampled logits exactly in this case.

| Grouped projection staging | Mean prefill time | Mean MLX peak |
| --- | ---: | ---: |
| FP32 unpack and scale, then BF16 conversion | 5.756 s | 24.36 GiB |
| Fused unpack and scale directly to BF16 | 1.906 s | 23.09 GiB |
