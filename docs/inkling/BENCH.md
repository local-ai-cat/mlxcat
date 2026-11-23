# Inkling 2 Mini and Qwen3.8 coding benchmark

The full three-repeat comparison is in progress. The transport stub is
excluded from the measurements below.

## Protocol

Run one model at a time on the M4 Pro under the machine's exclusive lease.
Use the same release build of `mlxcat-http` for both models. Run each of
T1–T6 three times from the same frozen fixture commit `ba6b87d`, with a fresh
checkout and Whisker thread per run. Use the Whisker harness in auto-edit mode,
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
