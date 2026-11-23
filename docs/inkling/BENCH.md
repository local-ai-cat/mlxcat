# Inkling 2 Mini and Qwen3.8 coding benchmark

T1 is complete on release build `a2059f1`. The T2–T6 cells are running on
release build `45d1209`, with the same frozen server for both models in
each task. The transport stub is excluded from the measurements below.

## Protocol

Run one model at a time on the M4 Pro under the machine's exclusive lease.
Use the same release build of `mlxcat-http` for both models in each task. Run each of
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

| Task | Independent acceptance |
| --- | --- |
| T1 | Fix the Swift off-by-one in one implementation file and pass the complete test suite. |
| T2 | Add the documented Python label counter, pass pytest, and handle duplicate and blank labels. |
| T3 | Rename the Swift type across sources and tests, build the package, and pass the affected test. |
| T4 | Retrieve three facts from the operations file and leave the checkout clean. |
| T5 | Replace three deprecated call sites, retain the wrapper, and build the Swift package. |
| T6 | Add the CLI uppercase flag with tests and documentation while preserving the default output. |

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

Each Qwen repeat reported 9,216 cached prompt tokens across its six turns.
Inkling reported zero and processed 16,933–38,312 prompt tokens across
eight to ten turns. The timings include each model's native cache behavior
and the conversation length produced by its tool choices.

## Three-repeat T2 comparison on the fused release

Both models passed the independent Python checks on release build `45d1209`.
Values show the median and observed range.

| Model | Passed | Wall s | Decode tok/s | TTFT s | Peak GiB | Tool calls | Failed calls | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling 2 Mini MLX 4/8-bit | 3/3 | 436.68 [306.08–446.57] | 15.64 [15.62–15.65] | 17.83 [7.84–28.01] | 25.26 [24.96–25.39] | 14 [10–14] | 4 [3–5] | 14 [11–15] |
| Qwen3.8-27B-4bit | 3/3 | 278.87 [275.20–327.54] | 13.07 [12.93–13.24] | 4.48 [3.53–5.80] | 21.80 [21.30–23.14] | 15 [14–16] | 0 [0–0] | 12 [10–13] |

## Three-repeat T3 comparison on the fused release

Inkling found the four files in every repeat. Each run tried two Bash calls
that the fixed auto-edit policy declined, and the checkout remained untouched.
Qwen completed two renames. Its failed checkout contained duplicate
`InventoryItem` declarations.

| Model | Passed | Wall s | Decode tok/s | TTFT s | Peak GiB | Tool calls | Failed calls | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling 2 Mini MLX 4/8-bit | 0/3 | 160.47 [138.18–181.99] | 16.83 [16.79–16.84] | 10.15 [10.04–10.48] | 24.39 [24.34–24.79] | 12 [11–13] | 2 [2–2] | 9 [8–10] |
| Qwen3.8-27B-4bit | 2/3 | 401.00 [376.90–589.30] | 13.02 [12.93–13.22] | 5.21 [4.68–6.22] | 22.59 [21.11–22.84] | 17 [17–30] | 2 [1–3] | 9 [9–17] |

## Three-repeat T4 comparison on the fused release

Both models answered all three questions and left the checkouts clean.
Inkling used one read per repeat and the median end-to-end time was 43.96 s.
Qwen's median end-to-end time was 83.83 s.

| Model | Passed | Wall s | Decode tok/s | TTFT s | Peak GiB | Tool calls | Failed calls | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling 2 Mini MLX 4/8-bit | 3/3 | 43.96 [43.78–44.32] | 16.14 [16.08–16.26] | 16.61 [16.45–18.12] | 24.65 [24.65–24.65] | 1 [1–1] | 0 [0–0] | 2 [2–2] |
| Qwen3.8-27B-4bit | 3/3 | 83.83 [82.78–108.58] | 13.06 [13.05–13.11] | 25.28 [4.56–29.97] | 21.28 [19.76–21.58] | 2 [1–2] | 0 [0–0] | 3 [2–3] |

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

## Two conversations on one loaded Inkling model

Send two warm 32-token requests separately, then submit both simultaneously
to one release server at `45d1209`. The native scheduler gives both requests
independent caches and runs them serially. Each prompt produced identical
greedy tokens in the solo and concurrent measurements.

| Request | Solo TTFT | Simultaneous submission TTFT | Solo wall | Simultaneous submission wall |
| --- | ---: | ---: | ---: | ---: |
| Saturn, served second | 0.284 s | 2.208 s | 1.938 s | 3.862 s |
| Jupiter, served first | 0.285 s | 0.287 s | 1.940 s | 2.209 s |

## Tool outcomes in the archived matrix

The T2 and T3 event streams from `matrix-45d1209` separate the old combined
failure column into three outcomes. The same independent acceptance checks
determine done status.

| Model | Task | Done | Declined | Malformed | Tool errors |
| --- | --- | ---: | ---: | ---: | ---: |
| Inkling | T2 | 3/3 | 12 | 0 | 0 |
| Inkling | T3 | 0/3 | 6 | 0 | 0 |
| Qwen | T2 | 3/3 | 0 | 0 | 0 |
| Qwen | T3 | 2/3 | 0 | 0 | 6 |

All 18 Inkling outcomes were policy declines of Bash calls. In T3 the
declined `mv` blocked the actual rename. The six Qwen T3 tool errors came
from edit calls. The T1 Inkling runs on `a2059f1` had seven Bash declines
and passed 3/3. Both models passed 3/3 in T4 with no tool failures. The
stopped T5 cell had zero completed repeats.

The paired rerun uses the same frozen `45d1209` binary and fresh clones of the
same fixture. Whisker yolo mode and the gateway tool whitelist permit Bash,
with filesystem writes confined to each run and Whisker's transient atomic
write files, and network access confined to the loopback gateway. The sandbox
harness archives raw tool events and per-turn usage per repeat. A decline
takes precedence in classification. A rejected invocation with malformed
arguments counts as malformed. Other failed tool events count as tool errors.

The corrected matrix is `matrix-bash-sandbox-45d1209`. The sandbox smoke
tests exposed two infrastructure issues before this matrix began. The shell
needed a run-local `TMPPREFIX` for heredocs, and SwiftPM needed to use its
native driver and skip its inner sandbox while still under the outer
filesystem sandbox. The earlier smoke artifacts are preserved separately.
The final tool metric also compares the stream's model-specific tool frame
markers with Whisker's invocation events to count frames dropped before
execution. A nonzero Bash exit receipt is a tool error even when Whisker
reports the event as completed.

| Corrected cell | Done | Median wall s | Median decode tok/s | Median-turn TTFT s | Peak GiB | Declined | Malformed | Tool errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling T2 | 3/3 | 135.88 | 15.94 | 8.15 | 25.10 | 0 | 0 | 0 |
| Qwen T2 | 3/3 | 176.52 | 13.19 | 4.12 | 20.37 | 0 | 0 | 2 |
| Inkling T3 | 3/3 | 351.07 | 15.87 | 19.35 | 25.12 | 0 | 0 | 5 |
| Qwen T3 | 3/3 | 192.81 | 13.31 | 6.46 | 20.81 | 0 | 0 | 0 |

The outcome columns are totals across the three repeats. Both executed Qwen
errors in the corrected T2 cell were configuration-file searches where
`grep` returned 1 for no matches. All three edits and independent acceptance
checks passed.

The corrected T3 paired reruns completed six of six checkouts. Inkling
recovered from five nonzero shell exits, including unquoted glob syntax
in its grep invocations. The independent acceptance check confirmed the
rename and the affected Swift test in all six checkouts.

## Native prefill and decode A/B

The release arms ran on the M4 Pro with 48 GiB of RAM and macOS 27.2.1.
Each arm started a fresh server, warmed up with a short greeting, then ran
the same 3,541-token chat and 3,558-token follow-up at temperature 0. The
four arms used one binary and changed one setting at a time. Both 16-token
outputs matched the control in every arm.

| Arm | Cold TTFT s | Cold prompt tok/s | Follow-up TTFT s | Cached follow-up tokens | Cold peak GiB | Follow-up peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 14.57 | 242.98 | 13.93 | 0 | 24.37 | 24.37 |
| Exact-boundary prefix reuse | 13.87 | 255.29 | 2.48 | 3,072 | 24.45 | 23.35 |
| Packed grouped NVFP4 | 15.17 | 233.41 | 15.06 | 0 | 23.04 | 23.00 |
| Mixed-storage vocabulary projection | 14.08 | 251.47 | 13.97 | 0 | 23.87 | 23.52 |

The prefix rematerialized 486 tokens on the follow-up. Its checkpoint
contains each visit's attention KV and all four convolution histories at
token 3,072. Packed grouped prefill reduced the cold peak by 1.32 GiB. It
took 0.60 s more than the staged grouped multiply on the cold prompt.

The control and vocabulary arms also ran a separate 256-token coding
completion. The mixed-storage kernel read the original BF16 vocabulary
table in the pinned FP32 GEMV reduction order. All 256 token IDs matched.
The server reported 18.81 to 19.98 tok/s on that completion. The measured
peak physical footprint went from 20.35 to 19.41 GiB.

Raw arm JSON, server logs and memory-pressure receipts are in
`native-prefix-packed-ab-1` under the shared Inkling work directory.

The packed path is the default for eligible prefill after all 760 greedy
tokens and 960 sampled logits matched the netprojects fixtures. The lower
peak is the reason for choosing it despite the cold latency increase on
this prompt. Set `MLXCAT_INKLING_PACKED_PREFILL=0` for staged prefill.
