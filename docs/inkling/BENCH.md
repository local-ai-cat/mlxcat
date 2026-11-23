# Inkling 2 Mini and Qwen3.8 coding benchmark

No real Whisker measurements have been recorded yet. The local harness passed
a transport smoke using a deterministic stub. Its results are excluded below.

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
