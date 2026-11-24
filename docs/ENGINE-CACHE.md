# Engine cache and two-chat measurements

The native scheduler already supports cross-turn text prefix reuse and
continuously admits compatible rows into one loaded model. These checks used
one loaded model per server, temperature zero and two independent cache
session keys. The local benchmark receipts are unpublished.

| Model | Identical follow-up, fresh session TTFT | Reused-session TTFT | Cached tokens | Exact IDs |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B-4bit | 27.17 s | 1.72 s | 3,072 | Yes |
| Kestrel-4B-Coder-4bit | 4.18 s | 0.25 s | 3,072 | Yes |

The fresh-session control kept the same rendered conversation, prefill path
and sampling settings. Disabling hybrid reuse was unsuitable as a token
control for Qwen3.8 because that switch also changes its cold VLM prefill.

Rolling caches now have a validated restore path for text when
`MLXCAT_ROLLING_PREFIX_REUSE=1`. It preserves the cursor, capacity and
complete layer state at existing prefill boundaries, following oMLX's
exact hybrid snapshot approach. Mixed full-attention and rotating layers
are captured together. The rolling path is opt-in pending model-backed
qualification. Quantized caches and media prompts keep their prior
eligibility restrictions. Both measured checkpoints use Qwen3.5-family
hybrid state, so these two runs do not qualify every family.

| Model | Two requests, width 1 | Two requests, width 2 | Aggregate tok/s, width 1 → 2 | Exact IDs |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B-4bit | 15.69 s | 8.69 s | 12.24 → 22.08 | Yes |
| Kestrel-4B-Coder-4bit | 2.84 s | 1.70 s | 67.49 → 113.21 | Yes |

Both requests arrived concurrently in both arms. The width-one arm forced
queueing with `--max-concurrent-requests 1`. Width two exercised the
existing scheduler and per-row positional state. As with vLLM-MLX's
continuous admission, the scheduler can join compatible requests while
another is generating. The existing per-family and multimodal restrictions
remain necessary. No scheduler default changed for these measurements.

## SSD restart comparison

Set `MLXCAT_PREFIX_DISK_DIRECTORY` to opt in to persisted named text
sessions. The native session store writes versioned safetensor snapshots
with recurrent checkpoints and restores them on an in-memory miss. Model
file identity and session identity scope the cache. As in oMLX, writes
are queued after the snapshot, with a bounded per-model budget. Buffer
materialization still happens before the background write. Media and
anonymous sessions do not take this path.

| Model | Cold restart TTFT | SSD restart TTFT | Restored tokens | Exact IDs |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B-4bit | 27.16 s | 1.95 s | 3,072 | Yes |
| Kestrel-4B-Coder-4bit | 4.19 s | 0.29 s | 3,072 | Yes |

Both arms used a new process, a short warm-up and the identical follow-up.
The SSD control was disabled for the cold arm. The versioned cache is
optional because it persists conversation token IDs and model state, and
because writing and restoring a large snapshot has real I/O and memory cost.
