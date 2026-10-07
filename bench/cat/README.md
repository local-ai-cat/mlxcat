# Cat inference benchmark

What the host app (Local AI Cat) asks of `mlxcat-http`, one request at a time,
over the OpenAI chat API. Four regular cases per model, plus a coding suite on
the coding model:

| case | prompt | check |
|---|---|---|
| `short_chat` | ~21–44 tok | replies `CAT_OK` |
| `grounded_extract` | ~56–78 tok | compact JSON `{project, owner, budget, deadline}` from a one-line source; `project` may be `Orion` or `Project Orion` (the source says "Project Orion") |
| `context_2k` / `context_8k` | ~2k / ~10k tok of records | returns the needle in record N |
| coding (coding model only) | 4 tasks | scored in a strict evaluator sandbox; fails closed when that sandbox is unavailable |

Each regular case runs 4 times: a first request, then 3 exact replays.
`temperature` 0, seed 42, 128 max tokens (512 for coding). The server is
launched per model with a loopback-only sandbox, `--max-concurrent-requests 1`,
and the default configuration (no `MLXCAT_*` levers set). The harness stops
the run if swap grows by more than 1 GiB, free disk drops under 40 GiB,
battery drops under 30 % or memory pressure warns.

The harness is in [`harness/`](harness/) (since 2026-10-07; see `harness/HARNESS.md`). The coding suite has 16 tasks and runs on any model via `--coding-models`. Each results
directory records the binary's SHA-256 and the mlxcat revision it was built
from. Results are not leaderboard rows (`bench/results/` is for `bench/run.py`).

## 2026-10-07, M4 Pro (Mac16,7, 48 GiB, macOS 27.0.1)

[`results/2026-10-07-m4-pro/`](results/2026-10-07-m4-pro/): `summary.md`,
`results.csv`, `results.json`. Built at `99beca4`, release
(`mlxcat-http` SHA-256 `c0f574e4ece8…`). 68 requests between 18:39:59 and
18:43:47 UTC. 1-minute load was 7.46 at the start. Swap held at 11.6 GiB used
from start to end, and `swap_delta_bytes` is 0 in every cell.

| model | short TTFT | decode tok/s | 2k first → replay TTFT | 8k first → replay TTFT | regular pass | peak RSS |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | 0.015 s | 333 | 0.53 → 0.53 s | 3.50 → 3.53 s | 8/16 (needles 0/8) | 0.46 GiB |
| Llama-3.2-3B-Instruct-4bit | 0.077 s | 125 | 1.94 → 1.91 s | 8.79 → 8.75 s | 16/16 | 1.89 GiB |
| Qwen3.5-4B-MLX-4bit | 0.067 s | 87 | **3.64 → 0.76 s** | **15.13 → 0.58 s** | 16/16 | 3.08 GiB |
| Qwen2.5-Coder-7B-Instruct-4bit | 0.163 s | 67 | 5.39 → 5.41 s | 23.44 → 23.47 s | 16/16 | 4.12 GiB |

Reading the table:

- **Decode tok/s** is the `short_chat` median.
- **Replay TTFT:**
  - Hybrid prefix reuse (Qwen3.5) turns an 8k replay into a 0.58 s TTFT, with 9728 of 10038 tokens served from cache.
  - Non-hybrid models reuse only the chat-template head on exact replays (3–43 tokens), so their replays cost what the first request did. `MLXCAT_PREFIX_FULL_MATCH_REUSE` (off; `docs/LEVERS.md`) is the lever for that.
- **Qwen3-0.6B** misses both needles. That is model quality at 0.6B; it is a wire-format smoke model.
- **Coding on Qwen2.5-Coder-7B: 0/4.**
  - Three tasks are BLOCKED: the evaluator sandbox is unavailable on this host, so they fail closed.
  - `json_tool_call` returned no parseable JSON.
