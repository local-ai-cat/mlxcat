# Fair fight — mlxcat at product config vs mlxcat at MLX defaults

Why this campaign exists: `--memory-ceiling-bytes` reaches only mlxcat's launch
template. oMLX, mlx-lm and mlx-serve have always run with MLX's own defaults
(memoryLimit 1.5× the recommended working set, cacheLimit equal to it — roughly
54 GiB on this 48 GiB box), while mlxcat ran at 24 GiB with a 4 GiB cache slice.
Starving buffer reuse costs prefill throughput, which is exactly the metric where
we score 0.61×. We were reading a product decision as an engine result.

Two arms, same binary, same day, same machine:

| pass | arm | ceiling |
|---|---|---|
| `01` | A — what the product ships | 24 GiB, cache clamped to 4 GiB |
| `02` | B — what the comparison is entitled to | none; MLX defaults |

Both are also the first mlxcat rows measured **after** the allocator fix
(`7f6cbb6`). Every row currently on the board predates it by ten hours.

```bash
bench/queue.sh status bench/queue/2026-08-fair-fight
bench/queue.sh run    bench/queue/2026-08-fair-fight
```

Each pass passes `--resume`, so a host that dies mid-matrix costs the cells it
had left. `--sync-after-engine` pushes rows as each engine finishes rather than
at the end — the 2026-08-22 panic stranded 160 finished rows for twelve hours
because the suite only synced once.
