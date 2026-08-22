# The pass queue

Drop a pass here as `NN-name.sh` and run `bench/queue.sh run` (or install the
LaunchAgent so login works the queue). Ordering is numeric; a pass that finishes
gets a `.done` marker and is never run again, so re-invoking after a reboot
resumes instead of restarting.

Passes are **committed**, not gitignored. The 2026-08-22 panic left the worker
holding the only copy of the campaign's orchestration — a machine that does not
come back takes the harness with it, which is the same lesson as the rows that
sat unreachable for twelve hours. A pass is cheap to keep and expensive to
rewrite from memory.

A pass should:

* `cd` to the repo and `git pull`, and **exit non-zero if the pull fails** —
  benchmarking a stale harness produces rows that look real and are not;
* pass `--resume` to `bench/run.py`, so a pass interrupted halfway does not pay
  again for cells it already recorded;
* not poll for the previous pass — the queue orders them.

```bash
bench/queue.sh status
bench/queue.sh run
bench/queue.sh install      # resume automatically after a reboot
bench/queue.sh retry 03-warm-cold
```
