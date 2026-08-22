# Shared settings for the fair-fight campaign. Sourced, not run.
MODELS=Qwen3.5-4B-MLX-4bit,gemma-4-E2B-it-qat-4bit,gpt-oss-20b-MXFP4-Q8,gemma-4-12B-it-qat-4bit,Qwen3.8-27B-4bit,Qwen3-Coder-30B-A3B-Instruct-4bit
CEIL=25769803776   # 24 GiB — the same ceiling every existing mlxcat row used

# Checkpoint rows off this machine after every engine. On 2026-08-22 the suite
# only synced at the end and a kernel panic left 160 finished rows unreachable
# for twelve hours. Failure here is reported, never fatal.
SYNC='git pull --rebase -q && git add bench/results/*.jsonl && git -c user.name="Atlas Codes AI" -c user.email="76924051+atlascodesai@users.noreply.github.com" commit -q -m "bench: checkpoint ${MLXCAT_BENCH_ENGINE} rows" && git push -q'

# The shell quiet-poll that used to live here required three consecutive
# loadavg readings under 5.0 and reset to zero on any blip. On a host that also
# runs CI that is an unbounded wait: on 2026-08-22 it spent 90 minutes and
# produced no rows. `--wait-for-quiet` inside run.py does the same job against
# the same thresholds as the guard, and a blip now costs one interval.
QUIET_WAIT=3600
