# Shared settings for the fair-fight campaign. Sourced, not run.
MODELS=Qwen3.5-4B-MLX-4bit,gemma-4-E2B-it-qat-4bit,gpt-oss-20b-MXFP4-Q8,gemma-4-12B-it-qat-4bit,Qwen3.8-27B-4bit,Qwen3-Coder-30B-A3B-Instruct-4bit
CEIL=25769803776   # 24 GiB — the same ceiling every existing mlxcat row used

# Checkpoint rows off this machine after every engine. On 2026-08-22 the suite
# only synced at the end and a kernel panic left 160 finished rows unreachable
# for twelve hours. Failure here is reported, never fatal.
SYNC='git pull --rebase -q && git add bench/results/*.jsonl && git -c user.name="Atlas Codes AI" -c user.email="76924051+atlascodesai@users.noreply.github.com" commit -q -m "bench: checkpoint ${MLXCAT_BENCH_ENGINE} rows" && git push -q'

wait_for_quiet() {
  # run.py REFUSES a loaded host rather than waiting, and this box also runs CI.
  local quiet=0 l
  while (( quiet < 3 )); do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    if (( $(echo "$l < 5.0" | bc -l) )); then quiet=$((quiet+1)); else quiet=0; fi
    echo "$(date +%H:%M) load=$l quiet=$quiet/3"
    (( quiet < 3 )) && sleep 120
  done
}
