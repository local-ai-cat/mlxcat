#!/bin/zsh
# First real leaderboard rows — waits for models + a quiet window, then benches.
source ~/.zshenv
cd ~/src/mlxcat
# 1. wait for the M4-side HF downloads (sentinel per model)
root="$HOME/Library/Caches/models/mlx-community"
for m in Qwen3.5-4B-MLX-4bit gemma-4-E2B-it-qat-4bit Qwen3.8-27B-4bit Qwen3-Coder-30B-A3B-Instruct-4bit; do
  until [ -f "$root/.done-$m" ]; do
    echo "$(date +%H:%M) waiting for model $m ($(du -sh "$root/$m" 2>/dev/null | cut -f1))"
    sleep 300
  done
  echo "model ready: $m"
done
# 2. wait for a quiet window (3 consecutive samples, load1 < 6)
quiet=0
while (( quiet < 3 )); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if (( $(echo "$l < 6.0" | bc -l) )); then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date +%F' '%H:%M) load=$l quiet=$quiet/3"
  (( quiet < 3 )) && sleep 120
done
echo QUIET_WINDOW_OPEN
# 3. bench — ceiling ARMED (24 GiB): this Mac has 48 GB and Qwen3.8-27B/16k
#    peaked 47.9 GiB with no ceiling; 24 GiB is the app-like config (19.8 GiB measured).
python3 bench/run.py --engines mlxcat,mlx-swift-lm-tokeniterator,mlxcat-inprocess \
  --models Qwen3.5-4B-MLX-4bit,gemma-4-E2B-it-qat-4bit,gpt-oss-20b-MXFP4-Q8,gemma-4-12B-it-qat-4bit,Qwen3.8-27B-4bit,Qwen3-Coder-30B-A3B-Instruct-4bit \
  --contexts short,4k,16k,longgen --concurrency 2,4,8 --runs 3 --warmup 1 --max-load 6 \
  --memory-ceiling-bytes 25769803776 \
  --tag "m4pro-first-rows ceiling-24G pin-e99f07c"
echo BENCH_RC=$?
python3 bench/leaderboard.py
echo BENCH_DONE
