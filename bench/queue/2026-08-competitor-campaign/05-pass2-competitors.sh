#!/bin/zsh
# Pass 2 — the competitor sweep. Waits for pass 1 to finish, then benches every
# other MLX-native engine over the SAME models/tiers/weights.
source ~/.zshenv
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
export OMLX_BIN="$HOME/.local/bin/omlx"
export MLX_LM_SERVER_BIN="$HOME/.local/bin/mlx_lm.server"
export VLLM_MLX_BIN="$HOME/.local/bin/vllm-mlx"
export MLX_SERVE_BIN="$HOME/src/mlx-serve-bin/mlx-serve-macos-arm64/mlx-serve"

# 1. wait for pass 1 (it prints BENCH_DONE into its own log)
# (the cross-pass wait that used to live here is gone: bench/queue.sh orders
#  the passes on disk, so a reboot no longer strands the one that was waiting)
echo PASS1_FINISHED
if ! git pull -q 2>&1 | tail -2; then
  echo "GIT PULL FAILED — refusing to benchmark stale harness code"; exit 91
fi

# 2. quiet window again
quiet=0
while (( quiet < 3 )); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if (( $(echo "$l < 6.0" | bc -l) )); then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date +%H:%M) load=$l quiet=$quiet/3"
  (( quiet < 3 )) && sleep 120
done
echo PASS2_QUIET

# 3. competitors, same matrix, same weights. Engines run one at a time so a
#    broken CLI in one never costs the others their rows.
for eng in omlx mlx-lm mlx-serve vllm-mlx; do
  echo "===== $eng ====="
  python3 bench/run.py --engines "$eng" \
    --models Qwen3.5-4B-MLX-4bit,gemma-4-E2B-it-qat-4bit,gpt-oss-20b-MXFP4-Q8,gemma-4-12B-it-qat-4bit,Qwen3.8-27B-4bit,Qwen3-Coder-30B-A3B-Instruct-4bit \
    --contexts short,4k,16k,longgen --concurrency 2,4,8 --runs 3 --warmup 1 --max-load 6 \
    --memory-ceiling-bytes 25769803776 \
    --tag "m4pro-competitors $eng"
  echo "$eng rc=$?"
done
python3 bench/leaderboard.py
echo PASS2_DONE
