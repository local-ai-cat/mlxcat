#!/bin/zsh
# Pass 5 — the WARM cache leg. Every number measured so far is cold-cache, which
# charges mlxcat and oMLX for their prefix caches and credits neither. This is the
# cell where a tiered prefix KV cache can actually hit.
source ~/.zshenv
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
CEIL=25769803776

# (the cross-pass wait that used to live here is gone: bench/queue.sh orders
#  the passes on disk, so a reboot no longer strands the one that was waiting)
if ! git pull -q 2>&1 | tail -2; then
  echo "GIT PULL FAILED — refusing to benchmark stale harness code"; exit 91
fi
git log -1 --format='at %h %s'

quiet=0
while (( quiet < 3 )); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if (( $(echo "$l < 6.0" | bc -l) )); then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date +%H:%M) load=$l quiet=$quiet/3"
  (( quiet < 3 )) && sleep 120
done
echo PASS5_QUIET

# cold+warm side by side, so each engine's cache is valued against its own cold row.
for eng in mlxcat omlx mlx-lm; do
  echo "===== warm-vs-cold: $eng ====="
  python3 bench/run.py --engines "$eng" \
    --models Qwen3.5-4B-MLX-4bit,gpt-oss-20b-MXFP4-Q8,Qwen3.8-27B-4bit \
    --contexts 4k,16k --cache-modes cold,warm --concurrency 2 --runs 3 --warmup 1 --max-load 6 \
    --memory-ceiling-bytes $CEIL --tag "prefix-cache value: $eng"
  echo "$eng rc=$?"
done
python3 bench/leaderboard.py
echo PASS5_DONE
