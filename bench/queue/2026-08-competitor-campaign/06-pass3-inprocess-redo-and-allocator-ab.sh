#!/bin/zsh
# Pass 3 — rebuild with the fixes, RE-RUN the in-process rows under a real memory
# policy (the first set measured swap, not the engine), then A/B the allocator fix.
source ~/.zshenv
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
CEIL=25769803776   # 24 GiB, same as the server rows

# (the cross-pass wait that used to live here is gone: bench/queue.sh orders
#  the passes on disk, so a reboot no longer strands the one that was waiting)
echo PASS3_START
if ! git pull -q 2>&1 | tail -2; then
  echo "GIT PULL FAILED — refusing to benchmark stale harness code"; exit 91
fi
git log -1 --format='building %h %s'
swift build -c release --product mlxcat-http 2>&1 | tail -1
swift build -c release --product mlxcat-baseline 2>&1 | tail -1
zsh scripts/build-metallib.sh 2>&1 | tail -1

quiet=0
while (( quiet < 3 )); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if (( $(echo "$l < 6.0" | bc -l) )); then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date +%H:%M) load=$l quiet=$quiet/3"
  (( quiet < 3 )) && sleep 120
done
echo PASS3_QUIET
M=Qwen3.5-4B-MLX-4bit,gemma-4-E2B-it-qat-4bit,gpt-oss-20b-MXFP4-Q8,gemma-4-12B-it-qat-4bit,Qwen3.8-27B-4bit,Qwen3-Coder-30B-A3B-Instruct-4bit

# 1. the in-process rows, redone with the ceiling armed (replaces the bad set)
python3 bench/run.py --engines mlx-swift-lm-tokeniterator,mlxcat-inprocess \
  --models $M --contexts short,4k,16k,longgen --runs 3 --warmup 1 --max-load 6 \
  --memory-ceiling-bytes $CEIL --tag "in-process redo, ceiling armed"
echo "inprocess-redo rc=$?"

# 2. allocator fix ON, at the cells where the cliff was measured
python3 bench/run.py --engines mlxcat \
  --models gpt-oss-20b-MXFP4-Q8,Qwen3.8-27B-4bit,Qwen3.5-4B-MLX-4bit \
  --contexts 16k --concurrency 2,4,8 --concurrency-tier longgen --runs 3 --warmup 1 --max-load 6 \
  --memory-ceiling-bytes $CEIL --tag "allocator-fix-ON ceiling-24G"
echo "A rc=$?"

# 3. control: same binary, cache slice forced back to the MLX default
MLXCAT_CACHE_LIMIT_BYTES=68719476736 python3 bench/run.py --engines mlxcat \
  --models gpt-oss-20b-MXFP4-Q8 --contexts 16k --runs 3 --warmup 1 --max-load 6 \
  --memory-ceiling-bytes $CEIL --tag "allocator-fix-CONTROL cache-limit-64G"
echo "B rc=$?"

python3 bench/leaderboard.py
echo PASS3_DONE
