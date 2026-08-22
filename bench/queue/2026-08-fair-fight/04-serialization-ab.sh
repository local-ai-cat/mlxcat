#!/bin/zsh
# The A/B that can actually move the leaderboard: batched decode ON vs OFF for
# the families NativeModelLoader excludes.
#
# `mlxcat`         status quo — gemma4 and qwen3_5 run serialized
# `mlxcat-batched` same binary, same ceiling, exclusions lifted for those families
#
# Expected shape if the exclusion is what costs us: aggregate throughput at c4
# is currently ~equal to a single stream (79.4 vs 80.8 tok/s on gemma-4-E2B,
# measured 2026-08-22) because four "concurrent" requests run one after another.
# If batching engages, aggregate should rise toward the 1.55-3.14x oMLX gets.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
source bench/queue/2026-08-fair-fight/lib.sh

python3 bench/run.py --engines mlxcat,mlxcat-batched \
  --models gemma-4-E2B-it-qat-4bit,Qwen3.5-4B-MLX-4bit \
  --contexts short,longgen --concurrency 2,4,8 --concurrency-tier longgen \
  --runs 3 --warmup 1 --max-load 6 --wait-for-quiet "$QUIET_WAIT" \
  --memory-ceiling-bytes "$CEIL" \
  --sync-after-engine "$SYNC" \
  --tag "serialization A/B: exclusions lifted for gemma4,qwen3_5"
rc=$?
echo "serialization-ab rc=$rc"
exit $rc
