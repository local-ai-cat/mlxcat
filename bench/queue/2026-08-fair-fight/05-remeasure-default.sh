#!/bin/zsh
# Re-measure mlxcat on the DEFAULT profile with the fixed engine.
#
# Every mlxcat row on the leaderboard predates two changes that move it:
#   - the allocator fix (7f6cbb6), so the memory column reports a closed bug
#   - SerializationPolicy.multimodalOnly (d630cee), so every gemma-4 concurrency
#     cell reports a 50x TTFT penalty the engine no longer has
#
# Until this runs, docs/COMPETITIVE.md's headline — fourth of four — is measuring
# an engine that no longer exists. The competitor rows are unaffected and stay.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
source bench/queue/2026-08-fair-fight/lib.sh

git pull --ff-only -q || { echo "GIT PULL FAILED — refusing to benchmark stale code"; exit 91 }
swift build -c release --product mlxcat-http 2>&1 | tail -1

python3 bench/run.py --engines mlxcat --profile default \
  --max-load 6 --wait-for-quiet "$QUIET_WAIT" --resume \
  --memory-ceiling-bytes "$CEIL" \
  --sync-after-engine "$SYNC" \
  --tag "post-fix: allocator bounded + multimodalOnly batching"
rc=$?
echo "remeasure rc=$rc"
exit $rc
