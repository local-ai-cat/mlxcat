#!/bin/zsh
# Arm A — mlxcat exactly as the product ships it: a 24 GiB ceiling, which since
# 7f6cbb6 also clamps the MLX buffer cache to 4 GiB.
#
# Every existing mlxcat row predates that fix (measured at harness e99f07c on
# 08-21 13:46-15:07; 7f6cbb6 landed 23:57 the same night), so this is not a
# re-run for its own sake — the current rows are measuring a bug we fixed.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
source bench/queue/2026-08-fair-fight/lib.sh

python3 bench/run.py --engines mlxcat \
  --models "$MODELS" --contexts short,4k,16k,longgen --concurrency 2,4,8 \
  --runs 3 --warmup 1 --max-load 6 --resume --wait-for-quiet "$QUIET_WAIT" \
  --memory-ceiling-bytes "$CEIL" \
  --sync-after-engine "$SYNC" \
  --tag "fair-fight arm A: product config, ceiling 24G, post-allocator-fix"
rc=$?
# `python3 ...; echo "rc=$?"` exits 0 no matter what python did, so the queue
# marked a NameError-crashed pass DONE and walked straight into the next one.
# The pass must carry its own exit code or the queue's halt-on-failure is decorative.
echo "armA rc=$rc"
exit $rc
