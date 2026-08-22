#!/bin/zsh
# A four-minute end-to-end proof that the whole loop works, before committing
# three hours to it. Both arms, the two models that matter for iOS, one width.
#
# This pass exists because the campaign has now been dispatched three times and
# crashed in the first second twice — on a NameError, then on a resume key that
# matched a different binary. Each failure cost a wait for the host to go quiet
# before it revealed itself. A cheap arm that runs the same code path first
# turns that into four minutes instead of ninety.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
source bench/queue/2026-08-fair-fight/lib.sh

python3 bench/run.py --engines mlxcat,mlxcat-defaults --profile quick \
  --max-load 6 --wait-for-quiet "$QUIET_WAIT" \
  --results-dir .build/quick-results \
  --tag "quick loop — smoke, never for the leaderboard"
rc=$?
echo "quick rc=$rc"
exit $rc
