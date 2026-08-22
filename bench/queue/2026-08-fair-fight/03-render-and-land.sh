#!/bin/zsh
# Land whatever the two arms produced. The per-engine sync already pushed the
# rows; this re-renders the leaderboard so CI's derived-from-results gate passes.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
cd ~/src/mlxcat
git pull --rebase -q 2>&1 | tail -2
python3 bench/leaderboard.py
git add LEADERBOARD.md bench/results/*.jsonl 2>/dev/null
git -c user.name="Atlas Codes AI" -c user.email="76924051+atlascodesai@users.noreply.github.com" \
  commit -q -m "bench: fair-fight arms — mlxcat at product config and at MLX defaults" || echo "nothing to commit"
git push -q && echo PUSHED
python3 bench/leaderboard.py --check && echo LEADERBOARD_OK
