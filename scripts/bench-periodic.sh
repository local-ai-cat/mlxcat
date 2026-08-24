#!/bin/zsh
# bench-periodic.sh — the recurring measurement, end to end.
#
# One command turns a quiet machine into a new pass on the board: pull, build
# the engine at HEAD, run the default profile, regenerate LEADERBOARD.md and
# the timeline/dashboard, commit and push. bench/timeline.py auto-discovers
# the new result file and labels it from the harness tag, so a periodic run
# lands on the board without anyone editing Python.
#
#   scripts/bench-periodic.sh                        # all matrix models
#   MLXCAT_BENCH_MODELS=gemma-4-E2B-it-qat-4bit scripts/bench-periodic.sh
#   MLXCAT_BENCH_ENGINES=mlxcat scripts/bench-periodic.sh
#
# Intended cadence: weekly on the M4 Pro worker (Mac16,7) — the board's device.
# On any other machine the rows are still valid leaderboard input, but the
# Mac16,7 timeline board will not move; the script warns and continues.
# Scheduling (launchd vs a dispatched tmux run) is the operator's call; this
# script is deliberately schedule-agnostic.
#
# The harness's own quiet-machine guard does the gating: up to an hour waiting
# for load/memory/thermals, and any row measured under pressure is stamped
# valid_for_leaderboard=false rather than silently kept.

set -uo pipefail
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
cd "$REPO_ROOT"

ENGINES="${MLXCAT_BENCH_ENGINES:-mlxcat,mlxcat-cache-held}"
MODELS="${MLXCAT_BENCH_MODELS:-}"
GIT_ID=(-c user.name="Atlas Codes AI" -c user.email="76924051+atlascodesai@users.noreply.github.com")
# The checkpoint regenerates LEADERBOARD.md too: CI gates every push on
# "leaderboard derives from results", so a checkpoint that adds rows without
# re-rendering turns the whole pipeline red until the final commit (this is
# what kept mlxcat CI failing through 2026-08-24).
SYNC='git pull --rebase -q && python3 bench/leaderboard.py >/dev/null && git add bench/results/*.jsonl LEADERBOARD.md && git -c user.name="Atlas Codes AI" -c user.email="76924051+atlascodesai@users.noreply.github.com" commit -q -m "bench: checkpoint ${MLXCAT_BENCH_ENGINE} rows" && git push -q'

if [[ "$(sysctl -n hw.model)" != "Mac16,7" ]]; then
  echo "⚠️  $(sysctl -n hw.model) is not the board's device (Mac16,7): rows are kept, the timeline will not move"
fi

echo "== pull =="
git pull --rebase || exit 1
git log --oneline -1

echo "== build engine at HEAD =="
swift build -c release --product mlxcat-http || exit 1

echo "== bench (engines=$ENGINES${MODELS:+ models=$MODELS}) =="
args=(--engines "$ENGINES" --profile default
      --max-load 6 --wait-for-quiet 3600
      --memory-ceiling-bytes 25769803776 --resume
      --sync-after-engine "$SYNC"
      --tag "periodic $(date +%F) @ $(git rev-parse --short HEAD)")
[[ -n "$MODELS" ]] && args+=(--models "$MODELS")
# Unbuffered: piped python block-buffers ~4KB, which once hid 50 minutes of
# state during an apparent wedge and got a possibly-healthy run killed blind.
PYTHONUNBUFFERED=1 python3 bench/run.py "${args[@]}"
rc=$?

echo "== regenerate board (bench exit $rc) =="
python3 bench/leaderboard.py || rc=1
python3 bench/timeline.py || rc=1
git add bench/results/*.jsonl LEADERBOARD.md bench/timeline.json bench/dashboard.html 2>/dev/null
git "${GIT_ID[@]}" commit -m "bench: periodic pass $(date +%F), board regenerated" || true
git push || rc=1
echo "== DONE rc=$rc =="
exit $rc
