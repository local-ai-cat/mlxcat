#!/bin/zsh
# Prove the engine before believing its numbers.
#
# mlxcat's model-gated suites — batch invariance, scheduler, chunked prefill,
# preemption, speculative decoding, prefix-cache reconstruction, hybrid/sliding/
# MoE batch integration, the 16k memory budget — only run when their env gate
# points at a local model, so `swift test` skips them and CI (hosted, no weights)
# has never executed one. The last recorded run of these was 2026-08-12.
#
# This pass ALWAYS exits 0 on purpose. A failing suite is the most important
# thing on the board, but halting an unattended benchmark night on it would cost
# the numbers as well as the news. The verdict is written to a file that pass 03
# reports next to the leaderboard, so it is loud without being fatal.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat

VERDICT=.build/core-tests-verdict.txt
: > "$VERDICT"

MLXCAT_NIGHTLY_OUT=.build/nightly-models.md zsh scripts/nightly-models.sh
failures=$?

{
  echo "failing_suites=$failures"
  echo "when=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit=$(git rev-parse --short HEAD)"
} > "$VERDICT"

if (( failures > 0 )); then
  echo "⚠️  $failures model-gated suite(s) FAILED — see .build/nightly-models.md"
else
  echo "✅ every model-gated suite that had a model on disk passed"
fi
exit 0
