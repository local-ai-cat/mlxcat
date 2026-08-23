#!/bin/zsh
# The first pass that measures qwen3_5 batching and the buffer-cache release.
#
# Pass 06 measured f0d39ba, which still had `qwen3_5` at `.always` because the
# family was believed to break batched decode. It does not. Its M-RoPE anchor
# lives in `LMOutput.State` rather than on the cache, `ContinuousBatchGenerator`
# dropped it the moment a second row joined, and the model's own precondition
# then trapped the SERVER PROCESS — which is what the benchmark saw as "width 4
# returns no usage frame, width 8 kills the server" and what
# docs/KNOWN-FAILURES.md §1d spent a day attributing to the HTTP layer. The
# crash report named the frame. See §1d, now rewritten.
#
# So this pass is the first that can measure two of the six leaderboard models
# under real concurrency:
#
#   Qwen3.5-4B and Qwen3.8-27B, both `.always` on every board so far. Pass 06
#   has 27B longgen c8 at 46.2 s TTFT with each stream running ALONE at full
#   speed; the prediction on file is 13-15 s.
#
# It also prices the other half of the day's work. `mlxcat-cache-held` runs the
# same binary with MLXCAT_IDLE_CLEAR_CACHE=0 and MLXCAT_DECODE_CLEAR_CACHE_STEPS=0
# — the behaviour that shipped before — so "the server hands 1.9 GiB back at
# idle" becomes a column rather than an in-process assertion.
#
# Read the peak column carefully: it is per PROCESS, one process serves every
# tier of a model, and MLX does not return its free list on its own. That is why
# pass 06's gemma-4-E2B longgen peak looked like a regression on cells that had
# not regressed — the 16k tier is new in 06, runs third, and its high-water mark
# was still resident. The `mlxcat` vs `mlxcat-cache-held` pair is the fix and the
# control for exactly that.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
source bench/queue/2026-08-fair-fight/lib.sh

git pull --ff-only -q || { echo "GIT PULL FAILED — refusing to benchmark stale code"; exit 91 }
swift build -c release --product mlxcat-http 2>&1 | tail -1

python3 bench/run.py --engines mlxcat,mlxcat-cache-held \
  --profile default \
  --max-load 6 --wait-for-quiet "$QUIET_WAIT" --resume \
  --memory-ceiling-bytes "$CEIL" \
  --sync-after-engine "$SYNC" \
  --tag "qwen3_5 batches + buffer-cache release: two of six models leave the serial lane"
rc=$?
echo "qwen35-and-cache-release rc=$rc"
exit $rc
