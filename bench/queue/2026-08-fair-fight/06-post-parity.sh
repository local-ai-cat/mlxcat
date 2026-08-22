#!/bin/zsh
# Measure the engine as it stands after the 2026-08-23 parity work.
#
# Pass 05 is the BASELINE: it measured 78b779f, before any of this. Since then,
# read out of the reference engines under guest/ rather than benchmarked into
# existence (see docs/PARITY-LEDGER.md):
#
#   - Memory.clearCache() between prefill chunks, as mlx-lm has always done
#     (guest/mlx-lm/mlx_lm/generate.py:451). Peak at 16k fell 58-64% on three
#     models and 20% on the fourth; 1.22-1.33x loaded weights across all four,
#     against 1.5-3.7x before.
#   - Idle and busy prefill get separate chunk widths (512 / 2048), because one
#     number was answering a memory question and a scheduling question at once.
#   - qwen3_moe batches at maxWidth(2) instead of not at all, and gemma4_unified
#     is capped at 4 — both from the cross-family logit gate, which gained a
#     width-1 control arm that proves the divergences are real.
#
# Every mlxcat number on the board predates all of it. This pass is what makes
# the claims numbers, and what `bench/parity.py --update` should be baselined
# from afterwards.
#
# The mlxcat-moe-uncapped arm prices the qwen3_moe width cap: uncapping is NOT a
# proposal (widths 4 and 8 fail the logit tolerance), it exists so the cost of
# holding the correctness bar is a number rather than an argument.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
source bench/queue/2026-08-fair-fight/lib.sh

# Rebuild before measuring. Pass 04's header records what happens when an A/B
# arm runs against a binary that predates the thing it is testing.
git pull --ff-only -q || { echo "GIT PULL FAILED — refusing to benchmark stale code"; exit 91 }
swift build -c release --product mlxcat-http 2>&1 | tail -1

python3 bench/run.py --engines mlxcat,mlxcat-moe-uncapped \
  --profile default \
  --max-load 6 --wait-for-quiet "$QUIET_WAIT" --resume \
  --memory-ceiling-bytes "$CEIL" \
  --sync-after-engine "$SYNC" \
  --tag "post-parity: clearCache between chunks, split chunk widths, width ceilings"
rc=$?
echo "post-parity rc=$rc"
exit $rc
