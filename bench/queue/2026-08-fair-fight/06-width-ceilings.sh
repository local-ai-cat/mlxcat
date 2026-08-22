#!/bin/zsh
# Price the width ceilings, on the same machine and profile as pass 05.
#
# Pass 05 measured mlxcat at 78b779f, where qwen3_moe was `.always` (never
# batched) and gemma4_unified was uncapped. Since then:
#
#   qwen3_moe        .always      -> .maxWidth(2)
#   gemma4_unified   uncapped     -> maxWidth 4, multimodal solitude kept
#
# Both come from the cross-family logit gate with its new width-1 control arm:
# batch 1 is bit-exact for every family, so the divergences are rows interfering,
# and the ceiling for each family is the widest batch that still clears the bar
# it can clear. See docs/KNOWN-FAILURES.md 1e.
#
# Three points, so the cap has a price rather than a justification:
#   pass 05 rows          fully serialized (qwen3_moe) / uncapped (gemma4_unified)
#   `mlxcat`              the ceilings as shipped
#   `mlxcat-moe-uncapped` no cap on qwen3_moe at all
#
# The uncapped arm is NOT a proposal — widths 4 and 8 fail the tolerance. It is
# here so "the correctness bar costs us X tok/s at c8" is a number.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
source bench/queue/2026-08-fair-fight/lib.sh

# Rebuild: the whole point is that the binary carries the ceilings. Pass 04's
# header records what happens when it does not.
git pull --ff-only -q || { echo "GIT PULL FAILED — refusing to benchmark stale code"; exit 91 }
swift build -c release --product mlxcat-http 2>&1 | tail -1

python3 bench/run.py --engines mlxcat,mlxcat-moe-uncapped \
  --models Qwen3-Coder-30B-A3B-Instruct-4bit,gemma-4-12B-it-qat-4bit \
  --contexts short,longgen --concurrency 2,4,8 --concurrency-tier longgen \
  --runs 3 --warmup 1 --max-load 6 --wait-for-quiet "$QUIET_WAIT" \
  --resume \
  --memory-ceiling-bytes "$CEIL" \
  --sync-after-engine "$SYNC" \
  --tag "width ceilings: qwen3_moe maxWidth 2, gemma4_unified maxWidth 4"
rc=$?
echo "width-ceilings rc=$rc"
exit $rc
