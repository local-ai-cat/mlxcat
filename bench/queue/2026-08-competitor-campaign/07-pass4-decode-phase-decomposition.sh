#!/bin/zsh
# Pass 4 — WHERE does the width-1 deficit live? Runs the built-in decode-phase
# decomposition on a quiet machine for the two models the M4 measured as having
# different root causes: Qwen3.5-4B (transport-bound: in-process ~= legacy) and
# gemma-4-E2B (engine-bound: in-process == HTTP, both ~0.90x legacy).
#
# Width1ThroughputTests interleaves TokenIterator / raw generator / engine /
# engine+prefix-store across rounds so host drift lands on each equally, and
# MLXSERVE_DECODE_PHASE_TIMING adds the per-phase breakdown of a decode step.
source ~/.zshenv
cd ~/src/mlxcat

# (the cross-pass wait that used to live here is gone: bench/queue.sh orders
#  the passes on disk, so a reboot no longer strands the one that was waiting)

quiet=0
while (( quiet < 3 )); do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  if (( $(echo "$l < 4.0" | bc -l) )); then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date +%H:%M) load=$l quiet=$quiet/3"
  (( quiet < 3 )) && sleep 120
done
echo PASS4_QUIET

root=$HOME/Library/Caches/models/mlx-community
for m in Qwen3.5-4B-MLX-4bit gemma-4-E2B-it-qat-4bit; do
  echo "===== width-1 decomposition: $m ====="
  MLXSERVE_DECODE_PHASE_TIMING=1 MLXSERVE_TEST_MODEL="$root/$m" \
    swift test --filter Width1ThroughputTests 2>&1 | grep -vE "^\[|Compiling|Build" | tail -18
done
echo PASS4_DONE
