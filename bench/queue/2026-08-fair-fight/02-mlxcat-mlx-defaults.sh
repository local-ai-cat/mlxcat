#!/bin/zsh
# Arm B — the fair fight. Same binary, no ceiling, so MemoryGuard is a no-op and
# MLX runs at its own defaults: memoryLimit 1.5x the recommended working set and
# cacheLimit equal to it, ~54 GiB on this 48 GiB box. That is what oMLX, mlx-lm
# and mlx-serve have been getting all along, and mlxcat is the only engine on the
# board that was not.
#
# Unbounded on a 48 GiB machine is the shape that panicked this host on
# 2026-08-22. It is only safe to dispatch because the runaway guard kills at 92%
# of RAM now instead of letting the driver fall over.
set -uo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat
source bench/queue/2026-08-fair-fight/lib.sh

# Rebuild before measuring. The queue builds once in pass 00, and on 2026-08-22
# engine code landed between that build and this pass — so the A/B arm whose only
# difference is an env var ran against a binary that predated the env var, and
# both arms produced identical numbers that looked like a real result. An
# incremental build is seconds when nothing changed.
git pull --ff-only -q || { echo "GIT PULL FAILED — refusing to benchmark stale code"; exit 91 }
swift build -c release --product mlxcat-http 2>&1 | tail -1

python3 bench/run.py --engines mlxcat-defaults \
  --models "$MODELS" --contexts short,4k,16k,longgen --concurrency 2,4,8 \
  --runs 3 --warmup 1 --max-load 6 --resume --wait-for-quiet "$QUIET_WAIT" \
  --sync-after-engine "$SYNC" \
  --tag "fair-fight arm B: MLX defaults, no ceiling"
rc=$?
# `python3 ...; echo "rc=$?"` exits 0 no matter what python did, so the queue
# marked a NameError-crashed pass DONE and walked straight into the next one.
# The pass must carry its own exit code or the queue's halt-on-failure is decorative.
echo "armB rc=$rc"
exit $rc
