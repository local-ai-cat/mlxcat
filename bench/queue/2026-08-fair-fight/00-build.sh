#!/bin/zsh
# Build the release binaries this campaign measures. Debug is 2-4x slower and
# reject_debug_build() will now refuse it rather than publish a fake regression.
set -euo pipefail
source ~/.zshenv 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/mlxcat

if ! git pull --ff-only 2>&1 | tail -2; then
  echo "GIT PULL FAILED — refusing to benchmark stale harness code"; exit 91
fi
git log -1 --format='building %h %s'

swift build -c release --product mlxcat-http 2>&1 | tail -2
zsh scripts/build-metallib.sh 2>&1 | tail -1
test -x .build/release/mlxcat-http || { echo "no release binary"; exit 92 }
python3 bench/test_run.py 2>&1 | tail -3
echo BUILD_OK
