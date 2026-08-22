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
# NOT `python3 bench/test_run.py | tail -3`. unittest writes its verdict to
# stderr, the end-to-end cases write rows to stdout, and tail -3 kept the rows
# and threw the verdict away — so this pass printed BUILD_OK over a test run
# nobody could see the result of. Silence is not a pass.
python3 bench/test_run.py 2>tests.err
test_rc=$?
grep -E "^(Ran |OK|FAILED)" tests.err || tail -5 tests.err
if (( test_rc != 0 )); then
  echo "HARNESS TESTS FAILED — refusing to benchmark with a broken harness"
  exit 93
fi
echo BUILD_OK
