#!/bin/zsh
# run-device-bench.sh — drive BenchHost on a real iPhone and collect the rows.
#
#   ios/BenchHost/run-device-bench.sh --device <name-or-udid> [--model <hf-id>]... [--out DIR]
#
# For each model (default: the ios roster below) it runs DeviceRowProducerTests
# on the device, pulls the mlxcat-bench-ios.jsonl attachment out of the
# xcresult, and appends the rows into --out (default .build/ios-rows/).
# Rows arrive valid_for_leaderboard=false; promote them after confirming the
# run conditions held (device unlocked ONCE to launch, then screen left alone;
# on power; not hot). First run per model downloads weights into the app
# container over the cable — leave the phone plugged in.
#
# The phone must be paired, trusted, and unlocked when the test starts.
# MLX does not run on simulators.
set -euo pipefail
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

# The ios: true roster from bench/matrix.json, perf-relevant subset.
# 0.6B is wire-format smoke only and deliberately not benched; the 8 GB
# iPhone app ceiling (~6 GB) rules out everything above the 4B.
DEFAULT_MODELS=(
  mlx-community/gemma-4-E2B-it-qat-4bit   # device floor — the headline row
  mlx-community/Qwen3-1.7B-4bit           # small
  mlx-community/Qwen3.5-4B-MLX-4bit       # ceiling of what the phone fits
)

DEVICE=""
MODELS=()
OUT="$SCRIPT_DIR/.build/ios-rows"
while (( $# )); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --model) MODELS+=("$2"); shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 64 ;;
  esac
done
[[ -n "$DEVICE" ]] || { echo "--device <name-or-udid> is required (xcrun xctrace list devices)" >&2; exit 64; }
(( ${#MODELS[@]} )) || MODELS=("${DEFAULT_MODELS[@]}")
mkdir -p "$OUT"

xcodegen generate --spec project.yml >/dev/null

# ONE xcodebuild invocation for the whole roster: a passcode phone can
# auto-lock in the minutes between per-model launches, and iOS SIGKILLs
# Metal work behind a locked screen — the first device night died to
# exactly that. The test loops the comma-separated list inside a single
# app session with the idle timer held.
model_list="${(j:,:)MODELS}"
stamp=$(date +%Y%m%d-%H%M%S)
result="$OUT/night-$stamp.xcresult"
console="$OUT/night-$stamp.console.log"
echo "== roster: $model_list on $DEVICE =="
TEST_RUNNER_BENCHHOST_MODEL_ID="$model_list" xcodebuild test \
  -project BenchHost.xcodeproj -scheme BenchHost \
  -derivedDataPath ".build/dd-$DEVICE" \
  -destination "platform=iOS,id=$DEVICE" \
  -resultBundlePath "$result" \
  -test-timeouts-enabled NO \
  -allowProvisioningUpdates \
  -skipMacroValidation -skipPackagePluginValidation \
  2>&1 | tee "$console" | grep -E "Test Case|Test Suite|error:|BENCHHOST |failed" | tail -30 || true

exportdir="$OUT/night-$stamp-attachments"
mkdir -p "$exportdir"
xcrun xcresulttool export attachments --path "$result" --output-path "$exportdir" >/dev/null 2>&1 || true
found=$(find "$exportdir" -type f ! -name manifest.json 2>/dev/null | head -1)
if [[ -n "${found:-}" ]]; then
  cat "$found" >> "$OUT/ios-rows.jsonl"
  echo "attachment rows appended -> $OUT/ios-rows.jsonl"
else
  # A crashed session leaves no attachment, but every completed cell printed
  # a BENCHROW line first — recover those instead of losing the night.
  rows=$(sed -nE 's/.*BENCHROW (\{.*)/\1/p' "$console" | sort -u)
  if [[ -n "$rows" ]]; then
    print -r -- "$rows" >> "$OUT/ios-rows.jsonl"
    echo "no attachment; $(echo "$rows" | grep -c .) BENCHROW line(s) recovered -> $OUT/ios-rows.jsonl"
  else
    echo "⚠️  no attachment and no BENCHROW lines — read $result in Xcode"
  fi
fi
echo "DONE. Review $OUT/ios-rows.jsonl, flip valid_for_leaderboard on rows whose run conditions held, then append into bench/results/ and re-render."
