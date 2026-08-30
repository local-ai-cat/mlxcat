#!/bin/zsh
# run-device-toolcalls.sh — tool-call CORRECTNESS on a real iPhone.
#
#   ios/BenchHost/run-device-toolcalls.sh --device <name-or-udid> [--model <hf-id>]... [--out DIR]
#
# Boots the real OpenAIServer inside the BenchHost app (loopback, ephemeral
# port) and replays bench/toolcalling's scenario suite against it — the same
# scenarios and validators as the Mac campaigns (toolcall-scenarios.json is
# dumped from run.py --dump-scenarios; regenerate it there, never edit here).
# Rows: TOOLROW console lines + mlxcat-toolcall-ios.jsonl attachment.
#
# Knobs (forwarded when set): BENCHHOST_TRIALS, BENCHHOST_TEMPS.
# The phone must be paired, trusted, and unlocked when the test starts.
set -euo pipefail
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

DEFAULT_MODELS=(
  mlx-community/Qwen3-1.7B-4bit     # reliable tool-caller tier (fleet 42/42)
  mlx-community/Qwen3.5-4B-MLX-4bit # phone ceiling, also reliable
)

DEVICE=""
MODELS=()
OUT="$SCRIPT_DIR/.build/ios-toolcalls"
while (( $# )); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --model) MODELS+=("$2"); shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 64 ;;
  esac
done
[[ -n "$DEVICE" ]] || { echo "--device <name-or-udid> is required" >&2; exit 64; }
(( ${#MODELS[@]} )) || MODELS=("${DEFAULT_MODELS[@]}")
mkdir -p "$OUT"

xcodegen generate --spec project.yml >/dev/null

model_list="${(j:,:)MODELS}"
stamp=$(date +%Y%m%d-%H%M%S)
result="$OUT/toolcalls-$stamp.xcresult"
console="$OUT/toolcalls-$stamp.console.log"
echo "== tool-call correctness: $model_list on $DEVICE =="
extra_env=()
for var in BENCHHOST_TRIALS BENCHHOST_TEMPS; do
  [[ -n "${(P)var:-}" ]] && extra_env+=("TEST_RUNNER_${var}=${(P)var}")
done
env "${extra_env[@]}" \
TEST_RUNNER_BENCHHOST_MODEL_ID="$model_list" xcodebuild test \
  -project BenchHost.xcodeproj -scheme BenchHost \
  -derivedDataPath ".build/dd-$DEVICE" \
  -destination "platform=iOS,id=$DEVICE" \
  -only-testing:BenchHostTests/DeviceToolCallTests \
  -resultBundlePath "$result" \
  -test-timeouts-enabled NO \
  -allowProvisioningUpdates \
  -skipMacroValidation -skipPackagePluginValidation \
  2>&1 | tee "$console" | grep -E "Test Case|Test Suite|error:|TOOLHOST |failed" | tail -40 || true

exportdir="$OUT/toolcalls-$stamp-attachments"
mkdir -p "$exportdir"
xcrun xcresulttool export attachments --path "$result" --output-path "$exportdir" >/dev/null 2>&1 || true
found=$(find "$exportdir" -type f ! -name manifest.json 2>/dev/null | head -1)
if [[ -n "${found:-}" ]]; then
  cat "$found" >> "$OUT/ios-toolcalls.jsonl"
  echo "attachment rows appended -> $OUT/ios-toolcalls.jsonl"
else
  rows=$(sed -nE 's/.*TOOLROW (\{.*)/\1/p' "$console" | sort -u)
  if [[ -n "$rows" ]]; then
    print -r -- "$rows" >> "$OUT/ios-toolcalls.jsonl"
    echo "no attachment; $(echo "$rows" | grep -c .) TOOLROW line(s) recovered -> $OUT/ios-toolcalls.jsonl"
  else
    echo "⚠️  no attachment and no TOOLROW lines — read $result in Xcode"
  fi
fi
echo "DONE."
