#!/bin/zsh
# run-context-ladder.sh — find the real context ceiling on a phone.
#
#   ios/BenchHost/run-context-ladder.sh --device <udid> [--model <hf-id>]... \
#       [--rungs 256,1024,4096,...] [--variant mlxcat|tokeniterator] [--out DIR]
#
# Walks prompt length up until the model stops fitting and reports, per model:
#   MAX SURVIVED  — the largest prompt that completed
#   KILLED AT     — the rung that was attempted and never finished (jetsam), or
#   REFUSED AT    — the rung the model itself rejected (context-window overflow)
#
# The distinction matters: "killed" is OUR memory problem and moves with the
# device; "refused" is the model's own context window and does not.
#
# A successful ladder ends with the app being SIGKILLed, so this script never
# relies on the test passing or on an xcresult attachment existing. It reads the
# BENCHRUNG console lines, which are printed before and after each rung.
set -euo pipefail
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

DEVICE=""; MODELS=(); RUNGS=""; VARIANT="mlxcat"
OUT="$SCRIPT_DIR/.build/ios-ladder"
while (( $# )); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --model) MODELS+=("$2"); shift 2 ;;
    --rungs) RUNGS="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 64 ;;
  esac
done
[[ -n "$DEVICE" ]] || { echo "--device <udid> is required" >&2; exit 64 }
(( ${#MODELS[@]} )) || MODELS=(mlx-community/gemma-4-E2B-it-qat-4bit)
mkdir -p "$OUT"

xcodegen generate --spec project.yml >/dev/null

model_list="${(j:,:)MODELS}"
stamp=$(date +%Y%m%d-%H%M%S)
console="$OUT/ladder-$stamp.console.log"
echo "== ladder: $model_list on $DEVICE (variant $VARIANT, rungs ${RUNGS:-default}) =="

# `|| true`: the ladder SUCCEEDING means the process dies, so a non-zero
# xcodebuild exit is the expected ending, not an error to propagate.
# KV-quantization env must be FORWARDED with the TEST_RUNNER_ prefix or it
# never reaches the device process and a "kv4 ladder" silently measures fp16
# — the exact settings-that-never-arrive failure the producer's kv echo line
# exists to catch. Invoke as: MLXCAT_KV_BITS=4 MLXCAT_QUANTIZED_KV_START=0 <script>
TEST_RUNNER_BENCHHOST_LADDER=1 \
TEST_RUNNER_BENCHHOST_MODEL_ID="$model_list" \
TEST_RUNNER_BENCHHOST_LADDER_RUNGS="$RUNGS" \
TEST_RUNNER_BENCHHOST_LADDER_VARIANT="$VARIANT" \
${MLXCAT_KV_BITS:+TEST_RUNNER_MLXCAT_KV_BITS="$MLXCAT_KV_BITS"} \
${MLXCAT_QUANTIZED_KV_START:+TEST_RUNNER_MLXCAT_QUANTIZED_KV_START="$MLXCAT_QUANTIZED_KV_START"} \
xcodebuild test \
  -project BenchHost.xcodeproj -scheme BenchHost \
  -derivedDataPath ".build/dd-$DEVICE" \
  -destination "platform=iOS,id=$DEVICE" \
  -only-testing:BenchHostTests/ContextCeilingLadderTests \
  -test-timeouts-enabled NO \
  -allowProvisioningUpdates \
  -skipMacroValidation -skipPackagePluginValidation \
  2>&1 | tee "$console" | grep -E "BENCHRUNG|BENCHHOST |error:" || true

rungs_file="$OUT/ladder-$stamp.jsonl"
sed -nE 's/.*BENCHRUNG [a-z]+ (\{.*)/\1/p' "$console" > "$rungs_file" || true
echo
echo "=== context ceiling ==="
python3 - "$rungs_file" << 'PY'
import json, sys
from collections import defaultdict
attempted, survived, refused, meta = defaultdict(set), defaultdict(dict), {}, {}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    key = (r.get("model"), r.get("device", {}).get("model"))
    n = r.get("prompt_tokens")
    if r.get("event") == "attempt":
        attempted[key].add(n)
        meta[key] = r
    elif r.get("event") == "survived":
        survived[key][n] = r
    elif r.get("event") == "refused":
        refused[key] = r
if not attempted:
    print("no BENCHRUNG lines — read the console log")
    raise SystemExit(1)
for key in sorted(attempted, key=lambda k: (k[0] or "", k[1] or "")):
    model, device = key
    ok = survived[key]
    top = max(ok) if ok else None
    gap = sorted(attempted[key] - set(ok))
    print(f"\n{device}  {model}")
    if top is None:
        print("  MAX SURVIVED  none — died on the first rung")
    else:
        row = ok[top]
        peak = row.get("peak_phys_footprint_bytes") or 0
        ceil = row.get("ceiling_bytes") or 0
        head = (ceil - peak) / 2**30 if ceil and peak else 0
        print(f"  MAX SURVIVED  {top:>7} prompt tokens"
              f"   peak {peak/2**30:5.2f} GiB   ceiling {ceil/2**30:5.2f} GiB"
              f"   headroom {head:5.2f} GiB")
    if key in refused:
        print(f"  REFUSED AT    {refused[key]['prompt_tokens']:>7} — the model's own context window, not memory")
        print(f"                {str(refused[key].get('error'))[:110]}")
    elif gap:
        print(f"  KILLED AT     {gap[0]:>7} prompt tokens — attempted, never finished (jetsam)")
    else:
        print("  ceiling NOT reached — every rung survived; extend --rungs")
PY
echo
echo "rungs -> $rungs_file"
echo "console -> $console"
