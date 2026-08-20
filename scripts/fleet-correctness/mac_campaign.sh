#!/bin/zsh
set -u
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
BIN="${MLXSERVE_HTTP_BIN:-$REPO_ROOT/.build/release/mlxcat-http}"
MODELS="${MLXSERVE_MODEL_DIR:-/Users/timapple/Library/Caches/models}"
PORT="${MLXSERVE_FLEET_PORT:-11500}"
RESULTS="${MLXSERVE_FLEET_RESULTS:-$SCRIPT_DIR/mac-campaign-results.log}"
SRVLOG="${MLXSERVE_FLEET_SERVER_LOG:-$SCRIPT_DIR/mac-campaign-server.log}"
: > "$RESULTS"

# Models smallest -> largest (bare leaf ids as the server discovers them).
MODEL_LIST=(
  Qwen3-0.6B-4bit
  Llama-3.2-1B-Instruct-4bit
  Qwen3-1.7B-4bit
  DeepSeek-R1-Distill-Qwen-1.5B-4bit
  Qwen2-VL-2B-Instruct-4bit
  Llama-3.2-3B-Instruct-4bit
  Qwen3.5-4B-MLX-4bit
  DeepSeek-R1-Distill-Qwen-7B-4bit
  Qwen2.5-Coder-7B-Instruct-4bit
  gemma-4-E2B-it-qat-4bit
  gemma-4-E4B-it-qat-4bit
  Ornith-1.0-9B-6bit
  gpt-oss-20b-MXFP4-Q8
  Qwen3.6-27B-4bit
  Qwen3-Coder-30B-A3B-Instruct-4bit
  Qwen3.6-35B-A3B-4bit
)

echo "booting mlxcat @ $PORT (ceiling 45G)..." | tee -a "$RESULTS"
"$BIN" --host 127.0.0.1 --port $PORT --model-dir "$MODELS" \
  --memory-ceiling-bytes 48318382080 > "$SRVLOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in {1..40}; do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
echo "server ready ($(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["data"]),"models")'))" | tee -a "$RESULTS"

for m in $MODEL_LIST; do
  # memory gate: wait until >40% free before loading the next model
  for w in {1..30}; do
    FREE=$(memory_pressure -Q | awk -F': ' '/System-wide memory free percentage/ {gsub(/%/,"",$2);print $2}')
    (( FREE > 40 )) && break
    sleep 3
  done
  echo "--- $m (free ${FREE}%) ---" | tee -a "$RESULTS"
  python3 "$SCRIPT_DIR/correctness_check.py" "http://127.0.0.1:$PORT" "$m" 600 2>&1 | tee -a "$RESULTS"
done

echo "=== CAMPAIGN DONE ===" | tee -a "$RESULTS"
