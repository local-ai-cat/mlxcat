#!/bin/zsh
# Tool-calling evidence campaign — boot one server, walk the tool-capable
# fleet smallest -> largest, both grammar arms per request. See README.md.
#
#   bench/toolcalling/campaign.sh [--bin PATH] [--port N] [--models a,b,c]
#
# The server binary should be built from a tree that carries the per-request
# `tool_grammar` field (feat/tool-grammar or later); on an older server the
# field is ignored and the arms measure the same thing — the summary will say
# so by showing identical arms.
set -uo pipefail
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"

BIN="${REPO_ROOT}/.build/release/mlxcat-http"
PORT=11702
MODELS="Qwen3-1.7B-4bit,Llama-3.2-3B-Instruct-4bit,Qwen3.5-4B-MLX-4bit,Qwen2.5-Coder-7B-Instruct-4bit,gemma-4-E2B-it-qat-4bit,gpt-oss-20b-MXFP4-Q8,Qwen3.6-27B-4bit,Qwen3-Coder-30B-A3B-Instruct-4bit"
MODEL_DIR="${HOME}/Library/Caches/models"
SCENARIOS=""
NO_STREAM=""
while (( $# )); do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --no-stream-arm) NO_STREAM=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 64 ;;
  esac
done

[[ -x "$BIN" ]] || { echo "no server binary at $BIN — swift build -c release --product mlxcat-http" >&2; exit 66; }

SRVLOG="$SCRIPT_DIR/evidence/server-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$SCRIPT_DIR/evidence"
echo "== booting $BIN on :$PORT =="
"$BIN" --host 127.0.0.1 --port "$PORT" --model-dir "$MODEL_DIR" \
  --memory-ceiling-bytes 48318382080 > "$SRVLOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for _ in {1..60}; do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "server never came up — $SRVLOG" >&2; exit 69; }

extra_args=()
[[ -n "$SCENARIOS" ]] && extra_args+=(--scenarios "$SCENARIOS")
[[ -n "$NO_STREAM" ]] && extra_args+=(--no-stream-arm)
python3 "$SCRIPT_DIR/run.py" --base "http://127.0.0.1:$PORT" \
  --models "$MODELS" "${extra_args[@]}" \
  --summary "$SCRIPT_DIR/evidence/summary-$(date +%F).md"
rc=$?
echo "== campaign done (rc=$rc); server log: $SRVLOG =="
exit $rc
