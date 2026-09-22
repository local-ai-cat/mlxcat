#!/bin/zsh
# One real pi coding task against one OpenAI-compatible engine.
#
#   bench/coding/pi_task.sh --label ENGINE --base http://127.0.0.1:PORT/v1 --model MODEL_ID \
#       --out DIR [--api-key KEY] [--fixture DIR] [--timeout-s 900] [--context-window 32768]
#
# The fixture is a tiny Swift package with two failing tests (a slugify bug). It is
# reset from a pristine copy (INCLUDING its prebuilt .build, at the same absolute
# path, so every engine starts from the same warm compile state). pi gets the same
# seed prompt every time, an isolated PI_CODING_AGENT_DIR (never ~/.pi), no context
# files, no extensions, no skills. Verdict = `swift test` passes afterwards AND the
# test file is byte-identical. Writes pi's JSON event stream (the transcript), the
# final diff, and a one-line summary JSON into --out.
set -uo pipefail
LABEL=""; BASE=""; MODEL=""; OUT=""; KEY="none"; TIMEOUT_S=900; CTX=32768
FIXTURE="${HOME}/.cache/lac-bench/pi-task/pristine"
WORK="/tmp/lac-pi-task/work"
while (( $# )); do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --api-key) KEY="$2"; shift 2 ;;
    --fixture) FIXTURE="$2"; shift 2 ;;
    --timeout-s) TIMEOUT_S="$2"; shift 2 ;;
    --context-window) CTX="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 64 ;;
  esac
done
[[ -n "$LABEL" && -n "$BASE" && -n "$MODEL" && -n "$OUT" ]] || { echo "need --label --base --model --out" >&2; exit 64; }
PI="${PI_BIN:-$HOME/.ovm/bin/pi}"
SEED='The tests in this Swift package fail. Run `swift test`, find the bug in Sources/, and fix it so that every test passes. Do not modify anything under Tests/. When all tests pass, stop and say DONE.'

mkdir -p "$OUT" "$(dirname $WORK)"
rsync -a --delete "$FIXTURE/" "$WORK/"
TEST_SHA_BEFORE=$(shasum -a 256 "$WORK/Tests/SlugTests/SlugTests.swift" | cut -d' ' -f1)

AGENT_DIR="$OUT/pi-agent-$LABEL"
rm -rf "$AGENT_DIR"; mkdir -p "$AGENT_DIR"
python3 - "$AGENT_DIR/models.json" "$BASE" "$MODEL" "$KEY" "$CTX" <<'PY'
import json, sys
path, base, model, key, ctx = sys.argv[1:]
json.dump({"providers": {"localaicat": {
    "baseUrl": base, "api": "openai-completions", "apiKey": key,
    "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
    "models": [{"id": model, "name": model, "reasoning": False, "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": int(ctx), "maxTokens": 8192}]}}}, open(path, "w"), indent=1)
PY

START=$(python3 -c 'import time;print(time.time())')
( cd "$WORK" && PI_CODING_AGENT_DIR="$AGENT_DIR" PI_OFFLINE=1 \
    timeout "$TIMEOUT_S" "$PI" --provider localaicat --model "$MODEL" --mode json -p \
    --no-context-files --no-extensions --no-skills --no-prompt-templates --no-themes \
    --session-dir "$OUT/sessions-$LABEL" "$SEED" ) > "$OUT/pi-$LABEL.jsonl" 2> "$OUT/pi-$LABEL.stderr"
PI_RC=$?
END=$(python3 -c 'import time;print(time.time())')

( cd "$WORK" && swift test ) > "$OUT/verdict-$LABEL.log" 2>&1
TEST_RC=$?
TEST_SHA_AFTER=$(shasum -a 256 "$WORK/Tests/SlugTests/SlugTests.swift" | cut -d' ' -f1)
( cd "$WORK" && git diff ) > "$OUT/diff-$LABEL.patch"

python3 - "$OUT" "$LABEL" "$MODEL" "$BASE" "$PI_RC" "$TEST_RC" "$TEST_SHA_BEFORE" "$TEST_SHA_AFTER" "$START" "$END" <<'PY'
import json, sys, os, collections
out, label, model, base, pi_rc, test_rc, sha0, sha1, t0, t1 = sys.argv[1:]
events = []
for line in open(os.path.join(out, f"pi-{label}.jsonl"), errors="replace"):
    line = line.strip()
    if line.startswith("{"):
        try: events.append(json.loads(line))
        except json.JSONDecodeError: pass
types = collections.Counter(e.get("type") for e in events)
tool_calls = collections.Counter()
for e in events:
    if e.get("type") == "tool_execution_start":
        tool_calls[e.get("toolName") or e.get("tool") or "?"] += 1
summary = {
    "label": label, "model": model, "base": base,
    "pi_exit": int(pi_rc), "timed_out": int(pi_rc) == 124,
    "tests_pass": int(test_rc) == 0, "tests_untouched": sha0 == sha1,
    "passed": int(test_rc) == 0 and sha0 == sha1,
    "wall_s": round(float(t1) - float(t0), 1),
    "turns": types.get("turn_end") or types.get("turn_start") or None,
    "tool_calls": dict(tool_calls), "event_types": dict(types),
}
print(json.dumps(summary))
open(os.path.join(out, f"summary-{label}.json"), "w").write(json.dumps(summary, indent=1) + "\n")
PY
