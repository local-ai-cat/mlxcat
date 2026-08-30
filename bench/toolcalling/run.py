#!/usr/bin/env python3
"""Tool-calling evidence harness — the end-to-end A/B the grammar work still owed.

The tool-grammar lever shipped default-ON priced by a 12-prompt, one-model A/B
(TOOLGRAMMAR-RESULTS.md). Its own "Still open" list names the gap this harness
closes: nothing measures the *application* cost and the *behavioral* payoff of
the grammar end to end, across the fleet, across scenarios harder than
quote-heavy bash strings. Every trial's raw response is retained: the summary
is derived from the evidence file, never the other way around.

Usage:
  python3 bench/toolcalling/run.py --base http://127.0.0.1:11702 \
      --models Qwen3-Coder-30B-A3B-Instruct-4bit[,...] \
      [--arms grammar-on,grammar-off]   # default; use `default` for non-mlxcat servers
      [--temps 0,0.7] [--trials 3] [--engine-label mlxcat]
      [--scenarios forced,auto_call,...] [--out bench/toolcalling/evidence]

Per (model x arm x temperature x scenario x trial) one evidence row lands in
  <out>/<date>-<host>-<runid>.jsonl
and a summary table renders to stdout (and --summary FILE if given).

Arms: `grammar-on` / `grammar-off` send mlxcat's per-request `tool_grammar`
bool; `default` sends no field (for LM Studio / other OpenAI-compatible
servers — label the engine via --engine-label).
"""

import argparse
import datetime
import json
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid

# --- scenarios ---------------------------------------------------------------
# Each scenario: tools offered, messages, expectation. The checker returns a
# dict of booleans; `call_expected` drives which rates are meaningful.

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
    },
}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

TICKET_TOOL = {
    "type": "function",
    "function": {
        "name": "file_ticket",
        "description": "File a structured support ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "priority": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "expected": {"type": "string"},
                        },
                        "required": ["action", "expected"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["title", "priority", "tags", "steps"],
            "additionalProperties": False,
        },
    },
}

NOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "Save a text note verbatim.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}

# --- long-context distractor -------------------------------------------------
# The grammar lever's documented value case is the long-context tail
# (TOOLGRAMMAR-RESULTS.md): unconstrained decoding once collapsed when the
# call had to be emitted after a large KV cache. The fleet campaign measured
# short-context scenarios only; these cells load the cache with neutral prose
# and then ask for the same calls, so any validity gap between arms is
# attributable to context depth, not task difficulty. The filler is
# deterministic (same text every run) and the ask explicitly sets it aside —
# pure KV load, no retrieval confound.

def distractor_text(approx_tokens):
    topics = [
        ("watermill", "the wheel turns the shaft, the shaft turns the "
         "stones, and the sluice gate meters the head of water"),
        ("lighthouse", "the lamp rotates behind a Fresnel lens whose panels "
         "concentrate the beam into timed flashes keepers once logged by hand"),
        ("orchard", "grafted rootstock sets the tree's size while the scion "
         "sets the fruit, and pruning trades this year's yield for next"),
        ("glacier", "accumulated snow compresses into firn and then ice that "
         "flows downhill, plucking and grinding the bedrock beneath"),
        ("printing press", "movable type is locked into a forme, inked, and "
         "pressed into dampened paper one sheet at a time"),
        ("tidal flat", "each ebb exposes a mud plain worked by wading birds "
         "whose bills are tuned to different burrow depths"),
    ]
    paragraphs = []
    total_chars = 0
    i = 0
    while total_chars < approx_tokens * 4:  # ~4 chars/token for English prose
        name, body = topics[i % len(topics)]
        para = (f"Section {i + 1}: notes on the {name}. In this section we "
                f"revisit how {body}. Observation {i + 1} was recorded on day "
                f"{(i * 7) % 365 + 1} of the survey and cross-checked twice; "
                f"no anomalies were found, and the log was archived unchanged.")
        paragraphs.append(para)
        total_chars += len(para) + 2
        i += 1
    return "\n\n".join(paragraphs)

def _longctx(approx_tokens, tool, ask):
    return {
        "tools": [tool],
        "messages": [
            {"role": "user", "content":
                "Here is a survey document for our records:\n\n"
                + distractor_text(approx_tokens)
                + "\n\nYou've read the document above. Setting it aside "
                  "completely: " + ask},
        ],
        "call_expected": True,
        "max_tokens": 1024,
    }

SCENARIOS = {
    # The original A/B's territory: forced call, hostile quoting.
    "forced": {
        "tools": [BASH_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "bash"}},
        "messages": [{"role": "user", "content":
            "Use bash to print exactly: she said \"don't\" and left. Use printf."}],
        "call_expected": True,
        "max_tokens": 1024,
    },
    # Unforced: does the model choose to call at all? (tool_choice omitted)
    "auto_call": {
        "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content":
            "What's the weather in London right now, in celsius? Use the tool."}],
        "call_expected": True,
        "max_tokens": 1024,
    },
    # Unforced with no reason to call: a call here is a FALSE call.
    "auto_no_call": {
        "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content":
            "What is 2+2? Answer directly with just the number."}],
        "call_expected": False,
        "max_tokens": 1024,
    },
    # Two independent lookups in one turn: parallel/multi tool calls.
    "parallel": {
        "tools": [WEATHER_TOOL],
        "messages": [{"role": "user", "content":
            "Get the weather for BOTH Paris and Tokyo, celsius. "
            "Call the tool once per city."}],
        "call_expected": True,
        "min_calls": 2,
        "max_tokens": 1024,
    },
    # Deep nested schema: arrays of objects with required keys.
    "nested_schema": {
        "tools": [TICKET_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "file_ticket"}},
        "messages": [{"role": "user", "content":
            "File a ticket: title 'Login fails on retina displays', priority 2, "
            "tags auth and ui, two repro steps (open login page expecting the "
            "form to render, submit valid credentials expecting the dashboard)."}],
        "call_expected": True,
        "max_tokens": 1024,
    },
    # Unicode, newlines, escapes riding through argument encoding.
    "unicode_args": {
        "tools": [NOTE_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "save_note"}},
        "messages": [{"role": "user", "content":
            "Save this note verbatim, preserving the line break:\n"
            "こんにちは \"世界\" — 100% ✓\nsecond line with a\ttab"}],
        "call_expected": True,
        "max_tokens": 1024,
    },
    # A long free-text argument: enclosure integrity over ~200+ tokens.
    "long_args": {
        "tools": [NOTE_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "save_note"}},
        "messages": [{"role": "user", "content":
            "Save a note that is a single ~150-word paragraph about how a river "
            "mill works, mentioning the wheel, the millstones, and the sluice "
            "gate. Prose only, no lists."}],
        "call_expected": True,
        "max_tokens": 1536,
        "min_arg_chars": 400,
    },
    # The grammar's value case: the same simple call, but emitted after a
    # deep KV cache. Unforced tool_choice so the grammar actually arms.
    "longctx_4k": dict(
        _longctx(4000, WEATHER_TOOL,
                 "what's the weather in London right now, in celsius? "
                 "Use the tool."),
    ),
    "longctx_12k": dict(
        _longctx(12000, WEATHER_TOOL,
                 "what's the weather in London right now, in celsius? "
                 "Use the tool."),
    ),
    # Nested schema after a deep cache — the hardest shape at the hardest
    # depth. Same file_ticket ask as nested_schema, but unforced.
    "longctx_nested_12k": dict(
        _longctx(12000, TICKET_TOOL,
                 "use the file_ticket tool to file a ticket: title 'Login "
                 "fails on retina displays', priority 2, tags auth and ui, "
                 "two repro steps (open login page expecting the form to "
                 "render, submit valid credentials expecting the dashboard)."),
    ),
    # Turn 2: after a tool result comes back, the model must ANSWER, not loop.
    "round_trip": {
        "tools": [WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "What's the weather in Oslo, celsius? Use the tool."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_rt1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": "{\"city\": \"Oslo\", \"unit\": \"celsius\"}"}}]},
            {"role": "tool", "tool_call_id": "call_rt1",
             "content": "{\"temp_c\": 7, \"conditions\": \"light rain\"}"},
        ],
        "call_expected": False,
        "expect_answer_contains": "7",
        "max_tokens": 1024,
    },
}

# --- minimal JSON-schema validation (objects/arrays/strings/ints/enums) ------

def schema_valid(value, schema):
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return False
        for key in schema.get("required", []):
            if key not in value:
                return False
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            if any(k not in props for k in value):
                return False
        return all(k not in props or schema_valid(v, props[k]) for k, v in value.items())
    if t == "array":
        return isinstance(value, list) and all(
            schema_valid(v, schema.get("items", {})) for v in value)
    if t == "string":
        return isinstance(value, str) and (
            "enum" not in schema or value in schema["enum"])
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    return True

# --- transport ---------------------------------------------------------------

def post_chat(base, body, timeout):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if body.get("stream"):
            first_at = None
            chunks = []
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if first_at is None:
                    first_at = time.monotonic()
                chunks.append(json.loads(payload))
            return chunks, started, first_at, time.monotonic()
        data = json.load(r)
    return data, started, None, time.monotonic()

def assemble_stream(chunks):
    """Reassemble OpenAI streaming deltas into a message dict."""
    calls = {}
    content = []
    finish = None
    usage = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0),
                                        {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
        if chunk.get("usage"):
            usage = chunk["usage"]
    tool_calls = [
        {"id": slot["id"], "type": "function",
         "function": {"name": slot["name"], "arguments": slot["arguments"]}}
        for _, slot in sorted(calls.items())
    ]
    return {"content": "".join(content) or None,
            "tool_calls": tool_calls, "finish": finish, "usage": usage}

# --- one trial ---------------------------------------------------------------

def run_trial(base, model, scenario_name, scenario, arm, temperature, stream, timeout):
    body = {
        "model": model,
        "messages": scenario["messages"],
        "tools": scenario["tools"],
        "max_tokens": scenario.get("max_tokens", 256),
        "temperature": temperature,
        "stream": stream,
    }
    if "tool_choice" in scenario:
        body["tool_choice"] = scenario["tool_choice"]
    if arm == "grammar-on":
        body["tool_grammar"] = True
    elif arm == "grammar-off":
        body["tool_grammar"] = False
    if stream:
        body["stream_options"] = {"include_usage": True}

    row = {
        "schema": "mlxcat-toolcall/1",
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds"),
        "model": model, "scenario": scenario_name, "arm": arm,
        "temperature": temperature, "stream": stream,
    }
    try:
        result, started, first_at, ended = post_chat(base, body, timeout)
    except Exception as e:  # noqa: BLE001 — the failure IS the evidence
        row.update({"error": f"{type(e).__name__}: {e}", "ok": False})
        return row

    if stream:
        message = assemble_stream(result)
        finish = message["finish"]
        usage = message["usage"]
        if first_at is not None:
            row["ttfc_ms"] = round((first_at - started) * 1000, 1)
    else:
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        message = {"content": message.get("content"),
                   "tool_calls": message.get("tool_calls") or [],
                   "finish": choice.get("finish_reason")}
        finish = message["finish"]
        usage = result.get("usage", {})

    wall = ended - started
    completion_tokens = usage.get("completion_tokens")
    row["wall_s"] = round(wall, 3)
    row["prompt_tokens"] = usage.get("prompt_tokens")
    row["completion_tokens"] = completion_tokens
    if completion_tokens and wall > 0:
        row["tok_per_s"] = round(completion_tokens / wall, 2)
    row["finish_reason"] = finish
    row["raw_content"] = (message["content"] or "")[:2000]
    row["raw_tool_calls"] = message["tool_calls"]

    calls = message["tool_calls"]
    called = len(calls) > 0
    row["called"] = called
    row["n_calls"] = len(calls)

    expected_tool = scenario["tools"][0]["function"]
    parsed_args = []
    json_ok = bool(calls)
    schema_ok = bool(calls)
    names_ok = bool(calls)
    for call in calls:
        fn = call.get("function") or {}
        if fn.get("name") != expected_tool["name"]:
            names_ok = False
        try:
            args = json.loads(fn.get("arguments") or "")
            parsed_args.append(args)
        except Exception:  # noqa: BLE001
            json_ok = False
            schema_ok = False
            continue
        if not schema_valid(args, expected_tool["parameters"]):
            schema_ok = False

    if scenario["call_expected"]:
        ok = called and json_ok and schema_ok and names_ok
        if "min_calls" in scenario:
            ok = ok and len(calls) >= scenario["min_calls"]
        if "min_arg_chars" in scenario and parsed_args:
            longest = max(len(str(v)) for a in parsed_args for v in a.values())
            row["longest_arg_chars"] = longest
            ok = ok and longest >= scenario["min_arg_chars"]
    else:
        ok = not called
        if "expect_answer_contains" in scenario:
            ok = ok and scenario["expect_answer_contains"] in (message["content"] or "")
    row.update({"json_valid": json_ok, "schema_valid": schema_ok,
                "name_correct": names_ok, "ok": ok})
    return row

# --- campaign ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:11702")
    ap.add_argument("--models", required=True)
    ap.add_argument("--arms", default="grammar-on,grammar-off")
    ap.add_argument("--temps", default="0,0.7")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--engine-label", default="mlxcat")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="bench/toolcalling/evidence")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--no-stream-arm", action="store_true",
                    help="skip the streaming duplicate of each cell")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    temps = [float(t) for t in args.temps.split(",")]
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in scenario_names if s not in SCENARIOS]
    if unknown:
        sys.exit(f"unknown scenarios: {unknown}; have {list(SCENARIOS)}")

    import os
    os.makedirs(args.out, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.date.today().isoformat()
    evidence_path = os.path.join(
        args.out, f"{stamp}-{socket.gethostname().split('.')[0]}-{run_id}.jsonl")

    rows = []
    with open(evidence_path, "a") as evidence:
        for model in models:
            for arm in arms:
                for temp in temps:
                    for name in scenario_names:
                        scenario = SCENARIOS[name]
                        stream_modes = [False] if args.no_stream_arm else [False, True]
                        # Stream only where a call is expected: the streaming
                        # question is "do deltas reassemble", not "does the
                        # model decline twice".
                        if not scenario["call_expected"]:
                            stream_modes = [False]
                        for stream in stream_modes:
                            for _ in range(args.trials):
                                row = run_trial(args.base, model, name, scenario,
                                                arm, temp, stream, args.timeout)
                                row["engine"] = args.engine_label
                                evidence.write(json.dumps(row) + "\n")
                                evidence.flush()
                                rows.append(row)
                                marker = "ok " if row.get("ok") else "FAIL"
                                print(f"{marker} {model} {arm} t={temp} {name}"
                                      f"{' stream' if stream else ''}"
                                      f" tok/s={row.get('tok_per_s', '—')}"
                                      f"{' err=' + row['error'][:60] if 'error' in row else ''}",
                                      flush=True)

    print(f"\nevidence -> {evidence_path}")
    summary = render_summary(rows, args.engine_label)
    print(summary)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(summary + "\n")

def render_summary(rows, engine):
    out = [f"\n## {engine} tool-calling summary "
           f"({len(rows)} trials, {datetime.date.today().isoformat()})",
           "", "| model | arm | t | pass | json | schema | false-call | med tok/s |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    def key(r):
        return (r["model"], r["arm"], r["temperature"])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    for (model, arm, temp), rs in sorted(groups.items()):
        n = len(rs)
        ok = sum(1 for r in rs if r.get("ok"))
        expected = [r for r in rs if SCENARIOS[r["scenario"]]["call_expected"]]
        json_ok = sum(1 for r in expected if r.get("json_valid"))
        schema_ok = sum(1 for r in expected if r.get("schema_valid"))
        no_call = [r for r in rs if not SCENARIOS[r["scenario"]]["call_expected"]
                   and "expect_answer_contains" not in SCENARIOS[r["scenario"]]]
        false_calls = sum(1 for r in no_call if r.get("called"))
        speeds = sorted(r["tok_per_s"] for r in rs if r.get("tok_per_s"))
        med = speeds[len(speeds) // 2] if speeds else None
        out.append(f"| {model} | {arm} | {temp} | {ok}/{n} "
                   f"| {json_ok}/{len(expected)} | {schema_ok}/{len(expected)} "
                   f"| {false_calls}/{len(no_call)} | {med or '—'} |")
    return "\n".join(out)

if __name__ == "__main__":
    main()
