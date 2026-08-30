#!/usr/bin/env python3
"""Replay the ORIGINAL failing pi conversation, verbatim, against both arms.

The 2026-08-27 pi session (Qwen3-Coder-30B via the app's Local API) produced
the malformed legacy-JSON tool body that started the tool-grammar work:
    <tool_call>{"name": "bash", "arguments": {"command":"find ... -name "*.xcodeproj" ...
The synthetic single-shot replay (pi_find_repro) never reproduced it in 220
trials — but that replayed the ASK, not the CONVERSATION. This script rebuilds
the exact message history from the pi session file, sends it to an
OpenAI-compatible server N times per arm, and classifies every response:

  structured_ok    server returned tool_calls whose arguments parse as JSON
  structured_bad   server returned tool_calls with unparseable arguments
  leaked_text      a raw <tool_call> block arrived as content text — the
                   original failure signature (parser could not structure it)
  no_call          plain prose, no call attempt
  length           ran out of budget

Usage:
  python3 bench/toolcalling/replay_bad_chat.py \
      [--session PATH] [--base http://127.0.0.1:11702] \
      [--model Qwen3-Coder-30B-A3B-Instruct-4bit] \
      [--trials 20] [--temps 0,0.7] [--arms grammar-on,grammar-off]

Caveats (recorded, not hidden): pi's system prompt and exact tool schemas are
not stored in the session file, so the replay approximates them with
pi-shaped equivalents. The message history, model, and ask are verbatim.
"""

import argparse
import datetime
import json
import os
import socket
import urllib.request
import uuid

DEFAULT_SESSION = (
    "/Users/timapple/.pi/agent/sessions/"
    "--Users-timapple-Documents-mcc-ventures-mochiexists-mochi-records--/"
    "2026-08-27T14-40-19-045Z_01a043aa-2464-7c14-b8f1-a4c045fc9a4f.jsonl"
)

# pi-shaped tool schemas (approximation; pi 0.74 offers read/write/edit/bash).
TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a file and return its contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write",
        "description": "Write content to a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replace text in a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "old": {"type": "string"},
                                      "new": {"type": "string"}},
                       "required": ["path", "old", "new"]}}},
]


def rebuild_messages(session_path, upto_failing=True):
    """pi session events -> OpenAI chat messages, stopping just before the
    failing assistant turn so the model must produce that turn itself."""
    messages = []
    pending_call_id = None
    for line in open(session_path):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "message":
            continue
        m = e["message"]
        role = m["role"]
        content = m.get("content")
        if role == "user" and isinstance(content, list):
            text = "".join(p.get("text", "") for p in content
                           if p.get("type") == "text")
            messages.append({"role": "user", "content": text})
        elif role == "assistant" and isinstance(content, list):
            texts = [p.get("text", "") for p in content if p.get("type") == "text"]
            calls = [p for p in content if p.get("type") == "toolCall"]
            joined = "".join(texts)
            if upto_failing and "on-the-app-store" in joined and "xcodeproj" in joined:
                break  # the failing turn — the replay must regenerate it
            msg = {"role": "assistant"}
            msg["content"] = joined or None
            if calls:
                msg["tool_calls"] = [{
                    "id": c["id"], "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c.get("arguments", {}))},
                } for c in calls]
                pending_call_id = calls[-1]["id"]
            messages.append(msg)
        elif role == "toolResult" and isinstance(content, list):
            text = "".join(p.get("text", "") for p in content
                           if p.get("type") == "text")
            messages.append({"role": "tool",
                             "tool_call_id": pending_call_id or "call_unknown",
                             "content": text})
    return messages


def classify(message, finish):
    calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    if calls:
        for c in calls:
            try:
                json.loads(c["function"]["arguments"] or "")
            except Exception:  # noqa: BLE001
                return "structured_bad"
        return "structured_ok"
    if "<tool_call>" in content:
        return "leaked_text"  # the original failure signature
    if finish == "length":
        return "length"
    return "no_call"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--base", default="http://127.0.0.1:11702")
    ap.add_argument("--model", default="Qwen3-Coder-30B-A3B-Instruct-4bit")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--temps", default="0,0.7")
    ap.add_argument("--arms", default="grammar-on,grammar-off")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="bench/toolcalling/evidence")
    args = ap.parse_args()

    messages = rebuild_messages(args.session)
    print(f"rebuilt {len(messages)} messages from the bad chat "
          f"(last: {messages[-1]['role']})")

    os.makedirs(args.out, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.date.today().isoformat()
    evidence_path = os.path.join(
        args.out,
        f"{stamp}-{socket.gethostname().split('.')[0]}-badchat-{run_id}.jsonl")

    tally = {}
    with open(evidence_path, "a") as evidence:
        for arm in [a.strip() for a in args.arms.split(",")]:
            for temp in [float(t) for t in args.temps.split(",")]:
                for i in range(args.trials):
                    body = {"model": args.model, "messages": messages,
                            "tools": TOOLS, "max_tokens": args.max_tokens,
                            "temperature": temp}
                    if arm == "grammar-on":
                        body["tool_grammar"] = True
                    elif arm == "grammar-off":
                        body["tool_grammar"] = False
                    req = urllib.request.Request(
                        args.base + "/v1/chat/completions",
                        data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"})
                    row = {"schema": "mlxcat-badchat-replay/1",
                           "timestamp": datetime.datetime.now(
                               datetime.timezone.utc).isoformat(timespec="seconds"),
                           "model": args.model, "arm": arm, "temperature": temp,
                           "trial": i}
                    try:
                        with urllib.request.urlopen(req, timeout=600) as r:
                            data = json.load(r)
                        choice = (data.get("choices") or [{}])[0]
                        message = choice.get("message") or {}
                        outcome = classify(message, choice.get("finish_reason"))
                        row["outcome"] = outcome
                        row["finish_reason"] = choice.get("finish_reason")
                        row["raw_content"] = (message.get("content") or "")[:1500]
                        row["raw_tool_calls"] = message.get("tool_calls") or []
                    except Exception as e:  # noqa: BLE001
                        outcome = "error"
                        row["outcome"] = "error"
                        row["error"] = f"{type(e).__name__}: {e}"
                    evidence.write(json.dumps(row) + "\n")
                    evidence.flush()
                    tally.setdefault((arm, temp), {}).setdefault(outcome, 0)
                    tally[(arm, temp)][outcome] += 1
                    print(f"{arm} t={temp} trial {i}: {outcome}", flush=True)

    print(f"\nevidence -> {evidence_path}\n")
    print("| arm | t | " + " | ".join(
        ["structured_ok", "structured_bad", "leaked_text", "no_call",
         "length", "error"]) + " |")
    print("|---|---:|" + "---:|" * 6)
    for (arm, temp), counts in sorted(tally.items()):
        cells = " | ".join(str(counts.get(k, 0)) for k in
                           ["structured_ok", "structured_bad", "leaked_text",
                            "no_call", "length", "error"])
        print(f"| {arm} | {temp} | {cells} |")


if __name__ == "__main__":
    main()
