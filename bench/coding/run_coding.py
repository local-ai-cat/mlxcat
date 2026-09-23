#!/usr/bin/env python3
"""Coding-shaped workloads that bench/run.py's filler-paragraph tiers cannot express.

See bench/coding/README.md. Two workloads, both emitting `mlxcat-bench/1` rows
into bench/results/ with `workload.kind` set:

  cached_prefix  a real-code prefix (9 K / 18 K tokens) sent cold on turn 1, then
                 five follow-up turns that keep the whole conversation (including
                 the engine's own replies) as the prefix. TTFT per turn is the
                 result; turns 2-6 collapsing toward decode-only latency means the
                 engine reused the prefix.
  file_rewrite   "rewrite this file with X changed": ~1.8 K tokens of structured
                 output. Decode tok/s, wall-clock, and a mechanical correctness
                 check of the rewritten file.

The script never launches an engine; --base/--engine-label/--model are asserted
by the caller. Standard library only; imports the guard + device helpers from
bench/run.py so the quiet-machine verdict is the same code path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run as harness  # noqa: E402

# Corpus, pinned by path; bytes are hashed into every row. Token counts were
# measured with the Qwen3.8-27B tokenizer on Local-AI-Chat 486d2d295:
# LLMEvaluator.swift = 18,314 tokens whole; its first 865 lines = 9,000.
PREFIX_SPECS = {
    "prefix9k": {"files": ["app/Local AI Chat/Core/LLM/LLMEvaluator.swift"], "max_lines": 865},
    "prefix18k": {"files": ["app/Local AI Chat/Core/LLM/LLMEvaluator.swift"], "max_lines": None},
}
FOLLOW_UPS = [
    "In one short paragraph: what does this file's main type do?",
    "Name the function that is most likely to block the main thread, and why, in two sentences.",
    "Which property holds the loaded model? Answer with just its name and type.",
    "Suggest one concrete refactor to shrink this file. One paragraph.",
    "Is there any force unwrap in the code above? Answer yes or no and cite one line if yes.",
    "Summarise the error-handling strategy in two sentences.",
]
REWRITE_FILE = "Packages/DevLoopStreamKit/Sources/DevLoopStreamKit/FrameRasterizer.swift"
REWRITE_INSTRUCTION = (
    "Rewrite the Swift file below with exactly two changes: rename the enum `FrameRasterizer` "
    "to `FrameRenderer`, and rename the function `makePixelBuffer` to `makeCVPixelBuffer` "
    "(update every call site). Change nothing else. Reply with only the complete updated file "
    "in a single ```swift code block."
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_corpus(corpus_dir: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    parts = []
    for rel in spec["files"]:
        lines = (corpus_dir / rel).read_text().splitlines(True)
        if spec.get("max_lines"):
            lines = lines[: spec["max_lines"]]
        parts.append(f"// FILE: {rel}\n" + "".join(lines))
    text = "\n".join(parts)
    return {"text": text, "files": spec["files"], "max_lines": spec.get("max_lines"), "sha256": sha256(text)}


class Client:
    def __init__(self, base: str, api_key: str, timeout: float, extra: Dict[str, Any]):
        self.base = base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra = extra

    def headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def stream(self, model: str, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        }
        body.update(self.extra)
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=self.headers())
        started = time.perf_counter()
        first: Optional[float] = None
        stamps: List[float] = []
        usage: Dict[str, Any] = {}
        content, reasoning = [], []
        finish = None
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    d = ch.get("delta") or {}
                    c = d.get("content")
                    r = d.get("reasoning_content") or d.get("reasoning")
                    if (isinstance(c, str) and c) or (isinstance(r, str) and r):
                        now = time.perf_counter()
                        first = first or now
                        stamps.append(now)
                    if isinstance(c, str):
                        content.append(c)
                    if isinstance(r, str):
                        reasoning.append(r)
        done = time.perf_counter()
        if first is None:
            raise RuntimeError("no visible output")
        ct = int(usage.get("completion_tokens") or 0)
        pt = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        dec_s = stamps[-1] - first
        return {
            "ttft_ms": (first - started) * 1000,
            "wall_ms": (done - started) * 1000,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cached_tokens": details.get("cached_tokens"),
            "finish_reason": finish,
            "chunks": len(stamps),
            # Chunk-cadence decode, published whether or not the engine streams one
            # token per chunk (speculative engines emit several per chunk). The
            # `tokens_per_chunk` field says which it was.
            "decode_tps": (ct - 1) / dec_s if ct > 1 and dec_s > 0.05 else None,
            "tokens_per_chunk": (ct / len(stamps)) if stamps else None,
            "content": "".join(content),
            "reasoning_chars": sum(len(x) for x in reasoning),
        }


def spread(vals):
    return harness.spread(vals)


def base_row(args, device, snapshot, violations, workload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "mlxcat-bench/1",
        "timestamp": harness.now_iso(),
        "platform": "macos",
        "device": device,
        "engine": {"name": args.engine_label, "transport": "http", "version": args.engine_version,
                   "weights": args.weights},
        "model": {"id": args.model_id or args.model, "offered_as": args.model},
        "harness": {"commit": harness.git_commit(HERE.parent.parent), "producer": "bench/coding/run_coding.py",
                    "tag": args.tag, "argv": sys.argv[1:]},
        "host": snapshot,
        "valid_for_leaderboard": not violations,
        "invalid_reason": "; ".join(violations) or None,
        "workload": workload,
    }


def cached_prefix(args, client, device, corpus_commit) -> List[Dict[str, Any]]:
    rows = []
    for size in [s.strip() for s in args.prefix_sizes.split(",") if s.strip()]:
        corpus = load_corpus(Path(args.corpus_dir), PREFIX_SPECS[size])
        per_turn: List[List[Dict[str, Any]]] = [[] for _ in range(args.turns)]
        pre = harness.host_snapshot()
        viol = harness.host_violations(pre, args)
        transcripts = []
        for run_index in range(args.runs):
            nonce = uuid.uuid4().hex[:12]
            messages = [
                {"role": "system", "content": f"[session {nonce}] You are a senior Swift reviewer. Answer briefly."},
                {"role": "user", "content": corpus["text"] + "\n\n" + FOLLOW_UPS[0]},
            ]
            for turn in range(args.turns):
                if turn > 0:
                    messages.append({"role": "user", "content": FOLLOW_UPS[turn % len(FOLLOW_UPS)]})
                r = client.stream(args.model, messages, args.followup_max_tokens)
                r.pop("content_hash", None)
                per_turn[turn].append(r)
                messages.append({"role": "assistant", "content": r["content"]})
                print(f"  {size} run{run_index + 1} turn{turn + 1}: ttft {r['ttft_ms']:.0f} ms, prompt {r['prompt_tokens']}, cached {r['cached_tokens']}", flush=True)
            transcripts.append([{"role": m["role"], "chars": len(m["content"]), "sha256": sha256(m["content"])} for m in messages])
        post = harness.host_snapshot()
        viol = viol + [v for v in harness.host_violations(post, args) if v not in viol]
        turns = []
        for i, rs in enumerate(per_turn):
            turns.append({
                "turn": i + 1,
                "ttft_ms": spread([r["ttft_ms"] for r in rs]),
                "prompt_tokens": int(statistics.median([r["prompt_tokens"] for r in rs])),
                "cached_tokens": [r["cached_tokens"] for r in rs],
                "decode_tps": spread([r["decode_tps"] for r in rs]),
                "completion_tokens": int(statistics.median([r["completion_tokens"] for r in rs])),
            })
        t1 = turns[0]["ttft_ms"]["median"]
        later = [t["ttft_ms"]["median"] for t in turns[1:]]
        row = base_row(args, device, {"start": pre, "end": post}, viol, {
            "kind": "cached_prefix", "context_tier": f"coding-{size}", "concurrency": 1,
            "runs": args.runs, "turns": args.turns, "max_tokens": args.followup_max_tokens, "temperature": 0,
            "corpus": {"repo": "Local-AI-Chat", "commit": corpus_commit, "files": corpus["files"],
                       "max_lines": corpus["max_lines"], "sha256": corpus["sha256"]},
        })
        row["metrics"] = {
            "turns": turns,
            "ttft_ms": turns[0]["ttft_ms"],  # turn-1 (cold) TTFT, so generic readers see the cold number
            "prompt_tokens": turns[0]["prompt_tokens"],
            "followup_ttft_median_ms": statistics.median(later) if later else None,
            "reuse_ratio": (statistics.median(later) / t1) if later and t1 else None,
            "prefix_reuse_inferred": bool(later and t1 and statistics.median(later) < 0.5 * t1),
            "transcript_shape": transcripts[0],
        }
        rows.append(row)
    return rows


def file_rewrite(args, client, device, corpus_commit) -> List[Dict[str, Any]]:
    src = (Path(args.corpus_dir) / REWRITE_FILE).read_text()
    results = []
    pre = harness.host_snapshot()
    viol = harness.host_violations(pre, args)
    samples = []
    for i in range(args.runs + 1):
        nonce = uuid.uuid4().hex[:12]
        msgs = [{"role": "user", "content": f"[request {nonce}] {REWRITE_INSTRUCTION}\n\n```swift\n{src}```"}]
        r = client.stream(args.model, msgs, args.rewrite_max_tokens)
        body = r.pop("content")
        m = re.search(r"```swift\n(.*?)```", body, re.S)
        code = m.group(1) if m else ""
        r["rewrite_ok"] = bool(code) and "enum FrameRenderer" in code and "makeCVPixelBuffer" in code \
            and "FrameRasterizer" not in code and "makePixelBuffer(" not in code \
            and abs(code.count("\n") - src.count("\n")) <= max(10, src.count("\n") // 10)
        r["output_lines"] = code.count("\n")
        if i == 0:
            print(f"  rewrite warmup: {r['wall_ms']:.0f} ms ok={r['rewrite_ok']}", flush=True)
            continue  # first request discarded, same as run.py's cold-shape priming
        samples.append({**r, "content_sha256": sha256(body)})
        print(f"  rewrite run{i}: wall {r['wall_ms']:.0f} ms, {r['completion_tokens']} tok, decode {r['decode_tps']}, ok={r['rewrite_ok']}", flush=True)
    post = harness.host_snapshot()
    viol = viol + [v for v in harness.host_violations(post, args) if v not in viol]
    row = base_row(args, device, {"start": pre, "end": post}, viol, {
        "kind": "file_rewrite", "context_tier": "coding-rewrite", "concurrency": 1, "runs": args.runs,
        "max_tokens": args.rewrite_max_tokens, "temperature": 0,
        "corpus": {"repo": "Local-AI-Chat", "commit": corpus_commit, "files": [REWRITE_FILE], "sha256": sha256(src)},
    })
    row["metrics"] = {
        "ttft_ms": spread([s["ttft_ms"] for s in samples]),
        "decode_tps": spread([s["decode_tps"] for s in samples]),
        "wall_ms": spread([s["wall_ms"] for s in samples]),
        "completion_tokens": int(statistics.median([s["completion_tokens"] for s in samples])),
        "prompt_tokens": int(statistics.median([s["prompt_tokens"] for s in samples])),
        "tokens_per_chunk": spread([s["tokens_per_chunk"] for s in samples]),
        "reasoning_chars": [s["reasoning_chars"] for s in samples],
        "finish_reasons": [s["finish_reason"] for s in samples],
        "rewrite_ok": [s["rewrite_ok"] for s in samples],
        "output_sha256": [s["content_sha256"] for s in samples],
    }
    results.append(row)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--engine-label", required=True)
    ap.add_argument("--engine-version", default=None)
    ap.add_argument("--weights", default="mlx-community safetensors (same files for every engine)")
    ap.add_argument("--model", required=True, help="model id as the engine offers it")
    ap.add_argument("--model-id", default=None, help="canonical model id for the ledger (default: --model)")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--corpus-dir", required=True, help="Local-AI-Chat checkout the corpus is read from")
    ap.add_argument("--workloads", default="cached_prefix,file_rewrite")
    ap.add_argument("--prefix-sizes", default="prefix9k,prefix18k")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--followup-max-tokens", type=int, default=128)
    ap.add_argument("--rewrite-max-tokens", type=int, default=4096)
    ap.add_argument("--extra-json", default="{}", help="extra request fields, JSON")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--max-load", type=float, default=8.0)
    ap.add_argument("--min-free-pct", type=float, default=35.0)
    ap.add_argument("--results-dir", default=str(HERE.parent / "results"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = harness.device_fingerprint()
    corpus_commit = subprocess.run(["git", "-C", args.corpus_dir, "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip() or None
    client = Client(args.base, args.api_key, args.timeout, json.loads(args.extra_json))
    out = Path(args.results_dir) / f"{dt.date.today().isoformat()}-{device['model'].replace(',', '-')}-coding-{uuid.uuid4().hex[:8]}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for wl in [w.strip() for w in args.workloads.split(",") if w.strip()]:
        fn = {"cached_prefix": cached_prefix, "file_rewrite": file_rewrite}[wl]
        print(f"== {args.engine_label} / {args.model} / {wl}", flush=True)
        try:
            rows = fn(args, client, device, corpus_commit)
        except Exception as error:  # noqa: BLE001 — record the failure as a row, never drop it
            rows = [base_row(args, device, harness.host_snapshot(), [f"error: {error}"], {"kind": wl})]
            rows[0]["metrics"] = {"error": str(error)}
            print(f"  ERROR {error}", flush=True)
        with out.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                n += 1
    print(f"wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
