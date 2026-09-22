#!/usr/bin/env python3
"""Render GAP-TABLE.md: per workload cell, the best engine vs mlxcat, with ledger citations.

Reads only committed evidence:
  bench/results/<day>-<device>-*.jsonl           run.py tiers + bench/coding rows (mlxcat-bench/1)
  bench/toolcalling/evidence/<day>-*.jsonl       tool-call trials (mlxcat-toolcall/1)
  <pi-dir>/summary-*.json                        one real pi task per engine

Every number printed is followed by a citation `file:line` into the ledger, so a
stranger can re-derive it. Valid rows win; a cell with only invalid rows is shown
with the guard's reason, never hidden. Levers are a first attribution from the
engine's known features and are overridable with --levers (JSON {cell: text}).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent

PERF_CELLS = [
    # (label, tier, concurrency, cache_mode, metric, higher_is_better)
    ("short c1 decode", "short", 1, "cold", "decode_tps", True),
    ("longgen c1 decode", "longgen", 1, "cold", "decode_tps", True),
    ("4k c1 TTFT (cold prefill)", "4k", 1, "cold", "ttft_ms", False),
    ("4k c1 TTFT (warm, same prompt)", "4k", 1, "warm", "ttft_ms", False),
    ("short c4 aggregate", "short", 4, "cold", "aggregate_tps", True),
    ("longgen c4 aggregate", "longgen", 4, "cold", "aggregate_tps", True),
    ("short c4 TTFT p95", "short", 4, "cold", "ttft_p95_ms", False),
]
MTP_ENGINES = {"mtplx", "ollama", "omlx-mtp"}
SPEC_LEVER = "native MTP speculative decoding"


def med(block: Any) -> Optional[float]:
    if isinstance(block, dict):
        return block.get("median")
    if isinstance(block, (int, float)):
        return float(block)
    return None


def load_rows(day: str, device: str) -> List[Tuple[str, int, Dict[str, Any]]]:
    rows = []
    for path in sorted(glob.glob(str(BENCH / "results" / f"{day}-{device}-*.jsonl"))):
        for n, line in enumerate(open(path), 1):
            line = line.strip()
            if line:
                rows.append((os.path.relpath(path, BENCH.parent), n, json.loads(line)))
    return rows


def pick(rows, engine, model, predicate, min_runs):
    """Newest valid row with enough runs; else newest row at all (flagged)."""
    cands = [(f, n, r) for f, n, r in rows if r["engine"]["name"] == engine and r["model"]["id"] == model and predicate(r)]
    good = [c for c in cands if c[2].get("valid_for_leaderboard") and (c[2]["workload"].get("runs") or 0) >= min_runs]
    pool = good or cands
    return max(pool, key=lambda c: c[2]["timestamp"]) if pool else None


def fmt(v: Optional[float], unit: str) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f} {unit}" if unit == "ms" else f"{v:,.1f} {unit}"


def cite(c) -> str:
    f, n, r = c
    flag = "" if r.get("valid_for_leaderboard") else f" ⚠invalid: {r.get('invalid_reason')}"
    return f"`{Path(f).name}:{n}`{flag}"


def lever_for(label: str, best: str, best_row: Dict[str, Any], mlx_row: Optional[Dict[str, Any]]) -> str:
    if best == "mlxcat":
        return "— mlxcat leads"
    same_weights = mlx_row and best_row["engine"].get("weights") == mlx_row["engine"].get("weights")
    parts = []
    if "decode" in label and best in MTP_ENGINES:
        parts.append(SPEC_LEVER)
    elif "warm" in label or "cached" in label or "follow-up" in label:
        parts.append("prefix cache reuse")
    elif "c4" in label:
        parts.append("scheduler admission / batched decode")
    elif "TTFT" in label or "cold" in label:
        parts.append("unknown (prefill path)")
    else:
        parts.append("unknown")
    if not same_weights:
        parts.append("quant format (different checkpoint: " + str(best_row["engine"].get("weights")) + ")")
    return "; ".join(parts)


def perf_section(rows, models, engines, min_runs, levers) -> List[str]:
    out = ["## Throughput and latency tiers (bench/run.py, N=5)", ""]
    for model in models:
        out += [f"### {model}", "",
                "| cell | best engine | best | mlxcat | gap (best ÷ mlxcat) | named lever | evidence (best · mlxcat) |",
                "|---|---|---|---|---|---|---|"]
        for label, tier, conc, mode, metric, hib in PERF_CELLS:
            pred = lambda r, tier=tier, conc=conc, mode=mode: (
                "kind" not in r["workload"] and r["workload"].get("context_tier") == tier
                and r["workload"].get("concurrency") == conc and (r["workload"].get("cache_mode") or "cold") == mode)
            picks = {e: pick(rows, e, model, pred, min_runs) for e in engines}
            picks = {e: c for e, c in picks.items() if c and med(c[2]["metrics"].get(metric)) is not None}
            if not picks:
                continue
            valid = {e: c for e, c in picks.items() if c[2].get("valid_for_leaderboard")}
            pool = valid or picks
            key = lambda e: med(pool[e][2]["metrics"][metric])
            best = max(pool, key=key) if hib else min(pool, key=key)
            unit = "ms" if metric.endswith("_ms") else "tok/s"
            bv = key(best)
            m = picks.get("mlxcat")
            mv = med(m[2]["metrics"][metric]) if m else None
            if mv and bv:
                gap = (bv / mv) if hib else (mv / bv)
                gap_s = f"{gap:.2f}×"
            else:
                gap_s = "—"
            lever = levers.get(f"{model}|{label}") or lever_for(label, best, pool[best][2], m[2] if m else None)
            out.append(f"| {label} | {best} | {fmt(bv, unit)} | {fmt(mv, unit)} | {gap_s} | {lever} | "
                       f"{cite(pool[best])} · {cite(m) if m else '—'} |")
        out.append("")
        # every engine's number, so the best is not the only thing visible
        out += [f"<details><summary>{model}: every engine per cell</summary>", ""]
        for label, tier, conc, mode, metric, hib in PERF_CELLS:
            pred = lambda r, tier=tier, conc=conc, mode=mode: (
                "kind" not in r["workload"] and r["workload"].get("context_tier") == tier
                and r["workload"].get("concurrency") == conc and (r["workload"].get("cache_mode") or "cold") == mode)
            cells = []
            for e in engines:
                c = pick(rows, e, model, pred, min_runs)
                if c and med(c[2]["metrics"].get(metric)) is not None:
                    unit = "ms" if metric.endswith("_ms") else "tok/s"
                    cells.append(f"{e} {fmt(med(c[2]['metrics'][metric]), unit)} ({cite(c)})")
            if cells:
                out.append(f"- **{label}**: " + "; ".join(cells))
        out += ["", "</details>", ""]
    return out


def coding_section(rows, models, engines, levers) -> List[str]:
    out = ["## Coding-shaped workloads (bench/coding/run_coding.py)", ""]
    specs = [
        ("cached 9K: turn-1 TTFT (cold)", "coding-prefix9k", "turn1", False),
        ("cached 9K: follow-up TTFT median (turns 2-6)", "coding-prefix9k", "follow", False),
        ("cached 18K: turn-1 TTFT (cold)", "coding-prefix18k", "turn1", False),
        ("cached 18K: follow-up TTFT median (turns 2-6)", "coding-prefix18k", "follow", False),
        ("file rewrite: decode", "coding-rewrite", "decode", True),
        ("file rewrite: wall-clock", "coding-rewrite", "wall", False),
    ]

    def value(r, what):
        m = r["metrics"]
        if what == "turn1":
            return med(m.get("ttft_ms")), "ms"
        if what == "follow":
            return m.get("followup_ttft_median_ms"), "ms"
        if what == "decode":
            return med(m.get("decode_tps")), "tok/s"
        return med(m.get("wall_ms")), "ms"

    for model in models:
        out += [f"### {model}", "",
                "| cell | best engine | best | mlxcat | gap | named lever | evidence (best · mlxcat) |",
                "|---|---|---|---|---|---|---|"]
        for label, tier, what, hib in specs:
            picks = {}
            for e in engines:
                c = pick(rows, e, model, lambda r, tier=tier: r["workload"].get("context_tier") == tier and "error" not in r["metrics"], 1)
                if c and value(c[2], what)[0] is not None:
                    picks[e] = c
            if not picks:
                continue
            valid = {e: c for e, c in picks.items() if c[2].get("valid_for_leaderboard")}
            pool = valid or picks
            key = lambda e: value(pool[e][2], what)[0]
            best = max(pool, key=key) if hib else min(pool, key=key)
            bv, unit = value(pool[best][2], what)
            m = picks.get("mlxcat")
            mv = value(m[2], what)[0] if m else None
            gap_s = f"{((bv / mv) if hib else (mv / bv)):.2f}×" if (bv and mv) else "—"
            lever = levers.get(f"{model}|{label}") or lever_for(label if what != "follow" else "follow-up " + label, best, pool[best][2], m[2] if m else None)
            out.append(f"| {label} | {best} | {fmt(bv, unit)} | {fmt(mv, unit)} | {gap_s} | {lever} | {cite(pool[best])} · {cite(m) if m else '—'} |")
        out.append("")
        reuse = []
        for e in engines:
            for tier in ("coding-prefix9k", "coding-prefix18k"):
                c = pick(rows, e, model, lambda r, tier=tier: r["workload"].get("context_tier") == tier and "error" not in r["metrics"], 1)
                if c:
                    m = c[2]["metrics"]
                    ratio = m.get("reuse_ratio")
                    reuse.append(f"{e} {tier[7:]}: follow-up/turn-1 = {ratio:.2f}" if ratio is not None else f"{e} {tier[7:]}: —")
        rewrite = []
        for e in engines:
            c = pick(rows, e, model, lambda r: r["workload"].get("context_tier") == "coding-rewrite" and "error" not in r["metrics"], 1)
            if c:
                oks = c[2]["metrics"].get("rewrite_ok") or []
                rewrite.append(f"{e} {sum(oks)}/{len(oks)} correct")
        if reuse:
            out += ["Prefix reuse (follow-up TTFT ÷ turn-1 TTFT; < 0.5 = the engine reused the prefix): " + "; ".join(reuse), ""]
        if rewrite:
            out += ["Rewrite correctness (renames applied, nothing else lost): " + "; ".join(rewrite), ""]
    return out


def toolcall_section(day: str) -> List[str]:
    stats = collections.defaultdict(lambda: collections.Counter())
    files = collections.defaultdict(set)
    for path in sorted(glob.glob(str(BENCH / "toolcalling" / "evidence" / f"{day}-*.jsonl"))):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("engine"), r.get("model"), r.get("scenario"))
            s = stats[key]
            s["n"] += 1
            s["ok"] += bool(r.get("ok"))
            raw = r.get("raw_content") or ""
            if not r.get("called") and ("<tool_call" in raw or "<function" in raw or '"arguments"' in raw):
                s["leak"] += 1
            if r.get("error"):
                s["error"] += 1
            files[(r.get("engine"), r.get("model"))].add(Path(path).name)
    if not stats:
        return []
    out = ["## Tool-call correctness (bench/toolcalling/run.py, arm `default`, t∈{0, 0.7} × 3 trials × stream/non-stream)", "",
           "| engine | model | scenario | pass | leaked as text | errors | evidence |", "|---|---|---|---|---|---|---|"]
    for (e, m, sc), s in sorted(stats.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out.append(f"| {e} | {m} | {sc} | {s['ok']}/{s['n']} | {s['leak']} | {s['error']} | {', '.join(sorted(files[(e, m)]))} |")
    return out + [""]


def pi_section(pi_dir: Optional[str]) -> List[str]:
    if not pi_dir:
        return []
    rows = []
    for path in sorted(glob.glob(os.path.join(pi_dir, "summary-*.json"))):
        s = json.load(open(path))
        rows.append(f"| {s['label']} | {s['model']} | {'PASS' if s['passed'] else 'FAIL'} | {s.get('turns')} | "
                    f"{s['wall_s']} s | {s.get('tool_calls')} | {'timeout' if s.get('timed_out') else s.get('pi_exit')} | "
                    f"`pi/pi-{s['label']}.jsonl`, `pi/diff-{s['label']}.patch` |")
    if not rows:
        return []
    return ["## One real pi task per engine (fixture: Swift `slugify` package, 2 failing tests)", "",
            "| engine | model | verdict | turns | wall | tool calls | pi exit | transcript |", "|---|---|---|---|---|---|---|---|"] + rows + [""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--device", default="Mac17-6")
    ap.add_argument("--models", default="Qwen3.8-27B-4bit,Qwen3-Coder-30B-A3B-Instruct-4bit")
    ap.add_argument("--engines", default="mlxcat,mtplx,ollama,lmstudio,llama-cpp,mlx-lm,omlx,omlx-mtp,rapid-mlx,localai-app-a218,mlxcat-defaults")
    ap.add_argument("--min-runs", type=int, default=5)
    ap.add_argument("--pi-dir", default=None)
    ap.add_argument("--levers", default=None, help="JSON file {\"model|cell\": \"lever text\"}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load_rows(args.day, args.device)
    models = [m for m in args.models.split(",") if m]
    engines = [e for e in args.engines.split(",") if e]
    levers = json.load(open(args.levers)) if args.levers else {}
    lines = ["# GAP-TABLE — mlxcat vs the best local engine on this Mac", "",
             f"Generated by `bench/coding/gap_table.py --day {args.day}` from the ledger; do not edit numbers by hand. "
             "Gap > 1 means the best engine is that many times better than mlxcat (faster decode, or lower latency). "
             "Only valid rows compete for 'best' when any exist; ⚠invalid marks a row the quiet-machine guard rejected.", ""]
    lines += perf_section(rows, models, engines, args.min_runs, levers)
    lines += coding_section(rows, models, engines, levers)
    lines += toolcall_section(args.day)
    lines += pi_section(args.pi_dir)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} from {len(rows)} ledger rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
