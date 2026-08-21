#!/usr/bin/env python3
"""Render LEADERBOARD.md from bench/results/*.jsonl (schema mlxcat-bench/1).

Rules:
  * only records with valid_for_leaderboard == true are ranked;
  * for each (platform, device chip, model, context tier, concurrency, engine)
    the NEWEST valid record wins — re-running replaces, it never accumulates;
  * rows are stratified by platform → device → model → context; engines are the
    rows, never the columns, so a new engine is one more line, not a schema change;
  * `--check` re-renders and fails if LEADERBOARD.md on disk differs (CI gate);
  * the smoke model (role=smoke) is rendered in its own "wire-format smoke" section
    so nobody quotes it as a performance number.

Usage:
  python3 bench/leaderboard.py                # write LEADERBOARD.md
  python3 bench/leaderboard.py --check        # exit 1 if LEADERBOARD.md is stale
  python3 bench/leaderboard.py --stdout       # print instead of writing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_DIR = HERE / "results"
OUTPUT = REPO / "LEADERBOARD.md"
MATRIX = HERE / "matrix.json"

PLATFORM_LABEL = {"macos": "macOS", "ios": "iOS", "ipados": "iPadOS"}


def load_records(results_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({error})")
            if record.get("schema") != "mlxcat-bench/1":
                raise SystemExit(f"{path}:{line_no}: unknown schema {record.get('schema')!r}")
            record["_source"] = f"{path.name}:{line_no}"
            records.append(record)
    return records


def med(block: Optional[Dict[str, Any]]) -> Optional[float]:
    if not block:
        return None
    return block.get("median")


def fmt(value: Optional[float], digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "—"
    if digits == 0:
        return f"{value:,.0f}{unit}"
    return f"{value:,.{digits}f}{unit}"


def gib(value: Optional[int]) -> str:
    return "—" if not value else f"{value / 2**30:.2f}"


def parsed_ts(record: Dict[str, Any]) -> float:
    """Timestamp as epoch seconds — local-offset ISO strings must not beat UTC lexically."""
    import datetime as _dt
    raw = str(record.get("timestamp", ""))
    try:
        parsed = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)  # naive → UTC, machine-independent
    return parsed.timestamp()


def device_label(device: Dict[str, Any]) -> str:
    """Device identity: chip alone is ambiguous (two 'M4 Max' Macs with different RAM)."""
    chip = device.get("chip") or "unknown"
    mem = device.get("memory_bytes")
    model = device.get("model")
    parts = [chip]
    if mem:
        parts.append(f"{round(mem / 2**30)} GB")
    if model and model != chip:
        parts.append(model)  # Mac17,6 vs a Studio with the same chip+RAM are different machines
    return " · ".join(parts)


def newest_per_key(records: Iterable[Dict[str, Any]]) -> Dict[Tuple, Dict[str, Any]]:
    chosen: Dict[Tuple, Dict[str, Any]] = {}
    for record in records:
        if not record.get("valid_for_leaderboard"):
            continue
        key = (
            record["platform"],
            device_label(record["device"]),
            record["model"]["id"],
            record["workload"]["context_tier"],
            int(record["workload"].get("max_tokens") or 0),
            int(record["workload"]["concurrency"]),
            record["engine"].get("transport") or "http",
            record["engine"]["name"],
        )
        current = chosen.get(key)
        if current is None or parsed_ts(record) > parsed_ts(current):
            chosen[key] = record
    return chosen


def bold_best(rows: List[Dict[str, Any]], field: str, higher_is_better: bool) -> Dict[str, bool]:
    """Best per TRANSPORT group — an in-process row must never be bolded as beating HTTP rows."""
    marks: Dict[Tuple[str, str], bool] = {}
    groups: Dict[str, Dict[str, float]] = {}
    for r in rows:
        transport = r["engine"].get("transport") or "http"
        value = med(r["metrics"].get(field))
        if value is not None:
            groups.setdefault(transport, {})[r["engine"]["name"]] = value
    for transport, clean in groups.items():
        if len(clean) < 2:
            continue
        best = max(clean, key=clean.get) if higher_is_better else min(clean, key=clean.get)
        marks[(transport, best)] = True
    return marks


def render(records: List[Dict[str, Any]], matrix: Dict[str, Any]) -> str:
    roles = {m["id"]: m.get("role", "") for m in matrix.get("models", [])}
    chosen = newest_per_key(records)
    total_valid = len(chosen)
    invalid = sum(1 for r in records if not r.get("valid_for_leaderboard"))

    out: List[str] = []
    out.append("# mlxcat leaderboard")
    out.append("")
    out.append(
        "Same client, same transport (`/v1/chat/completions`, streaming, greedy), same "
        "prompt corpus, same `mlx-community` weights unless a row says otherwise. "
        "Generated by `bench/leaderboard.py` from `bench/results/*.jsonl` — **do not edit by hand**; "
        "CI fails if this file is stale. Rows are the newest valid measurement per "
        "(device, model, context, concurrency, engine). Engines that cannot run on a platform "
        "simply have no row there."
    )
    out.append("")
    out.append(f"_{total_valid} ranked cells · {invalid} recorded-but-invalid rows (loaded host / errors) kept in `bench/results/` for audit._")
    out.append("")
    out.append("Metrics: **TTFT** time to first visible token (ms, median) · **prefill** prompt tokens ÷ TTFT · "
               "**decode** (completion−1) ÷ time after first token · **peak** sampled physical footprint of the engine "
               "process during the measured runs (GiB) · **agg** aggregate tok/s across N simultaneous streams. "
               "Spread (min–max over runs) lives in the JSONL.")
    out.append("")

    if not chosen:
        out.append("_No valid rows yet. Run `python3 bench/run.py` on a quiet machine._")
        return "\n".join(out) + "\n"

    # platform → device → model → tier → [records]
    tree: Dict[str, Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    conc: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, record in chosen.items():
        platform, device, model, tier, _max_tokens, width, _transport, _engine = key
        if width == 1:
            tree[platform][device][model][tier].append(record)
        else:
            conc[platform][device][model].append(record)

    def model_sort(model_id: str) -> Tuple[int, str]:
        return (1 if roles.get(model_id) == "smoke" else 0, model_id)

    for platform in sorted(tree.keys() | conc.keys()):
        out.append(f"## {PLATFORM_LABEL.get(platform, platform)}")
        out.append("")
        devices = sorted(set(tree[platform].keys()) | set(conc[platform].keys()))
        for device in devices:
            out.append(f"### {device}")
            out.append("")
            models = sorted(set(tree[platform][device].keys()) | set(conc[platform][device].keys()), key=model_sort)
            for model in models:
                role = roles.get(model, "")
                heading = f"#### {model}"
                if role == "smoke":
                    heading += " — _wire-format smoke model; not a performance claim_"
                elif role:
                    heading += f" — _{role}_"
                out.append(heading)
                out.append("")
                tiers = tree[platform][device].get(model, {})
                if tiers:
                    out.append("| context | gen tok | engine | transport | version | prompt tok | TTFT ms | prefill tok/s | decode tok/s | peak GiB | measured |")
                    out.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|---|")
                    tier_order = [t["name"] for t in matrix.get("context_tiers", [])]
                    for tier in sorted(tiers.keys(), key=lambda t: tier_order.index(t) if t in tier_order else 99):
                        rows = sorted(
                            tiers[tier],
                            key=lambda r: (r["engine"].get("transport") or "http", -(med(r["metrics"].get("decode_tps")) or 0), r["engine"]["name"]),
                        )
                        best_decode = bold_best(rows, "decode_tps", True)
                        best_prefill = bold_best(rows, "prefill_tps", True)
                        best_ttft = bold_best(rows, "ttft_ms", False)
                        for record in rows:
                            m = record["metrics"]
                            name = record["engine"]["name"]
                            transport = record["engine"].get("transport") or "http"
                            decode = fmt(med(m.get("decode_tps")))
                            prefill = fmt(med(m.get("prefill_tps")), 0)
                            ttft = fmt(med(m.get("ttft_ms")), 0)
                            if best_decode.get((transport, name)):
                                decode = f"**{decode}**"
                            if best_prefill.get((transport, name)):
                                prefill = f"**{prefill}**"
                            if best_ttft.get((transport, name)):
                                ttft = f"**{ttft}**"
                            weights_note = ""
                            if record["engine"].get("weights") and "same files" not in record["engine"]["weights"]:
                                weights_note = " ⚠︎"
                            out.append(
                                f"| {tier} | {record['workload'].get('max_tokens', '—')} | {name}{weights_note} | {transport} | {record['engine'].get('version') or '—'} | "
                                f"{m.get('prompt_tokens', '—')} | {ttft} | {prefill} | {decode} | "
                                f"{gib(m.get('peak_phys_footprint_bytes'))} | {record['timestamp'][:10]} |"
                            )
                    out.append("")
                conc_rows = conc[platform][device].get(model, [])
                if conc_rows:
                    by_tier: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for record in conc_rows:
                        by_tier[record["workload"]["context_tier"]].append(record)
                    for tier_name in sorted(by_tier):
                        rows_for_tier = by_tier[tier_name]
                        by_engine: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
                        widths = set()
                        for record in rows_for_tier:
                            width = int(record["workload"]["concurrency"])
                            widths.add(width)
                            by_engine[record["engine"]["name"]][width] = record
                        width_list = sorted(widths)
                        out.append(f"Concurrency (`{tier_name}` prompt, aggregate tok/s across N streams, median of runs):")
                        out.append("")
                        out.append("| engine | " + " | ".join(f"×{w}" for w in width_list) + " | peak GiB @max |")
                        out.append("|---|" + "---:|" * len(width_list) + "---:|")
                        for name in sorted(by_engine):
                            cells = []
                            for w in width_list:
                                r = by_engine[name].get(w)
                                cells.append(fmt(med(r["metrics"].get("aggregate_tps"))) if r else "—")
                            last = by_engine[name].get(width_list[-1])
                            out.append(f"| {name} | " + " | ".join(cells) + f" | {gib(last['metrics'].get('peak_phys_footprint_bytes')) if last else '—'} |")
                        out.append("")
        out.append("")

    out.append("⚠︎ = different weight artifacts than the `mlx-community` safetensors the other rows use (e.g. Ollama library, GGUF) — compare quality class, not bits.")
    out.append("")
    out.append("## How to add a row")
    out.append("")
    out.append("```bash")
    out.append("swift build -c release --product mlxcat-http            # once")
    out.append("python3 bench/run.py --engines mlxcat,omlx --model-set default   # quiet machine required")
    out.append("python3 bench/leaderboard.py                             # re-render this file")
    out.append("```")
    out.append("")
    out.append("See `bench/README.md` for the schema, the quiet-machine guard, and how iOS rows are produced.")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true", help="fail if the output file is stale")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(argv)

    matrix = json.loads(MATRIX.read_text(encoding="utf-8")) if MATRIX.exists() else {}
    records = load_records(Path(args.results_dir))
    text = render(records, matrix)
    output = Path(args.output)
    if args.stdout:
        sys.stdout.write(text)
        return 0
    if args.check:
        current = output.read_text(encoding="utf-8") if output.exists() else ""
        if current != text:
            print(f"{output.name} is stale — run: python3 bench/leaderboard.py", file=sys.stderr)
            return 1
        print(f"{output.name} is up to date ({len(records)} records)")
        return 0
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
