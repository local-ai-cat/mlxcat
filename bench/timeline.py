#!/usr/bin/env python3
"""Build the mlxcat-vs-competitors timeline that `bench/dashboard.html` renders.

The dashboard exists because every number in this repo has been argued from a
snapshot — one board, one pass — and a snapshot cannot show whether a change
helped. A trajectory can. This reduces `bench/results/*.jsonl` to one JSON blob:
for every (model, tier, concurrency) cell, mlxcat's value at each PASS, next to
the best competitor's value on the same machine.

Rules that keep it honest:

  * One device only. Rows are filtered to `Mac16,7` (the M4 Pro that ran the
    competitors); an M5 Max row next to an M4 Pro row is not a comparison.
  * A pass is a RESULT FILE, not a date — 2026-08-23 holds two passes, and
    collapsing them by date would average the before and after of the day's
    biggest change.
  * A/B arms (`mlxcat-cache-held`, `mlxcat-moe-uncapped`, `mlxcat-batched`,
    `mlxcat-inprocess`) are ours and are NOT competitors. Scoring against our own
    arms is the self-referential-gate trap this repo has already fallen into
    once.
"""
import collections
import json
import os
import glob

DEVICE = "Mac16,7"
OURS = {"mlxcat", "mlxcat-batched", "mlxcat-inprocess", "mlxcat-cache-held", "mlxcat-moe-uncapped"}

# Result file -> (order, short label, what changed). Ordered by when it ran, not
# by filename: the label is what a reader needs to connect a jump to a cause.
PASSES = [
    ("2026-08-21-Mac16-7-f8a42afe.jsonl", "P1 · 08-21", "first M4 Pro rows, 24 GiB ceiling"),
    ("2026-08-22-Mac16-7-8202937b.jsonl", "P2 · 08-22", "serialization A/B"),
    ("2026-08-22-Mac16-7-f2f48ab2.jsonl", "P3 · 08-22", "re-measure default"),
    ("2026-08-23-Mac16-7-97a15e73.jsonl", "P4 · 08-23", "clearCache between prefill chunks, split chunk widths, width ceilings, gemma text rows batch"),
    ("2026-08-23-Mac16-7-67c916d5.jsonl", "P5 · 08-23", "qwen3_5 batches, MLX buffer cache released at idle"),
]

METRICS = {
    # key -> (json path, lower_is_better, unit, label)
    "ttft": ("ttft_ms", True, "ms", "TTFT (median)"),
    "e2e": ("e2e_tps", False, "tok/s", "End-to-end throughput"),
    "peak": ("peak_phys_footprint_bytes", True, "GiB", "Peak footprint"),
}


def value(metrics, key):
    field, _, unit, _ = METRICS[key]
    raw = metrics.get(field)
    if isinstance(raw, dict):
        raw = raw.get("median")
    if raw in (None, 0):
        return None
    return raw / 2**30 if unit == "GiB" else raw


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get("device") or {}).get("model") != DEVICE:
            continue
        rows.append(row)
    return rows


def cell_key(row):
    w = row["workload"]
    return (row["model"]["id"], w["context_tier"], w["concurrency"])


def build(results_dir):
    by_pass = {}
    for filename, label, note in PASSES:
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        by_pass[label] = (note, load(path))

    # Competitors are taken from wherever they were measured — they do not have
    # passes, they have one measurement each, and re-measuring them is a
    # separate exercise from tracking ourselves.
    competitors = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, "*.jsonl")):
        for row in load(path):
            name = row["engine"]["name"]
            if name in OURS:
                continue
            competitors[name][cell_key(row)] = row["metrics"]

    cells = {}
    for label, (_, rows) in by_pass.items():
        for row in rows:
            if row["engine"]["name"] != "mlxcat":
                continue
            key = cell_key(row)
            entry = cells.setdefault(
                "|".join(str(part) for part in key),
                {"model": key[0], "tier": key[1], "concurrency": key[2], "ours": {}, "them": {}},
            )
            entry["ours"][label] = {k: value(row["metrics"], k) for k in METRICS}

    for name, byCell in competitors.items():
        for key, metrics in byCell.items():
            entry = cells.get("|".join(str(part) for part in key))
            if entry is None:
                continue
            entry["them"][name] = {k: value(metrics, k) for k in METRICS}

    return {
        "device": DEVICE,
        "passes": [
            {"label": label, "note": by_pass[label][0]}
            for label, _ in [(p[1], p[0]) for p in PASSES]
            if label in by_pass
        ],
        "metrics": {k: {"unit": v[2], "label": v[3], "lowerIsBetter": v[1]} for k, v in METRICS.items()},
        "cells": cells,
    }


def render(data, here):
    """Inline the data into the dashboard so the page is self-contained."""
    template = os.path.join(here, "dashboard.template.html")
    if not os.path.exists(template):
        return None
    html = open(template).read()
    blob = json.dumps(data, sort_keys=True).replace("</", "<\\/")
    out = os.path.join(here, "dashboard.html")
    with open(out, "w") as handle:
        handle.write(html.replace("__TIMELINE_JSON__", blob))
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = build(os.path.join(here, "results"))
    data["generated"] = os.environ.get("MLXCAT_TIMELINE_STAMP", "")
    out = os.path.join(here, "timeline.json")
    with open(out, "w") as handle:
        json.dump(data, handle, indent=1, sort_keys=True)
    filled = sum(1 for c in data["cells"].values() if c["them"])
    print(f"wrote {out}: {len(data['cells'])} cells, {filled} with a competitor, "
          f"{len(data['passes'])} passes")
    page = render(data, here)
    if page:
        print(f"wrote {page}: {os.path.getsize(page) // 1024} KiB, self-contained")
