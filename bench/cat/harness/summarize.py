#!/usr/bin/env python3
"""Summarize raw mlxcat receipts without inventing percentiles."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def median(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def range_text(values: list[float | int | None], digits: int = 3) -> str:
    present = [float(value) for value in values if value is not None]
    if not present:
        return "unavailable"
    return f"{min(present):.{digits}f}–{max(present):.{digits}f}"


def fmt(value: float | None, digits: int = 3) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("summary.md"))
    arguments = parser.parse_args()
    payload = json.loads(arguments.results.read_text())
    rows = payload["results"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["case"])].append(row)

    lines = [
        "# Measured summary",
        "",
        "Medians and ranges are shown with sample counts. Cache states come only from server-reported cached-token evidence; request roles are recorded separately. No p95 is inferred from three exact replays.",
        "",
        "| Model | Case | State | n | Pass | Prompt tok | Cached tok median | TTFT median s (range) | Prefill tok/s median | Decode tok/s median | Total median s | Peak RSS GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, case), case_rows in sorted(grouped.items()):
        if case.startswith("coding_"):
            continue
        for state in sorted({str(row["cache_state"]) for row in case_rows}):
            selected = [row for row in case_rows if row["cache_state"] == state]
            if not selected:
                continue
            peak = max((row["server_peak_rss_bytes"] or 0) for row in selected) / (1024**3)
            ttft = [row["server_ttft_seconds"] for row in selected]
            lines.append(
                "| " + " | ".join([
                    model,
                    case,
                    state,
                    str(len(selected)),
                    f"{sum(bool(row['passed']) for row in selected)}/{len(selected)}",
                    str(selected[0]["prompt_tokens"]),
                    fmt(median([row["cached_prompt_tokens"] for row in selected]), 0),
                    f"{fmt(median(ttft))} ({range_text(ttft)})",
                    fmt(median([row["prompt_tokens_per_second"] for row in selected]), 1),
                    fmt(median([row["decode_tokens_per_second"] for row in selected]), 1),
                    fmt(median([row["server_total_seconds"] for row in selected])),
                    f"{peak:.2f}",
                ]) + " |"
            )

    coding = [row for row in rows if row["case"].startswith("coding_")]
    lines.extend(["", "## Coding", ""])
    if coding:
        passed = sum(bool(row["passed"]) for row in coding)
        lines.append(f"Pass rate: **{passed}/{len(coding)} ({passed / len(coding) * 100:.0f}%)**.")
        lines.extend(["", "| Task | Result | Time s | Detail |", "|---|---:|---:|---|"])
        for row in coding:
            lines.append(
                f"| {row['case'].removeprefix('coding_')} | {'PASS' if row['passed'] else 'FAIL'} | "
                f"{fmt(row['client_total_seconds'])} | {str(row['detail']).replace('|', '/')} |"
            )
    else:
        lines.append("No coding cases completed.")

    arguments.output.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
