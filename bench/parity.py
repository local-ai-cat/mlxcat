#!/usr/bin/env python3
"""Gate mlxcat against the OTHER engines, not against its own past.

Every existing gate in this repo is self-referential. `MEMORY_BUDGETS` says so
outright — it pins "today's measured behaviour ... so the cliff cannot silently
worsen" — and `test_schedulerLevelConcurrencyScales` compares us to us. Both
ratchet against regression and neither can ever fail because a competitor is
faster or leaner. That is how the suite stayed green while mlxcat won 6 of 47
cells on aggregate throughput and 8 of 47 on TTFT.

This reads the committed `bench/results/*.jsonl` — no benchmarking, no weights,
no Metal, so it runs in the hosted Linux job next to the leaderboard check — and
compares each mlxcat cell against the BEST other engine measured on the same
device, model, context tier, cache mode and concurrency:

    ttft   ratio = best_other / ours     (higher is better; 1.0 is parity)
    e2e    ratio = ours / best_other     (higher is better)
    peak   ratio = best_other / ours     (higher is better; we want to use less)

`--update` writes the current ratios to bench/parity-baseline.json. `--check`
fails when a ratio falls more than `--tolerance` below its baseline, so closing a
gap locks the gain in and reopening one is a red build.

The baseline records where we STAND, which is mostly under 1.0. That is
deliberate: a bar set at parity would be red everywhere and get ignored, and a
gate everyone ignores is worse than no gate. The number to move is the count of
cells at or above 1.0, which this prints on every run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"
BASELINE = REPO / "bench" / "parity-baseline.json"

OURS = "mlxcat"
# Arms of our own binary that differ only by an env override. They are A/B arms,
# never opponents — scoring against them would let us "reach parity" by making a
# flag slower.
#
# The membership test is a PREFIX, not this set. The set was a hand-maintained
# list and it went stale the day `mlxcat-cache-held` was added: for one board
# our own control arm was scored as a rival engine, which quietly moved the
# parity counts (throughput 23/57 -> 14/57). A list that has to be updated in
# lockstep with a new arm will be forgotten again; a prefix cannot be.
OUR_ARMS = {
    "mlxcat",
    "mlxcat-defaults",
    "mlxcat-batched",
    "mlxcat-moe-uncapped",
    "mlxcat-cache-held",
    "mlxcat-inprocess",
}


def is_ours(engine_name: str) -> bool:
    """Every arm of our own binary, however it was named."""
    return engine_name == OURS or engine_name.startswith("mlxcat-")


def med(block: Optional[Dict[str, Any]]) -> Optional[float]:
    if not block:
        return None
    return block.get("median") if isinstance(block, dict) else block


def load(results_dir: Path) -> list[Dict[str, Any]]:
    records = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({error})")
            if not record.get("valid_for_leaderboard"):
                continue
            records.append(record)
    return records


def cell_key(record: Dict[str, Any]) -> Tuple:
    workload = record["workload"]
    device = record["device"]
    return (
        f"{device.get('chip', '?')} {device.get('model', '?')}",
        record["model"]["id"],
        workload["context_tier"],
        workload.get("cache_mode", "cold"),
        int(workload["concurrency"]),
    )


def newest_per_engine(records: list[Dict[str, Any]]) -> Dict[Tuple, Dict[str, Dict[str, Any]]]:
    cells: Dict[Tuple, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        engine = record["engine"]["name"]
        bucket = cells.setdefault(cell_key(record), {})
        current = bucket.get(engine)
        if current is None or record["timestamp"] > current["timestamp"]:
            bucket[engine] = record
    return cells


def ratios(cells: Dict[Tuple, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, by_engine in cells.items():
        ours = by_engine.get(OURS)
        others = {name: rec for name, rec in by_engine.items() if not is_ours(name)}
        if ours is None or not others:
            continue
        our_metrics = ours["metrics"]
        entry: Dict[str, float] = {}

        our_ttft = med(our_metrics.get("ttft_ms"))
        their_ttft = [t for t in (med(r["metrics"].get("ttft_ms")) for r in others.values()) if t]
        if our_ttft and their_ttft:
            entry["ttft"] = round(min(their_ttft) / our_ttft, 4)

        our_e2e = med(our_metrics.get("e2e_tps"))
        their_e2e = [t for t in (med(r["metrics"].get("e2e_tps")) for r in others.values()) if t]
        if our_e2e and their_e2e:
            entry["e2e"] = round(our_e2e / max(their_e2e), 4)

        our_peak = our_metrics.get("peak_phys_footprint_bytes")
        their_peak = [p for p in (r["metrics"].get("peak_phys_footprint_bytes") for r in others.values()) if p]
        if our_peak and their_peak:
            entry["peak"] = round(min(their_peak) / our_peak, 4)

        if entry:
            out["|".join(str(part) for part in key)] = entry
    return out


def summarize(current: Dict[str, Dict[str, float]]) -> str:
    lines = []
    for metric in ("ttft", "e2e", "peak"):
        values = [cell[metric] for cell in current.values() if metric in cell]
        if not values:
            continue
        at_parity = sum(1 for value in values if value >= 1.0)
        lines.append(
            f"  {metric:5} parity-or-better in {at_parity}/{len(values)} cells "
            f"(worst {min(values):.2f}x, best {max(values):.2f}x)"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on regression against the baseline")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline from current results")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="fractional slip allowed before a cell fails (default 0.10). TTFT medians move a few percent run to run on a shared host; a gate that cries wolf gets ignored, which is the failure mode this file exists to avoid.")
    args = parser.parse_args()

    current = ratios(newest_per_engine(load(RESULTS)))
    if not current:
        print("no cells where mlxcat and another engine were measured together — nothing to compare")
        return 0

    print(f"mlxcat vs best-other, {len(current)} comparable cells:")
    print(summarize(current))

    if args.update:
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {BASELINE.relative_to(REPO)}")
        return 0

    if not args.check:
        return 0

    if not BASELINE.exists():
        raise SystemExit(f"{BASELINE} is missing — run `python3 bench/parity.py --update` first")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    failures = []
    for cell, metrics in sorted(baseline.items()):
        have = current.get(cell)
        if have is None:
            # A cell can disappear legitimately (a model leaves the matrix). Say
            # so rather than failing, so a matrix edit is not a broken build.
            print(f"note: no current measurement for {cell}")
            continue
        for metric, was in metrics.items():
            now = have.get(metric)
            if now is None:
                continue
            if now < was * (1 - args.tolerance):
                failures.append(f"{cell} [{metric}] {was:.2f}x -> {now:.2f}x vs best-other")

    if failures:
        print("\nmlxcat lost ground against other engines:")
        for failure in failures:
            print(f"  {failure}")
        print("\nIf the loss is intended, re-run with --update and say why in the commit.")
        return 1

    print("\nno cell lost more than "
          f"{args.tolerance:.0%} against the best other engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
