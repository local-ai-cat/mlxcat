#!/usr/bin/env python3
"""Promote iOS BenchHost rows into bench/results/ and re-render the board.

The device harness appends every night's rows into one aggregate
(`ios/BenchHost/.build/ios-rows/ios-rows.jsonl`), stamped valid or invalid by
the measured run-condition guard. This script moves them onto the board:

  python3 bench/promote_ios_rows.py                       # today's rows
  python3 bench/promote_ios_rows.py --since 2026-08-30
  python3 bench/promote_ios_rows.py --input FILE [--all]

* splits rows per device model into `bench/results/<date>-<device>-<runid>.jsonl`
  (the Mac naming scheme with the device model in the Mac-model slot)
* keeps BOTH valid and invalid rows — invalid rows are audit evidence, exactly
  like the Mac harness's loaded-host rows
* dedupes against every row already in bench/results (timestamp+engine+model+
  tier+device), so re-running after a partial promote is safe
* annotates `device.memory_bytes`/`device.chip` on rows from producers that
  predate the enriched device stamp. RAM per hardware model is a hardware
  constant, not a run condition, so this is identity annotation, never
  measurement; rows from current producers arrive with these fields measured
  on-device and are left alone.
* re-renders LEADERBOARD.md / timeline.json / dashboard.html, honouring the
  CI gate that the board must derive from results
"""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "bench", "results")
DEFAULT_INPUT = os.path.join(
    REPO, "ios", "BenchHost", ".build", "ios-rows", "ios-rows.jsonl")

# Hardware constants for producers that predate the enriched device stamp.
KNOWN_DEVICE_MEMORY = {
    "iPhone17,1": 8 << 30,   # iPhone 16 Pro
    "iPhone17,2": 8 << 30,   # iPhone 16 Pro Max
    "iPhone18,1": 12 << 30,  # iPhone 17 Pro
    "iPhone18,2": 12 << 30,  # iPhone 17 Pro Max
    "iPhone18,3": 8 << 30,   # iPhone 17
}


def row_identity(row):
    return (
        row.get("timestamp"),
        (row.get("engine") or {}).get("name"),
        (row.get("model") or {}).get("id"),
        (row.get("workload") or {}).get("context_tier"),
        (row.get("device") or {}).get("model"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    # Row timestamps are UTC; a local "today" just after midnight would skip a
    # night that is still yesterday in UTC.
    ap.add_argument(
        "--since",
        default=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        help="promote rows with timestamp >= this UTC date (default: today, UTC)")
    ap.add_argument("--all", action="store_true", help="ignore --since")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing = set()
    for path in glob.glob(os.path.join(RESULTS, "*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.add(row_identity(json.loads(line)))
                    except json.JSONDecodeError:
                        pass

    per_device = {}
    skipped_old = skipped_dupe = 0
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("platform") != "ios" or row.get("schema") != "mlxcat-bench/1":
                continue
            if not args.all and (row.get("timestamp") or "") < args.since:
                skipped_old += 1
                continue
            if row_identity(row) in existing:
                skipped_dupe += 1
                continue
            device = row.setdefault("device", {})
            model = device.get("model") or "iPhone-unknown"
            if "memory_bytes" not in device and model in KNOWN_DEVICE_MEMORY:
                device["memory_bytes"] = KNOWN_DEVICE_MEMORY[model]
            device.setdefault("chip", model)
            existing.add(row_identity(row))
            per_device.setdefault(model, []).append(row)

    if not per_device:
        print(f"nothing to promote (skipped: {skipped_old} before --since, "
              f"{skipped_dupe} already in results)")
        return

    today = datetime.date.today().isoformat()
    total_valid = total_invalid = 0
    for model, rows in sorted(per_device.items()):
        valid = sum(1 for r in rows if r.get("valid_for_leaderboard"))
        total_valid += valid
        total_invalid += len(rows) - valid
        name = f"{today}-{model.replace(',', '-')}-{uuid.uuid4().hex[:8]}.jsonl"
        path = os.path.join(RESULTS, name)
        print(f"{model}: {valid} valid + {len(rows) - valid} invalid -> {name}")
        if not args.dry_run:
            with open(path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"promoted {total_valid} valid, {total_invalid} invalid (audit) rows; "
          f"skipped {skipped_old} old + {skipped_dupe} duplicate")
    if args.dry_run:
        return
    for script in ("leaderboard.py", "timeline.py"):
        subprocess.run([sys.executable, os.path.join(REPO, "bench", script)],
                       check=True, cwd=REPO)


if __name__ == "__main__":
    main()
