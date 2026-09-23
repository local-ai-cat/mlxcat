#!/usr/bin/env python3
"""Exactness gate for the native-MTP probe (MLXCatMTPProbe).

Compares a plain-decode run against an MTP run of the SAME checkpoint, revision,
prompts and greedy parameters. Green only when ALL hold:
  1. every prompt's token sequence is identical (MTP must be lossless at t=0);
  2. proposals > 0 and acceptances > 0 (the drafter actually ran);
  3. at least one rejection was exercised (proposed > accepted somewhere), so the
     rewind path was on the hot path, not just the happy path;
  4. no row fell back to plain decode or entered passthrough.
A gate that cannot fail is not a gate: run it against a
MLX_MTP_SABOTAGE_ACCEPT_WRONG=1 probe run and it must exit 1.

Usage: exactness_gate.py PLAIN.jsonl MTP.jsonl
"""
import json
import sys


def load(path):
    rows = {}
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            rows[(row["id"], row["rep"])] = row
    return rows


def main():
    plain, mtp = load(sys.argv[1]), load(sys.argv[2])
    failures = []
    if set(plain) != set(mtp):
        failures.append(f"prompt sets differ: {sorted(set(plain) ^ set(mtp))}")
    mismatched = []
    for key in sorted(set(plain) & set(mtp)):
        a, b = plain[key]["tokens"], mtp[key]["tokens"]
        if a != b:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            mismatched.append(f"{key[0]}@{first} (plain {len(a)} tok, mtp {len(b)} tok)")
    if mismatched:
        failures.append(f"{len(mismatched)} token-sequence mismatches: {', '.join(mismatched)}")
    proposed = sum(r["proposed"] for r in mtp.values())
    accepted = sum(r["accepted"] for r in mtp.values())
    rejecting_rows = sum(1 for r in mtp.values() if r["proposed"] > r["accepted"])
    if proposed == 0:
        failures.append("no proposals: the drafter never ran")
    if accepted == 0:
        failures.append("no acceptances")
    if rejecting_rows == 0:
        failures.append("no rejection exercised: rewind path never ran")
    fell_back = [k[0] for k, r in mtp.items() if r.get("fallback") or r.get("passthrough_reason")]
    if fell_back:
        failures.append(f"fallback/passthrough on: {fell_back}")
    sabotage = any(r.get("sabotage_armed") for r in mtp.values())
    print(f"prompts={len(mtp)} proposed={proposed} accepted={accepted} "
          f"acceptance={accepted / proposed if proposed else 0:.3f} "
          f"rows_with_rejection={rejecting_rows} sabotage_armed={sabotage}")
    if failures:
        print("GATE RED")
        for failure in failures:
            print(" -", failure)
        return 1
    print("GATE GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
