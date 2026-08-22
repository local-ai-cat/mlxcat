#!/usr/bin/env python3
"""Tests for the harness's own safeguards.

These exist because the failure they guard against is not a bad number — it is a
dead machine. On 2026-08-22 vllm-mlx panicked the macOS GPU driver on the M4 Pro
worker; the reboot took the three benchmark passes queued behind it, and the box
did not come back on its own. The guards below are what the harness learned from
that, so they get tests rather than trust.

    python3 bench/test_run.py            # or: python3 -m unittest discover bench
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

MACOS = sys.platform == "darwin"


class SwapGrowthTests(unittest.TestCase):
    """Absolute swap is not portable — a laptop idles with 20 GiB paged out and
    the worker idles at zero — so the guard reads growth from a baseline."""

    def violations(self, baseline, current, limit_gib):
        snapshot = {
            "loadavg_1m": 0.1,
            "memory_free_pct": 99,
            "swap_used_bytes": current,
            "thermal_cpu_speed_limit": 100,
        }
        return run.quiet_machine_violations(
            snapshot, max_load=8, min_free_pct=35,
            swap_baseline_bytes=baseline,
            max_swap_growth_bytes=int(limit_gib * 2 ** 30),
        )

    def test_high_absolute_swap_is_not_a_violation(self):
        gib = 2 ** 30
        self.assertEqual(self.violations(20 * gib, 20 * gib, 2), [])

    def test_growth_past_the_limit_is_a_violation(self):
        gib = 2 ** 30
        problems = self.violations(20 * gib, 23 * gib, 2)
        self.assertEqual(len(problems), 1)
        self.assertIn("swap grew 3.0 GiB", problems[0])

    def test_growth_under_the_limit_is_not(self):
        gib = 2 ** 30
        self.assertEqual(self.violations(0, 1 * gib, 2), [])

    def test_no_baseline_means_no_swap_check(self):
        self.assertEqual(self.violations(None, 99 * 2 ** 30, 2), [])


@unittest.skipUnless(MACOS, "proc_pid_rusage is macOS-only")
class RunawayGuardTests(unittest.TestCase):
    """The guard must actually kill. A guard that only logs is what we already
    had: `FootprintSampler` watched the footprint climb to 35.8 GiB and did
    nothing, because measuring was all it was for."""

    HOG = (
        "import time\n"
        "blocks = []\n"
        "for _ in range(60):\n"
        "    blocks.append(bytearray(20 * 1024 * 1024))\n"
        "    time.sleep(0.05)\n"
        "time.sleep(30)\n"
    )

    def test_kills_a_process_over_the_footprint_cap(self):
        child = subprocess.Popen([sys.executable, "-c", self.HOG])
        self.addCleanup(child.kill)
        guard = run.RunawayGuard(
            run.Footprint(), child.pid, "test",
            cap_bytes=100 * 1024 * 1024,
            swap_baseline_bytes=None, swap_growth_kill_bytes=None,
            interval=0.1,
        ).start()
        self.addCleanup(guard.stop)
        self.assertEqual(child.wait(timeout=30), -9, "child was not SIGKILLed")
        self.assertIn("exceeded the cap", guard.breach or "")

    def test_kills_on_host_swap_growth_even_when_the_engine_looks_small(self):
        """The host can be dying without this one process looking guilty — the
        kill signal is the machine's state, not only the engine's."""
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(child.kill)
        baseline = 0
        original = run.swap_used_bytes
        run.swap_used_bytes = lambda: 9 * 2 ** 30
        self.addCleanup(lambda: setattr(run, "swap_used_bytes", original))
        guard = run.RunawayGuard(
            run.Footprint(), child.pid, "test",
            cap_bytes=None,
            swap_baseline_bytes=baseline,
            swap_growth_kill_bytes=8 * 2 ** 30,
            interval=0.1,
        ).start()
        self.addCleanup(guard.stop)
        self.assertEqual(child.wait(timeout=15), -9, "child was not SIGKILLed")
        self.assertIn("host swapped", guard.breach or "")

    def test_leaves_a_well_behaved_process_alone(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        self.addCleanup(child.kill)
        guard = run.RunawayGuard(
            run.Footprint(), child.pid, "test",
            cap_bytes=8 * 2 ** 30,
            swap_baseline_bytes=0, swap_growth_kill_bytes=64 * 2 ** 30,
            interval=0.1,
        ).start()
        self.addCleanup(guard.stop)
        self.assertEqual(child.wait(timeout=15), 0)
        self.assertIsNone(guard.breach)


class ResumeIndexTests(unittest.TestCase):
    """`--resume` is the other half of surviving a panic: the cells already paid
    for are not paid for twice."""

    def write(self, directory, **overrides):
        row = {
            "device": {"model": "Mac16,7"},
            "engine": {"name": "mlxcat"},
            "model": {"id": "Qwen3.5-4B-MLX-4bit"},
            "workload": {"context_tier": "short", "concurrency": 1, "cache_mode": "cold", "max_tokens": 128},
            "valid_for_leaderboard": True,
        }
        for key, value in overrides.items():
            if isinstance(value, dict):
                row[key] = {**row[key], **value}
            else:
                row[key] = value
        with open(Path(directory) / "rows.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def test_indexes_a_good_row(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory)
            self.assertEqual(
                run.recorded_cells(Path(directory), "Mac16,7"),
                {("mlxcat", "Qwen3.5-4B-MLX-4bit", "short", 1, "cold", 128)},
            )

    def test_ignores_invalid_rows_and_other_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory, valid_for_leaderboard=False)
            self.write(directory, device={"model": "Mac17,6"})
            self.assertEqual(run.recorded_cells(Path(directory), "Mac16,7"), set())

    def test_cache_mode_and_width_are_part_of_the_key(self):
        """cold and warm are different measurements of the same cell; resuming
        must not let one stand in for the other."""
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory, workload={"cache_mode": "warm"})
            done = run.recorded_cells(Path(directory), "Mac16,7")
            self.assertNotIn(("mlxcat", "Qwen3.5-4B-MLX-4bit", "short", 1, "cold", 128), done)

    def test_survives_a_truncated_line(self):
        """A row half-written when the host died must not poison the index."""
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory)
            with open(Path(directory) / "rows.jsonl", "a", encoding="utf-8") as handle:
                handle.write('{"device": {"mod')
            self.assertEqual(len(run.recorded_cells(Path(directory), "Mac16,7")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
