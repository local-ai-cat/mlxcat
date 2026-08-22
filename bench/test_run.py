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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


class ConcurrencyDecompositionTests(unittest.TestCase):
    """The metric that made an earlier claim on this repo wrong.

    `aggregate_tps` is total completion tokens over wall clock. When prefill is
    serial, request N's first token lands after request 1 has been decoding for
    seconds, so that ratio mostly reports admission latency. Splitting the
    prefill wall from the decode wall is what tells the two engines apart.
    """

    def row(self, start, first, last, prompt=1000, completion=101):
        return {"t_start": start, "t_first": first, "t_last": last,
                "prompt_tokens": prompt, "completion_tokens": completion}

    def test_serial_and_batched_admission_look_identical_on_wall_clock(self):
        # Both finish at t=5 with the same tokens, so tokens/wall is equal...
        serial = [self.row(0, 1 + i, 5) for i in range(4)]        # prefills queue up
        batched = [self.row(0, 1, 5) for _ in range(4)]           # prefills overlap
        wall = lambda rows: sum(r["completion_tokens"] for r in rows) / (
            max(r["t_last"] for r in rows) - min(r["t_start"] for r in rows))
        self.assertAlmostEqual(wall(serial), wall(batched))

        # ...and the decomposition separates them cleanly.
        a = run.concurrency_metrics(serial, 1e9, 1e9)
        b = run.concurrency_metrics(batched, 1e9, 1e9)
        self.assertAlmostEqual(a["prefill_wall_ms"], 4000)   # last first-token at t=4
        self.assertAlmostEqual(b["prefill_wall_ms"], 1000)
        self.assertGreater(b["pp_tps"], a["pp_tps"] * 3)

    def test_goodput_excludes_requests_that_missed_the_sla(self):
        rows = [self.row(0, 0.5, 5), self.row(0, 0.5, 5), self.row(0, 9.0, 12), self.row(0, 9.0, 12)]
        metrics = run.concurrency_metrics(rows, sla_ttft_ms=2000, sla_tpot_ms=1e9)
        self.assertAlmostEqual(metrics["goodput_frac"], 0.5)

    def test_tail_is_reported_not_just_the_middle(self):
        rows = [self.row(0, 0.1, 5) for _ in range(9)] + [self.row(0, 4.0, 5)]
        metrics = run.concurrency_metrics(rows, 1e9, 1e9)
        # 9 fast + 1 slow: the mean barely moves, the tail does. p95 interpolates
        # between the 9th and 10th samples, so it lands at 2245 ms, not 4000.
        self.assertAlmostEqual(metrics["ttft_mean_ms"], 490)
        self.assertGreater(metrics["ttft_p95_ms"], 2000)

    def test_single_completion_token_rows_do_not_divide_by_zero(self):
        rows = [self.row(0, 1, 1, completion=1) for _ in range(2)]
        metrics = run.concurrency_metrics(rows, 1e9, 1e9)
        self.assertIsNone(metrics["tpot_mean_ms"])
        self.assertEqual(metrics["goodput_frac"], 1.0)

    def test_empty_input_is_not_a_crash(self):
        self.assertEqual(run.concurrency_metrics([], 1, 1), {})


class DebugBuildGuardTests(unittest.TestCase):
    """A debug binary of our own engine is a 2-4x handicap that reads as losing."""

    def test_rejects_a_debug_path(self):
        with self.assertRaises(run.EngineUnavailable):
            run.reject_debug_build("/repo/.build/debug/mlxcat-http", "mlxcat")

    def test_accepts_a_release_path(self):
        run.reject_debug_build("/repo/.build/release/mlxcat-http", "mlxcat")

    def test_does_not_trip_on_a_directory_merely_containing_the_word(self):
        run.reject_debug_build("/repo/.build/release/debugger-tools-mlxcat-http", "mlxcat")


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


class FakeEngineHandler(BaseHTTPRequestHandler):
    """Just enough OpenAI-compatible surface to drive a real cell."""

    protocol_version = "HTTP/1.1"
    COMPLETION_TOKENS = 6

    def log_message(self, *args):  # silence the test output
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"data": [{"id": "fake-model"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body["messages"][0]["content"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for index in range(self.COMPLETION_TOKENS):
            time.sleep(0.01)  # a measurable decode window, one chunk per token
            self._sse({"choices": [{"delta": {"content": f"t{index} "}}]})
        self._sse({
            "choices": [],
            "usage": {"prompt_tokens": max(len(prompt) // 4, 1),
                      "completion_tokens": self.COMPLETION_TOKENS},
        })
        self._raw(b"data: [DONE]\n\n")
        self._raw(b"")

    def _json(self, payload):
        blob = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _sse(self, payload):
        self._raw(f"data: {json.dumps(payload)}\n\n".encode())

    def _raw(self, chunk):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))


@unittest.skipUnless(MACOS, "run.main() builds a libproc Footprint")
class EndToEndTests(unittest.TestCase):
    """Drive the whole harness against a fake engine.

    This exists because of a specific failure: `--cache-modes` was parsed,
    printed in the plan, documented in bench/README.md — and `cache_mode` was
    read four times inside the cell loop and assigned nowhere. Every gate we had
    passed. `py_compile` passed, `--dry-run` passed (it returns before the loop),
    CI was green, and the first machine to actually reach that line died with a
    NameError after waiting sixteen minutes for the host to go quiet.

    A harness that is only ever tested by dry-run is tested nowhere near the code
    that produces rows.
    """

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEngineHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.results = tempfile.mkdtemp()

    def rows(self):
        out = []
        for path in Path(self.results).glob("*.jsonl"):
            out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return out

    def run_harness(self, *extra):
        argv = [
            "--engine-url", f"fake={self.url}",
            "--engines", "fake",
            "--models", "fake-model",
            "--contexts", "short",
            "--results-dir", self.results,
            "--runs", "1", "--warmup", "0",
            "--allow-loaded", "--max-load", "1e9", "--min-free-pct", "0",
            *extra,
        ]
        return run.main(argv)

    def test_writes_a_row_end_to_end(self):
        self.assertEqual(self.run_harness("--concurrency", "1"), 0)
        rows = self.rows()
        self.assertTrue(rows)
        metrics = rows[0]["metrics"]
        self.assertGreater(metrics["ttft_ms"]["median"], 0)
        self.assertEqual(metrics["completion_tokens"], FakeEngineHandler.COMPLETION_TOKENS)

    def test_cold_and_warm_are_both_produced(self):
        """The regression this class was written for."""
        self.assertEqual(self.run_harness("--cache-modes", "cold,warm", "--concurrency", "1"), 0)
        modes = {r["workload"]["cache_mode"] for r in self.rows()}
        self.assertEqual(modes, {"cold", "warm"})

    def test_an_unknown_cache_mode_is_rejected_not_ignored(self):
        self.assertEqual(self.run_harness("--cache-modes", "lukewarm"), 64)

    def test_concurrency_rows_carry_the_decomposed_metrics(self):
        self.assertEqual(
            self.run_harness("--concurrency", "2", "--concurrency-tier", "short"), 0
        )
        wide = [r for r in self.rows() if r["workload"]["concurrency"] > 1]
        self.assertTrue(wide, "no concurrency row was produced")
        metrics = wide[0]["metrics"]
        for key in ("pp_tps", "decode_agg_tps", "ttft_p95_ms", "goodput_frac"):
            self.assertIsNotNone(metrics.get(key), f"{key} missing from a concurrency row")

    def test_resume_skips_a_cell_it_already_recorded(self):
        self.assertEqual(self.run_harness("--concurrency", "1"), 0)
        first = len(self.rows())
        self.assertEqual(self.run_harness("--concurrency", "1", "--resume"), 1)
        self.assertEqual(len(self.rows()), first, "--resume re-recorded a finished cell")

    def test_open_loop_arrivals_are_recorded_on_the_row(self):
        self.assertEqual(
            self.run_harness("--concurrency", "2", "--concurrency-tier", "short",
                             "--request-rate", "50"), 0
        )
        wide = [r for r in self.rows() if r["workload"]["concurrency"] > 1]
        self.assertTrue(wide[0]["workload"]["arrival"].startswith("poisson:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
