#!/usr/bin/env python3
"""No-model corrective tests for the task-owned benchmark harness."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import benchmark
import coding_evaluator
import coding_tasks


REFERENCE_SOLUTIONS = {
    "function_completion": "def clamp(value, lower, upper):\n    if lower > upper:\n        raise ValueError('bounds')\n    return min(max(value, lower), upper)",
    "bug_fix": "Here you go:\n```python\ndef unique_counts(items):\n    counts = {}\n    for item in items:\n        counts[item] = counts.get(item, 0) + 1\n    return counts\n```",
    "simple_refactor": "def normalize_names(items):\n    return [name.strip().lower() for name in items if name is not None and name.strip()]",
    "json_tool_call": '{"tool":"lookup_weather","arguments":{"city":"London","units":"celsius"}}',
    "parse_duration": "import re\n\ndef parse_duration(text):\n    match = re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', text.strip())\n    if not text.strip() or not match:\n        raise ValueError(text)\n    h, m, s = (int(g) if g else 0 for g in match.groups())\n    return h * 3600 + m * 60 + s",
    "merge_intervals": "def merge_intervals(intervals):\n    out = []\n    for start, end in sorted(intervals):\n        if out and start <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], end)\n        else:\n            out.append([start, end])\n    return out",
    "slugify": "import re\ndef slugify(title):\n    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')",
    "roman_to_int": "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    for i, c in enumerate(s):\n        if i + 1 < len(s) and v[c] < v[s[i + 1]]:\n            total -= v[c]\n        else:\n            total += v[c]\n    return total",
    "top_words": "import re\nfrom collections import Counter\ndef top_words(text, k):\n    counts = Counter(re.findall('[a-z]+', text.lower()))\n    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]",
    "csv_line": "def parse_csv_line(line):\n    fields, cur, quoted, i = [], '', False, 0\n    while i < len(line):\n        c = line[i]\n        if quoted:\n            if c == '\"' and i + 1 < len(line) and line[i + 1] == '\"':\n                cur += '\"'\n                i += 1\n            elif c == '\"':\n                quoted = False\n            else:\n                cur += c\n        elif c == '\"':\n            quoted = True\n        elif c == ',':\n            fields.append(cur)\n            cur = ''\n        else:\n            cur += c\n        i += 1\n    fields.append(cur)\n    return fields",
    "bug_fix_binary_search": "def find_index(items, target):\n    lo, hi = 0, len(items)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return -1",
    "bug_fix_chunks": "def chunks(items, size):\n    if size < 1:\n        raise ValueError(size)\n    return [items[i:i + size] for i in range(0, len(items), size)]",
    "tool_call_reminder": '{"tool": "create_reminder", "arguments": {"title": "Call Mom", "due": {"date": "2026-10-08", "time": "09:00"}}}',
    "tool_call_typed_args": '{"tool":"search_files","arguments":{"query":"invoice","limit":5,"include_archived":false,"extensions":["csv","pdf"]}}',
    "swift_format_duration": "```swift\nimport Foundation\n\nfunc formatDuration(_ seconds: Int) -> String {\n    let h = seconds / 3600, m = (seconds % 3600) / 60, s = seconds % 60\n    if h > 0 { return String(format: \"%d:%02d:%02d\", h, m, s) }\n    return String(format: \"%d:%02d\", m, s)\n}\n```",
    "swift_median": "func median(_ values: [Double]) -> Double? {\n    guard !values.isEmpty else { return nil }\n    let s = values.sorted(); let n = s.count\n    return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2\n}",
}


class FakeSafety:
    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def snapshot(self) -> dict[str, object]:
        return {"at": benchmark.utc_now(), "swap_delta_bytes": 0}

    def require_safe(self) -> dict[str, object]:
        self.calls += 1
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise benchmark.SafetyStop("synthetic safety threshold")
        return self.snapshot()


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def score(self, task: str, source: str) -> tuple[bool, str]:
        return benchmark.score_coding_task(task, source, self.root / task)

    def test_reference_solutions_pass_every_task(self) -> None:
        for task, source in REFERENCE_SOLUTIONS.items():
            with self.subTest(task=task):
                passed, detail = self.score(task, source)
                self.assertTrue(passed, detail)
        self.assertEqual(set(REFERENCE_SOLUTIONS), set(coding_tasks.CODING_TASKS))

    def test_wrong_solutions_fail(self) -> None:
        self.assertFalse(self.score("function_completion", "def clamp(value, lower, upper):\n    return value")[0])
        self.assertFalse(self.score("bug_fix", "def unique_counts(items):\n    return {item: 1 for item in items}")[0])
        self.assertFalse(self.score("swift_median", "```swift\nfunc median(_ values: [Double]) -> Double? { values.first }\n```")[0])
        passed, detail = self.score("swift_format_duration", "func formatDuration(_ seconds: Int) -> String { seconds }")
        self.assertFalse(passed)
        self.assertIn("compile failed", detail)
        self.assertFalse(self.score("tool_call_typed_args", '{"tool":"search_files","arguments":{"query":"invoice","limit":"5","include_archived":false,"extensions":["pdf","csv"]}}')[0])

    def test_fenced_json_is_accepted_and_labelled(self) -> None:
        passed, detail = self.score("json_tool_call", '```json\n{"tool": "lookup_weather", "arguments": {"city": "London", "units": "celsius"}}\n```')
        self.assertTrue(passed, detail)
        self.assertTrue(detail.startswith("fenced JSON"))

    def test_introspection_and_unlisted_imports_are_rejected(self) -> None:
        self.assertIn("dunder", coding_evaluator.validate("def clamp(value, lower, upper):\n    return value.__class__", "clamp")[1])
        self.assertIn("import not allowed", coding_evaluator.validate("import os\ndef clamp(value, lower, upper):\n    return value", "clamp")[1])
        self.assertTrue(coding_evaluator.validate("import re\ndef clamp(value: int, lower: int = 0, upper: int = 1) -> int:\n    return value", "clamp")[0])

    def test_infinite_loop_is_killed(self) -> None:
        started = time.monotonic()
        passed, detail = self.score("function_completion", "def clamp(value, lower, upper):\n    while True:\n        pass\nclamp(1, 0, 2)")
        self.assertFalse(passed)
        self.assertLess(time.monotonic() - started, 6)
        self.assertNotIn("BLOCKED", detail)

    def test_strict_sandbox_allows_owned_files_and_blocks_the_rest(self) -> None:
        inside = self.root / "inside.txt"
        result = self.root / "probe-result.txt"
        script = self.root / "probe.zsh"
        outside = Path(f"/private/tmp/cat-m4-sandbox-sentinel-{os.getpid()}")
        inside.write_text("owned\n")
        outside.write_text("outside\n")
        script.write_text(
            "inside=$(<\"$1\") || exit 10\n"
            "[[ $inside == owned ]] || exit 11\n"
            "if outside=$(<\"$2\") 2>/dev/null; then exit 12; fi\n"
            "if print -r -- changed >\"$2\" 2>/dev/null; then exit 13; fi\n"
            "if home=$(<\"$5\") 2>/dev/null; then exit 16; fi\n"
            "zmodload zsh/net/tcp || exit 14\n"
            "if ztcp 127.0.0.1 \"$3\" 2>/dev/null; then exit 15; fi\n"
            "print -r -- boundary-enforced >\"$4\"\n"
        )
        profile = self.root / "strict.sb"
        benchmark.write_evaluator_profile(profile, self.root, [])
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        try:
            completed = subprocess.run(
                ["/usr/bin/sandbox-exec", "-f", str(profile), "/bin/zsh", "-f", str(script), str(inside), str(outside),
                 str(server.getsockname()[1]), str(result), str(Path.home() / ".zshenv")],
                env=benchmark.minimal_environment(self.root), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=3,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(result.read_text(), "boundary-enforced\n")
            self.assertEqual(outside.read_text(), "outside\n")
        finally:
            server.close()
            outside.unlink(missing_ok=True)

    def test_failed_request_has_exact_durable_receipt(self) -> None:
        receipts = self.root / "receipts"
        with mock.patch.object(benchmark, "post_json", side_effect=TimeoutError("synthetic timeout")):
            row = benchmark.run_request(
                "http://127.0.0.1:9", os.getpid(), "model", "short_chat", "prompt",
                2, 1, 1, FakeSafety(), receipts, 7,
            )
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["request_role"], "exact_replay")
        receipt = Path(row["receipt_path"])
        self.assertTrue(receipt.exists())
        self.assertIn("synthetic timeout", json.loads(receipt.read_text())["error"])

    def test_watchdog_terminates_only_its_owned_pid(self) -> None:
        process = subprocess.Popen(["/bin/sleep", "30"], text=True)
        watchdog = benchmark.SafetyWatchdog(process, FakeSafety(fail_after=1), interval=0.01)
        watchdog.start()
        try:
            process.wait(timeout=2)
        finally:
            watchdog.stop()
            if process.poll() is None:
                process.kill()
        self.assertEqual(watchdog.violation, "synthetic safety threshold")

    def test_server_policy_allows_loopback_and_environment_is_allowlisted(self) -> None:
        profile = self.root / "loopback.sb"
        benchmark.write_loopback_profile(profile)
        script = self.root / "loopback.py"
        script.write_text(
            "import socket\n"
            "server=socket.socket(); server.bind(('127.0.0.1',0)); server.listen(1)\n"
            "client=socket.socket(); client.connect(server.getsockname())\n"
            "peer,_=server.accept(); peer.close(); client.close(); server.close()\n"
        )
        completed = subprocess.run(
            ["/usr/bin/sandbox-exec", "-f", str(profile), "/usr/bin/python3", "-B", str(script)],
            env=benchmark.minimal_environment(self.root), capture_output=True, text=True, timeout=3,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            set(benchmark.minimal_environment(self.root)),
            {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "MLXCAT_TOOL_GRAMMAR"},
        )

    def test_grounded_extract_accepts_either_project_name_from_the_source(self) -> None:
        for project in ("Orion", "Project Orion"):
            content = json.dumps({"project": project, "owner": "Mira", "budget": 4200, "deadline": "Friday"})
            self.assertTrue(benchmark.check_case("grounded_extract", content)[0], project)
        for wrong in (
            {"project": "Apollo", "owner": "Mira", "budget": 4200, "deadline": "Friday"},
            {"project": "Orion", "owner": "Mira", "budget": 4300, "deadline": "Friday"},
            {"project": "Orion", "owner": "Mira", "budget": 4200, "deadline": "Friday", "extra": 1},
            {"owner": "Mira", "budget": 4200, "deadline": "Friday"},
        ):
            self.assertFalse(benchmark.check_case("grounded_extract", json.dumps(wrong))[0], wrong)
        self.assertFalse(benchmark.check_case("grounded_extract", "[1, 2]")[0])


if __name__ == "__main__":
    unittest.main()
