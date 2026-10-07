#!/usr/bin/env python3
"""Bounded, offline M4 benchmark for the existing mlxcat HTTP binary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from coding_tasks import CODING_TASKS, JSON, SWIFT, matches_expected

GIB = 1024**3
SIGXCPU = 24
DEFAULT_MODELS = [
    "Qwen3-0.6B-4bit",
    "Llama-3.2-3B-Instruct-4bit",
    "Qwen3.5-4B-MLX-4bit",
    "Qwen2.5-Coder-7B-Instruct-4bit",
]
CODING_MODEL = "Qwen2.5-Coder-7B-Instruct-4bit"
REGULAR_MAX_TOKENS = 128
CODING_MAX_TOKENS = 1024
SEED = 42


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=False, capture_output=True, text=True).stdout.strip()


def swap_used_bytes() -> int:
    output = command_output(["sysctl", "-n", "vm.swapusage"])
    match = re.search(r"used = ([0-9.]+)([MGT])", output)
    if not match:
        return 0
    scales = {"M": 1024**2, "G": GIB, "T": 1024**4}
    return int(float(match.group(1)) * scales[match.group(2)])


def battery_percent() -> int | None:
    output = command_output(["pmset", "-g", "batt"])
    match = re.search(r"(\d+)%", output)
    return int(match.group(1)) if match else None


def memory_free_percent() -> int | None:
    output = command_output(["memory_pressure"])
    match = re.search(r"System-wide memory free percentage: (\d+)%", output)
    return int(match.group(1)) if match else None


class SafetyStop(RuntimeError):
    pass


class SafetyMonitor:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.baseline_swap = swap_used_bytes()

    def snapshot(self) -> dict[str, Any]:
        free_disk = shutil.disk_usage(self.root).free
        battery = battery_percent()
        free_memory = memory_free_percent()
        pressure_raw = command_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
        pressure = int(pressure_raw) if pressure_raw.isdigit() else None
        swap = swap_used_bytes()
        return {
            "at": utc_now(),
            "free_disk_bytes": free_disk,
            "battery_percent": battery,
            "memory_free_percent": free_memory,
            "memory_pressure_level": pressure,
            "swap_used_bytes": swap,
            "swap_delta_bytes": swap - self.baseline_swap,
        }

    def require_safe(self) -> dict[str, Any]:
        state = self.snapshot()
        if state["free_disk_bytes"] < 40 * GIB:
            raise SafetyStop("free disk fell below 40 GiB")
        if state["battery_percent"] is not None and state["battery_percent"] < 30:
            raise SafetyStop("battery fell below 30 percent")
        if state["swap_delta_bytes"] > GIB:
            raise SafetyStop("swap grew by more than 1 GiB")
        if state["memory_pressure_level"] is not None and state["memory_pressure_level"] > 1:
            raise SafetyStop("memory pressure entered warning or critical state")
        if state["memory_free_percent"] is not None and state["memory_free_percent"] < 25:
            raise SafetyStop("less than 12 GiB estimated host headroom remains")
        return state


class RSSSampler:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            value = command_output(["ps", "-o", "rss=", "-p", str(self.pid)])
            if value.strip().isdigit():
                self.samples.append(int(value.strip()) * 1024)
            self.stop_event.wait(0.1)

    def __enter__(self) -> "RSSSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    @property
    def peak(self) -> int | None:
        return max(self.samples) if self.samples else None


class SafetyWatchdog:
    """Continuously stop only the server process owned by this harness."""

    def __init__(self, process: subprocess.Popen[str], safety: SafetyMonitor, interval: float = 0.5) -> None:
        self.process = process
        self.safety = safety
        self.interval = interval
        self.stop_event = threading.Event()
        self.violation: str | None = None
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.safety.require_safe()
            except SafetyStop as error:
                self.violation = str(error)
                if self.process.poll() is None:
                    self.process.terminate()
                return

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


def post_json(url: str, payload: dict[str, Any], timeout: int = 240) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get_json(url: str, timeout: int = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(base_url: str, process: subprocess.Popen[str], watchdog: SafetyWatchdog) -> float:
    started = time.monotonic()
    while time.monotonic() - started < 30:
        if process.poll() is not None:
            if watchdog.violation:
                raise SafetyStop(watchdog.violation)
            raise RuntimeError(f"mlxcat server exited with {process.returncode}")
        try:
            get_json(f"{base_url}/health")
            return time.monotonic() - started
        except (OSError, ValueError, urllib.error.URLError):
            time.sleep(0.1)
    raise TimeoutError("mlxcat server did not become healthy in 30 seconds")


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def check_case(case: str, content: str) -> tuple[bool, str]:
    stripped = content.strip()
    if case == "short_chat":
        return "CAT_OK" in stripped, "expected CAT_OK"
    if case == "grounded_extract":
        try:
            candidate = re.search(r"\{.*\}", stripped, re.DOTALL)
            parsed = json.loads(candidate.group(0) if candidate else stripped)
        except (json.JSONDecodeError, AttributeError) as error:
            return False, f"invalid JSON: {error}"
        # The source names the project "Project Orion", so either form is faithful to it.
        expected = {"owner": "Mira", "budget": 4200, "deadline": "Friday"}
        accepted_projects = ("Orion", "Project Orion")
        ok = (
            isinstance(parsed, dict)
            and parsed.get("project") in accepted_projects
            and {key: value for key, value in parsed.items() if key != "project"} == expected
        )
        return ok, f"expected project in {accepted_projects!r} and {expected!r}, got {parsed!r}"
    if case == "context_2k":
        return "NEEDLE_2K_7319" in stripped, "expected NEEDLE_2K_7319"
    if case == "context_8k":
        return "NEEDLE_8K_2846" in stripped, "expected NEEDLE_8K_2846"
    return True, "unchecked"


def context_prompt(records: int, marker: str) -> str:
    target = records * 3 // 4
    lines = []
    for index in range(records):
        value = marker if index == target else f"ordinary-{index:04d}"
        lines.append(f"Record {index:04d} has retrieval value {value} and category violet.")
    return (
        "Read the records below. Return only the retrieval value in record "
        f"{target:04d}; do not explain.\n" + "\n".join(lines)
    )


def regular_cases() -> list[tuple[str, str]]:
    return [
        ("short_chat", "Reply with exactly CAT_OK and nothing else."),
        (
            "grounded_extract",
            "Source: Project Orion is owned by Mira. Its approved budget is 4200 credits and its deadline is Friday. "
            "Return only compact JSON with keys project, owner, budget, deadline. Use the source exactly.",
        ),
        ("context_2k", context_prompt(125, "NEEDLE_2K_7319")),
        ("context_8k", context_prompt(500, "NEEDLE_8K_2846")),
    ]


FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
UNCLOSED_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*)\Z", re.DOTALL)
# Swift answers are graded as a file in an app target, where Foundation is a given.
SWIFT_PRELUDE = "import Foundation\n\n"


def code_from_output(content: str, language: str) -> str:
    """The first fenced block tagged with the language, else the first fence, else the whole reply."""
    blocks = FENCE.findall(content)
    for tag, body in blocks:
        if tag.lower() == language:
            return body.strip()
    if blocks:
        return blocks[0][1].strip()
    unclosed = UNCLOSED_FENCE.search(content)
    if unclosed:
        # A reply cut off by max_tokens leaves its fence open; grade what was written.
        return unclosed.group(1).strip()
    return content.strip()


def json_from_output(content: str) -> tuple[Any, str]:
    """Strict parse first; then the body of a single ```json fence. Returns (value, how)."""
    try:
        return json.loads(content.strip()), "strict"
    except json.JSONDecodeError:
        pass
    blocks = FENCE.findall(content)
    if len(blocks) == 1:
        return json.loads(blocks[0][1].strip()), "fenced"
    raise json.JSONDecodeError("no strict JSON and not exactly one fenced block", content, 0)


def python_runtime() -> tuple[Path, Path]:
    """The real interpreter behind /usr/bin/python3 and its base prefix.

    /usr/bin/python3 is an xcrun shim that dlopens libxcrun from the active
    developer dir; launching the resolved interpreter keeps that outside the
    sandbox's read set.
    """
    query = subprocess.run(
        ["/usr/bin/python3", "-I", "-c", "import sys; print(sys.base_prefix); print('%d.%d' % sys.version_info[:2])"],
        capture_output=True, text=True, check=True, timeout=5,
    ).stdout.splitlines()
    prefix = Path(query[0]).resolve()
    return prefix / "bin" / f"python{query[1]}", prefix


def run_sandboxed(command: list[str], profile_path: Path, environment: dict[str, str], wall_seconds: float, rss_limit_bytes: int) -> tuple[int | None, str | None, str]:
    process = subprocess.Popen(
        ["/usr/bin/sandbox-exec", "-f", str(profile_path), *command],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment,
    )
    deadline = time.monotonic() + wall_seconds
    stop_reason: str | None = None
    while process.poll() is None:
        if time.monotonic() >= deadline:
            stop_reason = f"exceeded {wall_seconds:g} second wall-clock limit"
            process.kill()
            break
        rss = command_output(["ps", "-o", "rss=", "-p", str(process.pid)])
        if rss.isdigit() and int(rss) * 1024 > rss_limit_bytes:
            stop_reason = f"exceeded {rss_limit_bytes // 1024**2} MiB resident-memory limit"
            process.kill()
            break
        time.sleep(0.02)
    output, _ = process.communicate(timeout=5)
    return process.returncode, stop_reason, output or ""


def score_coding_task(task: str, content: str, workspace: Path) -> tuple[bool, str]:
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    (workspace / "raw-output.txt").write_text(content)
    specification = CODING_TASKS[task]
    if specification["kind"] == JSON:
        try:
            parsed, how = json_from_output(content)
        except json.JSONDecodeError as error:
            return False, f"no parseable JSON: {error.msg}"
        passed = matches_expected(parsed, specification["expected"])
        return passed, f"{how} JSON; expected {specification['expected']!r}, got {parsed!r}"
    if specification["kind"] == SWIFT:
        return score_swift_task(task, content, workspace)
    code = code_from_output(content, "python")
    candidate_path = workspace / "candidate.py"
    tests_path = workspace / "tests.py"
    evaluator_path = workspace / "coding_evaluator.py"
    profile_path = workspace / "sandbox.sb"
    result_path = workspace / "evaluation-result.json"
    candidate_path.write_text(code + "\n")
    tests_path.write_text(specification["tests"] + "\n")
    shutil.copyfile(Path(__file__).with_name("coding_evaluator.py"), evaluator_path)
    result_path.write_text("")
    interpreter, prefix = python_runtime()
    write_evaluator_profile(profile_path, workspace, [prefix])
    returncode, stop_reason, output = run_sandboxed(
        [str(interpreter), "-I", "-B", str(evaluator_path), str(candidate_path), str(tests_path), specification["entry"], str(result_path)],
        profile_path, minimal_environment(workspace), 5, 256 * 1024**2,
    )
    (workspace / "evaluator-output.txt").write_text(output)
    if stop_reason:
        return False, f"sandboxed evaluator {stop_reason}"
    try:
        result = json.loads(result_path.read_text())
    except json.JSONDecodeError:
        if returncode == -SIGXCPU:
            return False, "evaluator hit its 2 s CPU limit (SIGXCPU)"
        return False, f"BLOCKED: strict evaluator sandbox unavailable (exit {returncode}); failed closed"
    return bool(result["passed"]), str(result["detail"])


def score_swift_task(task: str, content: str, workspace: Path) -> tuple[bool, str]:
    """Compile candidate + fixed preconditions with swiftc, then run the binary; both sandboxed."""
    code = code_from_output(content, "swift")
    source_path = workspace / "main.swift"
    source_path.write_text(SWIFT_PRELUDE + code + "\n\n// fixed tests\n" + CODING_TASKS[task]["tests"] + "\n")
    developer = Path(command_output(["xcode-select", "-p"]).strip()).resolve()
    swiftc = command_output(["xcrun", "-f", "swiftc"]).strip()
    sdk = command_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"]).strip()
    app_root = next((parent for parent in developer.parents if parent.suffix == ".app"), developer)
    compile_profile = workspace / "compile.sb"
    run_profile = workspace / "sandbox.sb"
    write_evaluator_profile(compile_profile, workspace, [app_root, Path("/Library/Developer/CommandLineTools")])
    write_evaluator_profile(run_profile, workspace, [])
    binary = workspace / "candidate-bin"
    environment = minimal_environment(workspace)
    environment["TMPDIR"] = str(workspace) + "/"
    returncode, stop_reason, output = run_sandboxed(
        [swiftc, "-Onone", "-sdk", sdk, "-module-cache-path", str(workspace / "module-cache"), str(source_path), "-o", str(binary)],
        compile_profile, environment, 90, 2 * 1024**3,
    )
    (workspace / "compile-output.txt").write_text(output)
    if stop_reason:
        return False, f"swiftc {stop_reason}"
    if returncode != 0:
        first_error = next((line for line in output.splitlines() if "error:" in line), output.strip()[:200])
        return False, f"compile failed: {first_error}"
    returncode, stop_reason, output = run_sandboxed([str(binary)], run_profile, environment, 5, 256 * 1024**2)
    (workspace / "run-output.txt").write_text(output)
    if stop_reason:
        return False, f"swift tests {stop_reason}"
    if returncode == 0 and "SWIFT_TESTS_PASSED" in output:
        return True, "fixed Swift preconditions passed in sandbox"
    return False, f"fixed Swift preconditions failed (exit {returncode})"


def minimal_environment(runtime_dir: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(runtime_dir),
        "TMPDIR": str(runtime_dir),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MLXCAT_TOOL_GRAMMAR": "0",
    }


def write_evaluator_profile(path: Path, workspace: Path, extra_read_roots: list[Path]) -> None:
    """Deny file data by default; allow system runtime, the given roots and owned scratch.

    `(literal "/")` is required: dyld reads the root directory while starting any
    process, and without it even /usr/bin/true aborts with SIGABRT (exit 134) —
    the 2026-10-07 "sandbox unavailable" result.
    """
    read_rules = [
        '(literal "/")',
        '(subpath "/usr/lib")',
        '(subpath "/usr/bin")',
        '(subpath "/bin")',
        '(subpath "/System")',
        '(subpath "/private/etc")',
        '(subpath "/private/var/db/timezone")',
        '(subpath "/private/var/select")',
        '(literal "/dev/null")',
        '(literal "/dev/urandom")',
        '(literal "/dev/random")',
    ]
    read_rules.extend(f'(subpath "{root}")' for root in sorted({Path(root).resolve() for root in extra_read_roots}, key=str))
    owned = str(workspace.resolve()).replace('"', '\\"')
    path.write_text("\n".join([
        '(version 1)',
        '(allow default)',
        '(deny network*)',
        '(deny file-read-data)',
        '(deny file-write*)',
        '(allow file-read-metadata)',
        '(allow file-read-data ' + " ".join(read_rules) + f' (subpath "{owned}"))',
        f'(allow file-write* (subpath "{owned}") (literal "/dev/null"))',
    ]) + "\n")


def write_loopback_profile(path: Path) -> None:
    path.write_text(
        '(version 1)\n(allow default)\n(deny network*)\n'
        '(allow network-inbound (local ip "localhost:*"))\n'
        '(allow network-outbound (remote ip "localhost:*"))\n'
    )


def model_revision(model_dir: Path) -> str | None:
    resolved = model_dir.resolve()
    if resolved.parent.name == "snapshots":
        return resolved.name  # a shared Hugging Face cache snapshot is named by its revision
    metadata = model_dir / ".cache/huggingface/download/config.json.metadata"
    if not metadata.exists():
        return None
    lines = metadata.read_text().splitlines()
    return lines[0].strip() if lines else None


def model_manifest(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text())
    weights = sorted(model_dir.glob("*.safetensors"))
    return {
        "id": model_dir.name,
        "path": str(model_dir),
        "revision": model_revision(model_dir),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "quantization": config.get("quantization") or config.get("quantization_config"),
        "max_context": config.get("max_position_embeddings") or (config.get("text_config") or {}).get("max_position_embeddings"),
        "weights_bytes": sum(path.stat().st_size for path in weights),
        "config_sha256": sha256(model_dir / "config.json"),
        "tokenizer_config_sha256": sha256(model_dir / "tokenizer_config.json"),
    }


def run_request(
    base_url: str,
    server_pid: int,
    model: str,
    case: str,
    prompt: str,
    repetition: int,
    max_tokens: int,
    timeout_seconds: int,
    safety: SafetyMonitor,
    receipts: Path,
    request_ordinal: int,
) -> dict[str, Any]:
    receipts.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts / f"{model}__{case}__r{repetition}.json"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": SEED,
        "enable_thinking": False,
        "max_tokens": max_tokens,
        "stream": False,
    }
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    response: dict[str, Any] = {}
    content = ""
    error_text: str | None = None
    peak: int | None = None
    client_total: float | None = None
    started = time.monotonic()
    try:
        before = safety.require_safe()
        with RSSSampler(server_pid) as sampler:
            response = post_json(f"{base_url}/v1/chat/completions", payload, timeout=timeout_seconds)
        peak = sampler.peak
        client_total = time.monotonic() - started
        after = safety.require_safe()
        content = extract_content(response)
        passed, detail = check_case(case, content) if not case.startswith("coding_") else (True, "scored separately")
        status = "completed"
    except Exception as error:
        client_total = time.monotonic() - started
        error_text = f"{type(error).__name__}: {error}"
        passed, detail, status = False, error_text, "failed"
        try:
            after = safety.snapshot()
        except Exception:
            after = None
    usage = response.get("usage") or {}
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    row = {
        "at": utc_now(),
        "model": model,
        "case": case,
        "repetition": repetition,
        "status": status,
        "request_role": "first_process_request" if request_ordinal == 0 else ("first_case_request" if repetition == 0 else "exact_replay"),
        "cache_state": "server_reported_cached" if isinstance(cached_tokens, int) and cached_tokens > 0 else "unverified_no_reported_cache",
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_prompt_tokens": cached_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "server_ttft_seconds": usage.get("time_to_first_token"),
        "server_total_seconds": usage.get("total_time"),
        "prompt_tokens_per_second": usage.get("prompt_tokens_per_second"),
        "decode_tokens_per_second": usage.get("generation_tokens_per_second"),
        "client_total_seconds": client_total,
        "server_peak_rss_bytes": peak,
        "swap_delta_bytes": after.get("swap_delta_bytes") if after else None,
        "passed": passed,
        "detail": detail,
        "finish_reason": ((response.get("choices") or [{}])[0]).get("finish_reason"),
        "output": content,
        "resource_before": before,
        "resource_after": after,
        "receipt_path": str(receipt_path),
    }
    receipt_path.write_text(json.dumps({"request": payload, "result": row, "raw_response": response, "error": error_text}, indent=2) + "\n")
    return row


def run_model(
    binary: Path,
    model_dir: Path,
    port: int,
    output: Path,
    safety: SafetyMonitor,
    coding_models: set[str],
    coding_tasks: list[str],
    run_regular: bool = True,
    server_environment: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = model_dir.name
    log_path = output / "logs" / f"{model}.server.log"
    log_handle = log_path.open("w")
    runtime_dir = output / "runtime" / model
    runtime_dir.mkdir(parents=True, exist_ok=True)
    profile_path = runtime_dir / "loopback-only.sb"
    write_loopback_profile(profile_path)
    environment = minimal_environment(runtime_dir)
    environment.update(server_environment or {})
    server_command = [
        str(binary),
        "--host", "127.0.0.1",
        "--port", str(port),
        # Resolved: the server treats a symlinked --model-dir as a root to search, not a model.
        "--model-dir", str(model_dir.resolve()),
        "--model-id", model,
        "--max-concurrent-requests", "1",
        "--memory-ceiling-bytes", str(24 * GIB),
        "--pin", model,
    ]
    command = ["/usr/bin/sandbox-exec", "-f", str(profile_path), *server_command]
    started = utc_now()
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=environment)
    watchdog = SafetyWatchdog(process, safety)
    watchdog.start()
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "model": model,
        "command": command,
        "environment_keys": sorted(environment),
        "network_policy": str(profile_path),
        "started_at": started,
    }
    request_ordinal = 0
    try:
        metadata["server_ready_seconds"] = wait_for_server(f"http://127.0.0.1:{port}", process, watchdog)
        receipts = output / "receipts"
        for case, prompt in (regular_cases() if run_regular else []):
            for repetition in range(4):
                row = run_request(
                    f"http://127.0.0.1:{port}", process.pid, model, case, prompt,
                    repetition, REGULAR_MAX_TOKENS, 240, safety, receipts, request_ordinal,
                )
                rows.append(row)
                request_ordinal += 1
                if row["status"] != "completed":
                    raise RuntimeError(f"request failed; receipt: {row['receipt_path']}")
                if watchdog.violation:
                    raise SafetyStop(watchdog.violation)
        if model in coding_models:
            for task in coding_tasks:
                specification = CODING_TASKS[task]
                case = f"coding_{task}"
                row = run_request(
                    f"http://127.0.0.1:{port}", process.pid, model, case,
                    specification["prompt"], 0, CODING_MAX_TOKENS, 240, safety,
                    output / "receipts", request_ordinal,
                )
                rows.append(row)
                request_ordinal += 1
                if row["status"] != "completed":
                    raise RuntimeError(f"request failed; receipt: {row['receipt_path']}")
                passed, detail = score_coding_task(task, row["output"], output / "coding" / model / task)
                row["passed"] = passed
                row["detail"] = detail
                receipt_path = Path(row["receipt_path"])
                receipt = json.loads(receipt_path.read_text())
                receipt["result"] = row
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
    finally:
        watchdog.stop()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_handle.close()
        metadata["ended_at"] = utc_now()
        metadata["exit_code"] = process.returncode
        metadata["server_log"] = str(log_path)
    return rows, metadata


def write_results(output: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    json_path = output / "results.json"
    json_path.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")
    fields = [
        "model", "case", "repetition", "status", "request_role", "cache_state", "prompt_tokens", "cached_prompt_tokens",
        "completion_tokens", "server_ttft_seconds", "server_total_seconds", "prompt_tokens_per_second",
        "decode_tokens_per_second", "client_total_seconds", "server_peak_rss_bytes", "swap_delta_bytes",
        "passed", "detail", "finish_reason",
    ]
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--port", type=int, default=18192)
    parser.add_argument("--source-revision", required=True, help="mlxcat commit the binary was built from")
    parser.add_argument("--coding-models", nargs="*", default=[CODING_MODEL], help="models that also run the coding suite")
    parser.add_argument("--coding-tasks", nargs="*", default=list(CODING_TASKS), choices=list(CODING_TASKS))
    parser.add_argument("--coding-only", action="store_true", help="skip the regular cases")
    parser.add_argument("--server-env", nargs="*", default=[], metavar="KEY=VALUE", help="extra MLXCAT_* settings for the server")
    arguments = parser.parse_args()
    server_environment = dict(item.split("=", 1) for item in arguments.server_env)
    if any(not key.startswith("MLXCAT_") for key in server_environment):
        parser.error("--server-env only takes MLXCAT_* settings")
    arguments.output.mkdir(parents=True, exist_ok=True)
    for child in ("logs", "receipts", "coding"):
        (arguments.output / child).mkdir(exist_ok=True)
    safety = SafetyMonitor(arguments.output)
    metadata: dict[str, Any] = {
        "schema": "cat-m4-inference-results/v1",
        "started_at": utc_now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "binary": str(arguments.binary),
        "binary_sha256": sha256(arguments.binary),
        "mlxcat_source_revision": arguments.source_revision,
        "sampling": {"temperature": 0, "seed": SEED, "enable_thinking": False},
        "regular_max_tokens": REGULAR_MAX_TOKENS,
        "coding_max_tokens": CODING_MAX_TOKENS,
        "coding_models": arguments.coding_models,
        "coding_tasks": arguments.coding_tasks,
        "coding_only": arguments.coding_only,
        "server_environment": server_environment,
        "models": [],
        "runs": [],
        "baseline_resources": safety.snapshot(),
    }
    rows: list[dict[str, Any]] = []
    status = "completed"
    for offset, model in enumerate(arguments.models):
        try:
            model_dir = arguments.models_root / model
            metadata["models"].append(model_manifest(model_dir))
            model_rows, run = run_model(
                arguments.binary, model_dir, arguments.port + offset, arguments.output, safety,
                set(arguments.coding_models), arguments.coding_tasks, not arguments.coding_only,
                server_environment,
            )
            rows.extend(model_rows)
            metadata["runs"].append(run)
            if run["status"] != "completed":
                status = "partial"
                metadata["error"] = run.get("error")
                write_results(arguments.output, rows, metadata)
                break
            write_results(arguments.output, rows, metadata)
        except Exception as error:
            status = "partial"
            metadata["error"] = f"{type(error).__name__}: {error}"
            print(metadata["error"], flush=True)
            write_results(arguments.output, rows, metadata)
            break
    metadata["status"] = status
    metadata["ended_at"] = utc_now()
    metadata["final_resources"] = safety.snapshot()
    write_results(arguments.output, rows, metadata)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
