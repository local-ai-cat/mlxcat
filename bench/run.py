#!/usr/bin/env python3
"""mlxcat bench — same-transport benchmark harness for local inference engines.

Every engine is measured the same way: an OpenAI-compatible streaming
`/v1/chat/completions` request from the same client, greedy sampling, the same
prompt corpus, the same token budgets. The harness records time-to-first-token,
prefill and decode throughput, end-to-end throughput, inter-token latency,
aggregate throughput under concurrency, and the peak physical footprint of the
engine process (sampled with `proc_pid_rusage`, the same metric the pkg-99 /
pkg-102 matrices used).

Results are appended as newline-delimited JSON (one record per cell) under
`bench/results/` and rendered into `LEADERBOARD.md` by `bench/leaderboard.py`.
See `bench/README.md` for the schema and the rules a row must satisfy before it
is allowed onto the leaderboard.

Python 3.9+, standard library only.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import ctypes
import ctypes.util
import hashlib
import datetime as dt
import json
import os
import platform
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "mlxcat-bench/1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_RESULTS_DIR = HERE / "results"
DEFAULT_ENGINES = HERE / "engines.json"
DEFAULT_MATRIX = HERE / "matrix.json"

# A fixed, public-domain-flavoured filler paragraph. Repeated to reach a target
# prompt length; the trailing question is what the model answers. The content
# is deliberately bland so no engine gets a "clever" shortcut.
FILLER = (
    "The river runs past the old mill and under the stone bridge, where the "
    "light breaks on the water in the early morning. Farmers bring grain in "
    "carts drawn by patient horses, and the miller weighs each sack on a brass "
    "scale that has hung from the same beam for a hundred years. "
)
QUESTION = (
    "Summarize the passage above in three sentences, then explain in plain "
    "prose how a CPU executes an instruction. Keep writing until you have "
    "several complete paragraphs."
)


# --------------------------------------------------------------------------- #
# Host: fingerprint, quiet-machine guard, peak footprint sampling
# --------------------------------------------------------------------------- #


def sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return ""


def device_fingerprint() -> Dict[str, Any]:
    memsize = sysctl("hw.memsize")
    return {
        "model": sysctl("hw.model"),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(memsize) if memsize.isdigit() else None,
        "os": "macOS " + platform.mac_ver()[0],
        "hostname": socket.gethostname(),
        "perf_cores": int(sysctl("hw.perflevel0.physicalcpu") or 0) or None,
        "efficiency_cores": int(sysctl("hw.perflevel1.physicalcpu") or 0) or None,
    }


def memory_free_percent() -> Optional[float]:
    try:
        out = subprocess.check_output(["memory_pressure", "-Q"], text=True)
    except Exception:
        return None
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
    return float(match.group(1)) if match else None


def swap_used_bytes() -> Optional[int]:
    """Swap in use, from `sysctl vm.swapusage`.

    Absolute swap is not comparable across machines (a laptop idles with tens of
    GiB paged out and is perfectly healthy), so callers compare against the
    baseline taken when the run started. Growth is the signal: a benchmark that
    pushes the host into fresh swap is both measuring garbage and walking toward
    the memory-accounting panic that killed the M4 on 2026-08-22.
    """
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    except Exception:
        return None
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", out)
    if not match:
        return None
    scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def thermal_cpu_speed_limit() -> Optional[int]:
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"], text=True)
    except Exception:
        return None
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    return int(match.group(1)) if match else None


def host_snapshot() -> Dict[str, Any]:
    load1, load5, _ = os.getloadavg()
    return {
        "loadavg_1m": round(load1, 2),
        "loadavg_5m": round(load5, 2),
        "memory_free_pct": memory_free_percent(),
        "swap_used_bytes": swap_used_bytes(),
        "thermal_cpu_speed_limit": thermal_cpu_speed_limit(),
    }


def quiet_machine_violations(
    snapshot: Dict[str, Any],
    max_load: float,
    min_free_pct: float,
    swap_baseline_bytes: Optional[int] = None,
    max_swap_growth_bytes: Optional[int] = None,
) -> List[str]:
    problems: List[str] = []
    if (
        swap_baseline_bytes is not None
        and max_swap_growth_bytes
        and snapshot.get("swap_used_bytes") is not None
    ):
        growth = snapshot["swap_used_bytes"] - swap_baseline_bytes
        if growth > max_swap_growth_bytes:
            problems.append(
                f"swap grew {growth / 2 ** 30:.1f} GiB since the run started "
                f"(> {max_swap_growth_bytes / 2 ** 30:.1f} GiB)"
            )
    if snapshot["loadavg_1m"] > max_load:
        problems.append(f"loadavg_1m {snapshot['loadavg_1m']} > {max_load}")
    free = snapshot.get("memory_free_pct")
    if free is not None and free < min_free_pct:
        problems.append(f"memory_free_pct {free} < {min_free_pct}")
    limit = snapshot.get("thermal_cpu_speed_limit")
    if limit is not None and limit < 100:
        problems.append(f"thermal CPU_Speed_Limit {limit} < 100")
    return problems


class Footprint:
    """Physical footprint of a process via libproc `proc_pid_rusage` (RUSAGE_INFO_V4).

    Offsets: 16-byte uuid, then uint64 fields. `ri_phys_footprint` is the 8th
    uint64, `ri_lifetime_max_phys_footprint` the 29th (see <sys/resource.h>).
    """

    RUSAGE_INFO_V4 = 4
    _PHYS_INDEX = 7
    _LIFETIME_MAX_INDEX = 28

    def __init__(self) -> None:
        # libproc is macOS-only. A missing library must not be fatal at
        # construction: CI's ubuntu job runs `--dry-run` through main(), which
        # builds this object but never measures. read() returning None on such
        # a host is honest — no footprint, rather than no harness.
        try:
            path = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
            self._lib = ctypes.CDLL(path)
            self._lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self._lib.proc_pid_rusage.restype = ctypes.c_int
        except OSError:
            self._lib = None

    def read(self, pid: int) -> Optional[Dict[str, int]]:
        if self._lib is None:
            return None
        buffer = ctypes.create_string_buffer(1024)
        rc = self._lib.proc_pid_rusage(pid, self.RUSAGE_INFO_V4, buffer)
        if rc != 0:
            return None
        raw = buffer.raw

        def field(index: int) -> int:
            offset = 16 + 8 * index
            return int.from_bytes(raw[offset : offset + 8], "little")

        return {
            "phys_footprint": field(self._PHYS_INDEX),
            "lifetime_max_phys_footprint": field(self._LIFETIME_MAX_INDEX),
        }


class FootprintSampler:
    """Polls the engine pid's phys footprint on a thread; reports the max seen."""

    def __init__(self, footprint: Footprint, pid: Optional[int], interval: float = 0.05) -> None:
        self._footprint = footprint
        self._pid = pid
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak = 0
        self.samples = 0

    def __enter__(self) -> "FootprintSampler":
        if self._pid:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            reading = self._footprint.read(self._pid or 0)
            if reading:
                self.peak = max(self.peak, reading["phys_footprint"])
                self.samples += 1
            self._stop.wait(self._interval)

    def lifetime_max(self) -> Optional[int]:
        if not self._pid:
            return None
        reading = self._footprint.read(self._pid)
        return reading["lifetime_max_phys_footprint"] if reading else None


def recorded_cells(results_dir: Path, device_model: str) -> set:
    """Cells this device already has a *good* row for, keyed the way the
    leaderboard keys them. Feeds `--resume`, so a host that dies mid-matrix costs
    only the cells it had left."""
    done = set()
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("valid_for_leaderboard"):
                continue
            if (row.get("device") or {}).get("model") != device_model:
                continue
            workload = row.get("workload") or {}
            engine_row = row.get("engine") or {}
            done.add((
                engine_row.get("name"),
                engine_row.get("build_id") or engine_row.get("version"),
                (row.get("model") or {}).get("id"),
                workload.get("context_tier"),
                int(workload.get("concurrency") or 1),
                workload.get("cache_mode") or "cold",
                int(workload.get("max_tokens") or 0),
            ))
    return done


def sync_results(args: argparse.Namespace, out_path: Path, engine_name: str) -> None:
    """Checkpoint finished rows off this machine while there is still a machine.

    Rows are appended per cell, so nothing is lost to a crash locally — but the
    file is only useful where someone can read it, and on 2026-08-22 that turned
    out to be nowhere for twelve hours. The command is the caller's choice (git,
    rsync, scp) so the harness carries no opinion about where results live.
    """
    if not args.sync_after_engine:
        return
    environment = dict(os.environ, MLXCAT_BENCH_RESULT=str(out_path), MLXCAT_BENCH_ENGINE=engine_name)
    try:
        completed = subprocess.run(
            args.sync_after_engine, shell=True, env=environment,
            capture_output=True, text=True, timeout=600,
        )
    except Exception as error:  # noqa: BLE001
        print(f"[{engine_name}] result sync failed to start: {error}", file=sys.stderr)
        return
    if completed.returncode == 0:
        print(f"[{engine_name}] results synced")
    else:
        print(
            f"[{engine_name}] result sync exited {completed.returncode} — rows are still in "
            f"{out_path}\n{(completed.stderr or completed.stdout).strip()[:500]}",
            file=sys.stderr,
        )


def host_violations(snapshot: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    """`quiet_machine_violations` bound to the run's flags and swap baseline."""
    return quiet_machine_violations(
        snapshot,
        args.max_load,
        args.min_free_pct,
        getattr(args, "_swap_baseline_bytes", None),
        getattr(args, "_swap_growth_invalid_bytes", None),
    )


class RunawayGuard:
    """Kills an engine process before it can take the host down with it.

    On 2026-08-22 vllm-mlx panicked the macOS GPU driver on the M4 Pro
    (`completeMemory() prepare count underflow` @IOGPUMemory.cpp:550) roughly
    nine minutes after its first cell timed out, and the reboot took the three
    benchmark passes queued behind it. Nothing in the harness was watching: the
    quiet-machine guard samples between cells, and `FootprintSampler` only runs
    during the measured requests — not during launch, calibration, the discarded
    cold request, or the warmup, which is exactly where that engine died.

    So this runs for the whole life of an engine process, at 1 Hz, and kills it
    on either of the two signals that precede a host death:

    * the engine's own physical footprint crosses a fraction of installed RAM;
    * the machine has paged out materially more than it had when the run began.

    Swap is measured as *growth* from a baseline because absolute swap is not
    comparable between machines — a laptop can idle with 20 GiB paged out and be
    fine, while the M4 worker sits at zero.

    A breach is recorded, not just acted on: the cell it interrupts is written
    with `invalid_reason`, and the caller abandons the engine rather than
    relaunching into the same wall.
    """

    def __init__(
        self,
        footprint: "Footprint",
        pid: Optional[int],
        label: str,
        cap_bytes: Optional[int],
        swap_baseline_bytes: Optional[int],
        swap_growth_kill_bytes: Optional[int],
        interval: float = 1.0,
    ) -> None:
        self._footprint = footprint
        self._pid = pid
        self._label = label
        self._cap = cap_bytes
        self._swap_baseline = swap_baseline_bytes
        self._swap_kill = swap_growth_kill_bytes
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.breach: Optional[str] = None

    def start(self) -> "RunawayGuard":
        if self._pid and (self._cap or self._swap_kill):
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            reason = self._check()
            if reason:
                self.breach = reason
                print(f"[{self._label}] RUNAWAY GUARD: {reason} — killing pid {self._pid}", file=sys.stderr)
                self._kill()
                return
            self._stop.wait(self._interval)

    def _check(self) -> Optional[str]:
        if self._cap:
            reading = self._footprint.read(self._pid or 0)
            if reading is None:
                return None  # process already gone; nothing to guard
            used = reading["phys_footprint"]
            if used > self._cap:
                return (
                    f"engine footprint {used / 2 ** 30:.1f} GiB exceeded the cap "
                    f"{self._cap / 2 ** 30:.1f} GiB"
                )
        if self._swap_kill and self._swap_baseline is not None:
            current = swap_used_bytes()
            if current is not None and current - self._swap_baseline > self._swap_kill:
                return (
                    f"host swapped {(current - self._swap_baseline) / 2 ** 30:.1f} GiB "
                    f"since the run started (> {self._swap_kill / 2 ** 30:.1f} GiB)"
                )
        return None

    def _kill(self) -> None:
        if not self._pid:
            return
        try:
            os.kill(self._pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(
                f"[{self._label}] cannot kill pid {self._pid} (not ours) — stop it by hand",
                file=sys.stderr,
            )


def binary_build_id(path: str) -> Optional[str]:
    """Content hash of the engine binary, short.

    mlxcat-http reports no version on any endpoint we probe, so `engine.version`
    is null on every row it produces and a row cannot be attributed to a build.
    That is not cosmetic: `--resume` skipped 220 cells on 2026-08-22 because they
    matched on (engine, model, tier, width, cache mode) — and every one of them
    had been measured by a DIFFERENT binary, the one from before the allocator
    fix. The run designed to re-measure them reused them instead.

    Hashing the file we are about to launch fixes both: rows carry the identity
    of the build that produced them, and resume stops recognising cells measured
    by a different one.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()[:12]
    except OSError:
        return None


def reject_debug_build(binary: str, engine_name: str) -> None:
    """Refuse to benchmark a debug binary of our own engine.

    mlx-serve's benchmark notes put it plainly: "Debug is 2-4x slower = a fake
    regression". Our launcher resolves `{repo}/.build/release/...`, but
    MLXCAT_HTTP_BIN can point anywhere, and `swift build` without `-c release`
    leaves a debug binary sitting in a path a tired operator will happily export.
    A silent 3x handicap on our own engine is the worst possible measurement
    error here: it looks exactly like losing.
    """
    resolved = os.path.realpath(binary)
    if f"{os.sep}debug{os.sep}" in resolved + os.sep:
        raise EngineUnavailable(
            f"{engine_name}: {resolved} is a DEBUG build — 2-4x slower than release and "
            f"indistinguishable from a real regression in the results. "
            f"Build with `swift build -c release` and re-point MLXCAT_HTTP_BIN."
        )


def ensure_metallib(binary: str) -> str:
    """`swift build` products need mlx.metallib beside them; build it once if missing.

    Only applies to binaries under this repo's .build (anything else is assumed to
    ship its own). Mirrors Tests/MLXCatTests/Support/MLXMetalRuntime.swift.
    """
    binary_dir = os.path.dirname(os.path.abspath(binary))
    if not binary_dir.startswith(str(REPO / ".build")):
        return binary
    if os.path.exists(os.path.join(binary_dir, "mlx.metallib")):
        return binary
    script = REPO / "scripts" / "build-metallib.sh"
    print(f"mlx.metallib missing beside {os.path.basename(binary)} — running {script.name} (one-time)")
    subprocess.run(["zsh", str(script), binary_dir], check=True)
    return binary


def pid_listening_on(port: int) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    pids = [int(p) for p in out.split() if p.strip().isdigit()]
    return pids[0] if pids else None


def free_port(start: int) -> int:
    port = start
    while port < start + 200:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("no free port found")


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #


class Engine:
    def __init__(self, name: str, spec: Dict[str, Any], args: argparse.Namespace, model_root: Path) -> None:
        self.name = name
        self.spec = spec
        self.args = args
        self.model_root = model_root
        self.url: str = spec.get("url") or ""
        self.token: str = os.environ.get(spec.get("token_env", ""), "") if spec.get("token_env") else ""
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.log_path: Optional[Path] = None
        self.version: Optional[str] = None
        self.port: Optional[int] = None
        self.guard: Optional[RunawayGuard] = None
        self.build_id: Optional[str] = None

    # -- lifecycle -------------------------------------------------------- #

    def launch_for(self, model: Dict[str, Any], log_dir: Path) -> None:
        """Start one engine process for one model (so peak memory is per model)."""
        launch = self.spec.get("launch")
        if not launch:
            if not self.url:
                raise RuntimeError(f"{self.name}: neither 'launch' nor 'url' configured")
            self.pid = pid_listening_on(self._port_from_url(self.url))
            self._arm_guard()
            self.wait_ready()
            return
        binary = self._resolve_binary(launch["bin"])
        if binary is None:
            raise EngineUnavailable(f"{self.name}: binary not found ({launch['bin']}); set {launch.get('bin_env', 'the path')}")
        self.build_id = binary_build_id(binary)
        self.port = free_port(int(launch.get("port_base", 11700)))
        variables = {
            "port": str(self.port),
            "model_root": str(self.model_root),
            "model_dir": str(self.model_root / model["id"]),
            "model_id": model["id"],
            "memory_ceiling_bytes": str(self.args.memory_ceiling_bytes or 0),
        }
        argv = [binary] + [a.format(**variables) for a in launch["args"]]
        env = dict(os.environ)
        for key, value in (launch.get("env") or {}).items():
            env[key] = value.format(**variables)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{self.name}-{model['id']}.log"
        handle = open(self.log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
        self.pid = self.process.pid
        self.url = f"http://127.0.0.1:{self.port}"
        self._arm_guard()
        self.wait_ready(timeout=float(launch.get("ready_timeout_s", 180)))

    def _arm_guard(self) -> None:
        """Watch this process for the whole time it exists, not just while a cell
        is being measured — the engine that killed the host died during launch."""
        guard_context = getattr(self.args, "_guard_context", None)
        if not guard_context or not self.pid:
            return
        self.guard = RunawayGuard(
            footprint=guard_context["footprint"],
            pid=self.pid,
            label=self.name,
            cap_bytes=guard_context["cap_bytes"],
            swap_baseline_bytes=guard_context["swap_baseline_bytes"],
            swap_growth_kill_bytes=guard_context["swap_growth_kill_bytes"],
        ).start()

    def stop(self) -> None:
        if self.guard:
            self.guard.stop()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def wait_ready(self, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        last_error = "not started"
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited early (rc={self.process.returncode}); see {self.log_path}")
            try:
                self.request_json("/v1/models", timeout=5)
                return
            except Exception as error:  # noqa: BLE001
                last_error = str(error)
                time.sleep(0.5)
        raise RuntimeError(f"{self.name} not ready after {timeout:.0f}s: {last_error}")

    # -- helpers ---------------------------------------------------------- #

    def _resolve_binary(self, spec: str) -> Optional[str]:
        env_name = self.spec.get("launch", {}).get("bin_env")
        if env_name and os.environ.get(env_name):
            reject_debug_build(os.environ[env_name], self.name)
            return ensure_metallib(os.environ[env_name])
        candidate = spec.format(repo=str(REPO))
        if os.path.sep in candidate:
            if not os.path.exists(candidate):
                return None
            reject_debug_build(candidate, self.name)
            return ensure_metallib(candidate)
        return shutil.which(candidate)

    @staticmethod
    def _port_from_url(url: str) -> int:
        match = re.search(r":(\d+)", url.split("//", 1)[-1])
        return int(match.group(1)) if match else 80

    def headers(self, accept: str) -> Dict[str, str]:
        headers = {"Accept": accept, "Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def request_json(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 30) -> Dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.url.rstrip("/") + path, data=data, headers=self.headers("application/json"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")

    def resolve_model(self, requested: str) -> Optional[str]:
        body = self.request_json("/v1/models")
        offered = [item.get("id") for item in body.get("data") or [] if item.get("id")]
        leaf = requested.rsplit("/", 1)[-1]
        for candidate in (requested, leaf):
            if candidate in offered:
                return candidate
        for item in offered:
            if item.rsplit("/", 1)[-1] == leaf:
                return item
        aliases = self.spec.get("model_aliases") or {}
        alias = aliases.get(requested) or aliases.get(leaf)
        if alias and alias in offered:
            return alias
        return None

    def detect_version(self) -> Optional[str]:
        for path in ("/v1/local/version", "/version", "/api/version", "/health"):
            try:
                body = self.request_json(path, timeout=5)
            except Exception:  # noqa: BLE001
                continue
            for key in ("version", "build", "engine"):
                if isinstance(body.get(key), str):
                    return body[key]
        return self.spec.get("version")

    # -- measurement ------------------------------------------------------ #

    def stream_once(self, model: str, prompt: str, max_tokens: int, timeout: float, nonce: bool = True) -> Dict[str, Any]:  # noqa: D401
        # Unique per-request prefix so no engine can serve the measurement from a
        # prompt/prefix cache (mlxcat's tiered prefix cache would otherwise turn
        # repeated identical prompts into cache-hit TTFTs while other engines do
        # real prefill — Codex round 1 BLOCKER). Uniform across engines.
        if nonce:
            prompt = f"[request {uuid.uuid4().hex[:12]}] " + prompt
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # No hidden per-engine defaults: engines without extra_request_fields
        # get NOTHING extra, same as the four rivals whose specs already said
        # {}. The old default ({"enable_thinking": false}) reached only mlxcat,
        # so on hybrid-thinking models (qwen3.5/3.8) our longgen rows answered
        # and EOS'd near 520 tokens while every rival thought its way to the
        # full 1024 budget — two different workloads wearing one cell, worth
        # 5-8 board points against us (2026-08-24 cliff analysis; the
        # finish_reasons metric exists to catch exactly this class).
        extra = self.spec.get("extra_request_fields") or {}
        body.update(extra)
        request = urllib.request.Request(
            self.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=self.headers("text/event-stream"),
        )
        started = time.perf_counter()
        first_output: Optional[float] = None
        chunk_times: List[float] = []
        usage: Dict[str, Any] = {}
        chars = 0
        finish_reason: Optional[str] = None
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(text, str) and text:
                        now = time.perf_counter()
                        chars += len(text)
                        if first_output is None:
                            first_output = now
                        chunk_times.append(now)
        finished = time.perf_counter()
        if first_output is None:
            raise RuntimeError(f"{self.name}: no visible output")
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if completion_tokens <= 0:
            raise RuntimeError(f"{self.name}: usage.completion_tokens missing (needed for honest tok/s)")
        # Decode ends at the LAST OUTPUT CHUNK, not at [DONE]/usage framing — an
        # engine that dawdles over its terminal frames must not read as slower decode.
        # An engine that buffers the whole completion into one visible chunk has no
        # measurable decode window over this transport: decode_tps is None, never a
        # nonsense number from a ~0 denominator.
        last_output = chunk_times[-1] if chunk_times else finished
        decode_seconds = last_output - first_output
        # Token-granular streaming only: if the engine coalesced tokens into a few
        # buffered chunks, the chunk cadence does not observe decode speed (two
        # chunks 2 µs apart carrying 128 tokens is not 64M tok/s). Require about
        # one chunk per token before publishing a decode rate.
        completion_estimate = int(usage.get("completion_tokens") or 0)
        raw_rate = (completion_estimate - 1) / decode_seconds if decode_seconds > 0 else float("inf")
        # Plausibility bound, not proof: chunk cadence cannot strictly prove token
        # timing (a server could burst-flush many chunks), so on top of the
        # one-chunk-per-token requirement the window must be >= 50 ms and the
        # implied rate below 2,500 tok/s — an order of magnitude above any decode
        # rate ever measured on this hardware class. Anything outside is reported
        # as unmeasurable, never as a record-breaking number.
        decode_measurable = (
            len(chunk_times) >= max(2, completion_estimate // 2)
            and decode_seconds >= 0.05
            and raw_rate <= 2500
        )
        ttft = first_output - started
        gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
        return {
            # Absolute perf_counter marks. A concurrency leg needs them to tell
            # prefill wall from decode wall: with per-row serial prefill the last
            # request's first token can arrive after the first request has been
            # decoding for seconds, and a single tokens/wall figure blends the two
            # into a number that reports admission latency and calls it throughput.
            "t_start": started,
            "t_first": first_output,
            "t_last": last_output,
            "ttft_ms": ttft * 1000,
            "wall_ms": (finished - started) * 1000,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            # Why the request ended ("stop" = EOS, "length" = max_tokens). At
            # temp 0 on longgen, some engines run all 1024 tokens while others
            # EOS near 520 for the same model — aggregate tok/s over different
            # token mixes is a workload comparison, not an engine one. Recording
            # this makes the confound visible per row instead of discovered by
            # archaeology (2026-08-24 concurrency-cliff analysis).
            "finish_reason": finish_reason,
            "prefill_tps": (prompt_tokens / ttft) if prompt_tokens and ttft > 0 else None,
            "decode_tps": (completion_tokens - 1) / decode_seconds if (completion_tokens > 1 and decode_measurable) else None,
            "e2e_tps": completion_tokens / max(finished - started, 1e-9),
            "itl_p50_ms": statistics.median(gaps) * 1000 if gaps else None,
            "itl_p95_ms": percentile(gaps, 95) * 1000 if gaps else None,
            "chunks": len(chunk_times),
            "chars": chars,
            "server_tps": usage.get("generation_tokens_per_second") or usage.get("tokens_per_second"),
        }


class EngineUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Stats + prompts
# --------------------------------------------------------------------------- #


def percentile(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def spread(values: List[Optional[float]]) -> Optional[Dict[str, float]]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    ordered = sorted(clean)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "n": len(ordered),
        "spread_ratio": ordered[-1] / max(ordered[0], 1e-9),
    }


def build_prompt(target_tokens: int, chars_per_token: float) -> str:
    if target_tokens <= 0:
        return QUESTION
    budget_chars = max(int(target_tokens * chars_per_token) - len(QUESTION), 0)
    repeats = max(budget_chars // len(FILLER), 0)
    return (FILLER * repeats) + QUESTION


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def git_commit(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def concurrency_metrics(rows: List[Dict[str, Any]], sla_ttft_ms: float, sla_tpot_ms: float) -> Dict[str, Any]:
    """Decompose one concurrent burst the way oMLX and vLLM do.

    A single `total tokens / wall` figure is what this harness shipped first, and
    it is close to meaningless when prefill is serial: request 8's first token can
    land after request 1 has been decoding for seconds, so the ratio mostly
    reports how long admission took. Splitting the two windows is what makes
    "batching is worth +22%" impossible to conclude from a short-tier burst.

      prefill wall  = first token of the LAST request  -  earliest request start
      decode  wall  = last token of the LAST request   -  first token of the LAST request

    Percentiles and goodput come from vLLM's serving benchmark: a median hides the
    tail, and tail latency is exactly where serial admission hurts. Goodput counts
    only the requests that met an SLA — throughput bought by making some requests
    unusably slow should not read as throughput.
    """
    if not rows:
        return {}
    start = min(r["t_start"] for r in rows)
    last_first = max(r["t_first"] for r in rows)
    end = max(r["t_last"] for r in rows)
    prompt_tokens = sum(int(r["prompt_tokens"]) for r in rows)
    completion_tokens = sum(int(r["completion_tokens"]) for r in rows)
    ttfts = sorted((r["t_first"] - r["t_start"]) * 1000 for r in rows)
    tpots = [
        ((r["t_last"] - r["t_first"]) * 1000) / (int(r["completion_tokens"]) - 1)
        for r in rows
        if int(r["completion_tokens"]) > 1 and r["t_last"] > r["t_first"]
    ]
    prefill_wall = max(last_first - start, 1e-9)
    decode_wall = max(end - last_first, 1e-9)
    met = sum(
        1
        for r in rows
        if (r["t_first"] - r["t_start"]) * 1000 <= sla_ttft_ms
        and (
            int(r["completion_tokens"]) <= 1
            or ((r["t_last"] - r["t_first"]) * 1000) / (int(r["completion_tokens"]) - 1) <= sla_tpot_ms
        )
    )
    return {
        "pp_tps": prompt_tokens / prefill_wall,
        "decode_agg_tps": completion_tokens / decode_wall,
        "prefill_wall_ms": prefill_wall * 1000,
        "decode_wall_ms": decode_wall * 1000,
        "ttft_mean_ms": statistics.fmean(ttfts),
        "ttft_p95_ms": percentile(ttfts, 95),
        "tpot_mean_ms": statistics.fmean(tpots) if tpots else None,
        "tpot_p95_ms": percentile(tpots, 95) if tpots else None,
        "goodput_frac": met / len(rows),
    }


def run_cell(
    engine: Engine,
    offered: str,
    model: Dict[str, Any],
    tier: Dict[str, Any],
    concurrency: int,
    prompt: str,
    args: argparse.Namespace,
    sampler_factory,
    cache_mode: str = "cold",
) -> Dict[str, Any]:
    """cache_mode 'cold' gives every request a unique nonce prefix so no prefix
    cache can serve it — the fair cross-engine default. 'warm' repeats the SAME
    prompt so an engine's prefix cache CAN hit, which is the only way to measure
    the feature mlxcat and oMLX both exist for: does turn 2 of a conversation get
    cheaper? The warm cell deliberately primes with the discarded cold request
    first, so the measured runs are the hits."""
    nonce = cache_mode == "cold"
    max_tokens = int(tier.get("max_tokens", args.max_tokens))
    timeout = float(args.request_timeout)
    # warm: first call is discarded (cold shapes/compilation), then `warmup` more.
    cold = engine.stream_once(offered, prompt, max_tokens, timeout, nonce=nonce)
    for _ in range(args.warmup):
        engine.stream_once(offered, prompt, max_tokens, timeout, nonce=nonce)

    # Arrival process. Firing N requests at the same instant is a closed-loop
    # burst — the worst case for an engine that admits serially, and not how a
    # server is actually used. vLLM's serving benchmark models an OPEN loop
    # instead: requests arrive at a rate, with a burstiness factor shaping the
    # gap distribution (gamma; 1.0 is Poisson, <1 burstier, >1 smoother). Both
    # are worth measuring, so --request-rate 0 keeps the burst and any positive
    # rate staggers arrivals.
    rate = float(getattr(args, "request_rate", 0) or 0)
    burstiness = max(float(getattr(args, "burstiness", 1.0) or 1.0), 1e-6)

    def fire(index: int) -> Dict[str, Any]:
        if rate > 0 and index > 0:
            delay = sum(
                random.gammavariate(burstiness, (1.0 / rate) / burstiness)
                for _ in range(index)
            )
            time.sleep(delay)
        return engine.stream_once(offered, prompt, max_tokens, timeout, nonce=nonce)

    runs: List[Dict[str, Any]] = []
    aggregate_samples: List[float] = []
    burst_metrics: List[Dict[str, Any]] = []
    with sampler_factory() as sampler:
        for _ in range(args.runs):
            if concurrency == 1:
                runs.append(engine.stream_once(offered, prompt, max_tokens, timeout, nonce=nonce))
            else:
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows = list(pool.map(fire, range(concurrency)))
                elapsed = time.perf_counter() - started
                total = sum(int(r["completion_tokens"]) for r in rows)
                aggregate_samples.append(total / max(elapsed, 1e-9))
                burst_metrics.append(
                    concurrency_metrics(rows, args.sla_ttft_ms, args.sla_tpot_ms)
                )
                runs.extend(rows)
        peak_sampled = sampler.peak
        lifetime_max = sampler.lifetime_max()

    metrics: Dict[str, Any] = {
        "ttft_ms": spread([r["ttft_ms"] for r in runs]),
        "prefill_tps": spread([r["prefill_tps"] for r in runs]),
        "decode_tps": spread([r["decode_tps"] for r in runs]),
        "e2e_tps": spread([r["e2e_tps"] for r in runs]),
        "itl_p50_ms": spread([r["itl_p50_ms"] for r in runs]),
        "itl_p95_ms": spread([r["itl_p95_ms"] for r in runs]),
        "server_tps": spread([r["server_tps"] for r in runs]),
        "prompt_tokens": int(statistics.median([r["prompt_tokens"] for r in runs])),
        "completion_tokens": int(statistics.median([r["completion_tokens"] for r in runs])),
        "cold_first_request_ms": cold["wall_ms"],
        "peak_phys_footprint_bytes": peak_sampled or None,
        "lifetime_max_phys_footprint_bytes": lifetime_max,
        # Count per reason ({"stop": 5, "length": 1, "unreported": ...}) so a
        # cell whose rows EOS early is distinguishable from one that ran its
        # full token budget without re-deriving it from completion_tokens.
        "finish_reasons": dict(
            collections.Counter(r.get("finish_reason") or "unreported" for r in runs)
        ),
    }
    if concurrency > 1:
        metrics["aggregate_tps"] = spread(aggregate_samples)
        for key in (
            "pp_tps", "decode_agg_tps", "prefill_wall_ms", "decode_wall_ms",
            "ttft_mean_ms", "ttft_p95_ms", "tpot_mean_ms", "tpot_p95_ms", "goodput_frac",
        ):
            metrics[key] = spread([b.get(key) for b in burst_metrics])
    return metrics


def run_producer(
    engine_name: str,
    spec: Dict[str, Any],
    models: List[str],
    model_specs: Dict[str, Dict[str, Any]],
    tiers: Dict[str, Dict[str, Any]],
    chosen_tiers: List[str],
    model_root: Path,
    args: argparse.Namespace,
    stamp: Dict[str, Any],
    out_path: Path,
) -> int:
    """Drive an in-process producer (e.g. mlxcat-baseline) that prints JSONL rows itself.

    The producer measures in-process (transport "in-process"); this function only
    stamps device/host/validity/harness and maps its prompt-token targets back to
    the matrix tier names so the leaderboard can place the rows.
    """
    producer = spec["producer"]
    env_name = producer.get("bin_env")
    binary = os.environ.get(env_name) if env_name else None
    if not binary:
        candidate = producer["bin"].format(repo=str(REPO))
        binary = candidate if os.path.exists(candidate) else shutil.which(candidate)
    if not binary:
        print(f"[{engine_name}] producer binary not found ({producer['bin']}); build it or set {env_name}", file=sys.stderr)
        return 0
    binary = ensure_metallib(binary)
    written = 0
    target_to_tier = {int(tiers[t]["prompt_tokens"]): t for t in chosen_tiers}
    for model_id in models:
        model_dir = model_root / model_id
        if not model_dir.exists():
            print(f"[{engine_name}/{model_id}] model dir missing under {model_root} — skipped")
            continue
        variables = {
            "model_dir": str(model_dir),
            "model_id": model_id,
            "max_tokens": str(args.max_tokens),
            "runs": str(args.runs),
            "warmup": str(args.warmup),
            # Producers must measure under the same memory policy as launched
            # engines or the two transports are not comparable.
            "memory_ceiling_bytes": str(args.memory_ceiling_bytes or 0),
        }
        argv = [binary] + [a.format(**variables) for a in producer["args"]]
        flag = producer.get("cell_flag", "--cell")
        for tier_name in chosen_tiers:
            tier = tiers[tier_name]
            argv += [flag, f"{int(tier['prompt_tokens'])}:{int(tier.get('max_tokens', args.max_tokens))}"]
        print(f"[{engine_name}/{model_id}] {' '.join(os.path.basename(a) if i == 0 else a for i, a in enumerate(argv))}")
        # Stream rows as the producer finishes each cell, so validity is stamped
        # with a snapshot taken when THAT cell's numbers were produced — a host
        # that gets loud mid-producer must not leave later tiers marked valid.
        lines: List[str] = []
        # stderr goes to a FILE, not a pipe — a chatty producer (MLX diagnostics)
        # would fill an undrained pipe and deadlock against our stdout read loop.
        # A kill-timer bounds the whole read, since `for line in stdout` has no
        # timeout of its own.
        stderr_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"mlxcat-bench-producer-{engine_name}-{os.getpid()}.err"
        timed_out = False
        with open(stderr_path, "w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr_handle, text=True)
            # The timer holds a BOUND method of THIS process object (created after
            # Popen), so an expiring timer can never late-bind to a later model's
            # process; join() after cancel closes the fired-vs-cancelled race.
            deadline = threading.Timer(float(args.request_timeout) * 4, process.kill)
            deadline.start()
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    if line.strip().startswith("{"):
                        lines.append(line.strip() + "\t" + json.dumps(host_snapshot()))
                process.wait()
            finally:
                deadline.cancel()
                deadline.join(timeout=2)
                timed_out = process.returncode == -9
        if timed_out:
            print(f"[{engine_name}/{model_id}] producer timed out (killed)", file=sys.stderr)
        if process.returncode != 0:
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace").strip()[-400:]
            print(f"[{engine_name}/{model_id}] producer rc={process.returncode}: {stderr_tail}", file=sys.stderr)
        for tagged in lines:
            line, _, snap_json = tagged.partition("\t")
            row_snapshot = json.loads(snap_json)
            row_violations = host_violations(row_snapshot, args)
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            target = int(record.get("workload", {}).get("prompt_tokens_target", 0))
            record["workload"]["context_tier"] = target_to_tier.get(target, record["workload"].get("context_tier"))
            record["model"] = {**record.get("model", {}), **{k: v for k, v in model_specs.get(model_id, {}).items() if k != "id"}, "id": model_id}
            record["engine"]["name"] = engine_name
            record.update(stamp)
            record["host"] = row_snapshot
            record["valid_for_leaderboard"] = not row_violations and "error" not in record.get("metrics", {})
            record["invalid_reason"] = "; ".join(row_violations) or None
            with open(out_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1
            m = record["metrics"]
            dec = (m.get("decode_tps") or {}).get("median")
            pre = (m.get("prefill_tps") or {}).get("median")
            print(f"[{engine_name}/{model_id}/{record['workload']['context_tier']}/c1] prompt {m.get('prompt_tokens')} tok · TTFT {((m.get('ttft_ms') or {}).get('median') or 0):.0f} ms · prefill {pre or 0:.0f} tok/s · decode {dec or 0:.1f} tok/s · peak {(m.get('peak_phys_footprint_bytes') or 0) / 2**30:.2f} GiB (in-process)")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", default="mlxcat", help="comma-separated engine names from engines.json (default: mlxcat)")
    parser.add_argument("--engine-url", action="append", default=[], metavar="NAME=URL", help="override/attach an engine at a running URL (repeatable)")
    parser.add_argument("--models", default="", help="comma-separated model ids (default: matrix.json 'default' set)")
    parser.add_argument("--model-set", default="default", help="named model set in matrix.json (default, smoke, flagship, all)")
    parser.add_argument("--contexts", default="", help="comma-separated context tiers (default: matrix.json)")
    parser.add_argument("--concurrency", default="", help="comma-separated widths for the concurrency leg (default: matrix.json)")
    parser.add_argument("--concurrency-tier", default="", help="comma-separated context tier(s) for the concurrency leg (default: matrix.json concurrency.tiers)")
    parser.add_argument("--cache-modes", default="cold", help="cold (unique prompt per request; no prefix cache can hit) and/or warm (repeated prompt; measures prefix-cache reuse). Comma-separated.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--model-root", default=os.environ.get("MLXCAT_BENCH_MODEL_ROOT", str(Path.home() / "Library/Caches/models/mlx-community")))
    parser.add_argument("--memory-ceiling-bytes", type=int, default=0, help="passed to launchers that support a ceiling (0 = engine default)")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--allow-quarantined",
        action="store_true",
        help="run engines marked 'quarantined' in engines.json. They are quarantined because they "
             "destabilised the HOST, not because they scored badly — run them alone, never ahead of "
             "work you care about.",
    )
    parser.add_argument("--engines-file", default=str(DEFAULT_ENGINES))
    parser.add_argument("--matrix-file", default=str(DEFAULT_MATRIX))
    parser.add_argument("--max-load", type=float, default=8.0, help="quiet-machine guard: 1-minute load average ceiling")
    parser.add_argument("--min-free-pct", type=float, default=35.0, help="quiet-machine guard: memory_pressure free %% floor")
    parser.add_argument("--allow-loaded", action="store_true", help="run even if the guard trips; rows are marked invalid for the leaderboard")
    parser.add_argument(
        "--wait-for-quiet",
        type=float,
        default=0,
        help="seconds to wait for the host to go quiet before giving up, instead of refusing "
             "immediately. Replaces the shell poll loops the campaign passes used to carry, which "
             "reset their counter on a single blip and could wait forever on a host that also runs "
             "CI. 0 (default) keeps the old refuse-now behaviour.",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="a named plan from matrix.json profiles: 'quick' for a few-minute iteration loop, "
             "'default' for the weekly matrix, 'full' for the long ladder. Sets models, contexts "
             "and concurrency together; explicit flags still win.",
    )
    parser.add_argument(
        "--engine-memory-cap-pct",
        type=float,
        default=92.0,
        help="runaway guard: SIGKILL an engine whose physical footprint passes this %% of installed "
             "RAM. The highest honest row we have measured is 90.8%% (mlx-serve, gemma-4-12B, c4 on "
             "48 GiB), so this sits just above it: it is a host-survival line, not a memory budget. "
             "0 disables.",
    )
    parser.add_argument(
        "--swap-growth-kill-gb",
        type=float,
        default=8.0,
        help="runaway guard: SIGKILL an engine once the host has paged out this much MORE than it "
             "had when the run started. Growth, not absolute, because a laptop idles with tens of "
             "GiB swapped and the worker idles at zero. 0 disables.",
    )
    parser.add_argument(
        "--swap-growth-invalid-gb",
        type=float,
        default=2.0,
        help="quiet-machine guard: rows measured after the host swapped this much are recorded but "
             "not ranked. Well below the kill line — thrash corrupts numbers long before it "
             "threatens the machine.",
    )
    parser.add_argument(
        "--engine-failure-budget",
        type=int,
        default=3,
        help="abandon an engine after this many consecutive failed cells. vllm-mlx was allowed to "
             "keep failing for nine minutes before it panicked the host; a sick engine gets a short "
             "leash and the run moves on to the next one.",
    )
    parser.add_argument(
        "--sync-after-engine",
        default="",
        metavar="CMD",
        help="shell command run after each engine finishes, with MLXCAT_BENCH_RESULT set to the "
             "output file. The 2026-08-22 panic stranded 160 finished rows on an unreachable "
             "machine for twelve hours because the suite only synced at the end; an engine is the "
             "natural checkpoint. Failures are reported, never fatal — a sync problem must not cost "
             "the run.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=0.0,
        help="open-loop arrivals for the concurrency leg, in requests/second. 0 (default) fires the "
             "whole width at once — a closed-loop burst, which is the worst case for serial "
             "admission and not how a server is used. vLLM's serving benchmark measures the open "
             "loop; so should we, at least once, before concluding anything about batching.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="shape of the gap distribution when --request-rate is set: 1.0 is Poisson, below 1 is "
             "burstier, above 1 is smoother.",
    )
    parser.add_argument(
        "--sla-ttft-ms",
        type=float,
        default=2000.0,
        help="goodput SLA: a concurrent request counts only if its TTFT is under this.",
    )
    parser.add_argument(
        "--sla-tpot-ms",
        type=float,
        default=100.0,
        help="goodput SLA: ...and its mean time-per-output-token is under this. Throughput bought "
             "by making some requests unusably slow is not throughput.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cells already recorded for this device in --results-dir. A host that dies "
             "mid-matrix then costs the cells it had left, not the ones it had already paid for.",
    )
    parser.add_argument("--tag", default="", help="free-text tag stored on every record (e.g. 'pin a481734 + ceiling')")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)

    engines_spec = load_json(Path(args.engines_file))["engines"]
    matrix = load_json(Path(args.matrix_file))
    model_root = Path(args.model_root).expanduser()

    # A profile sets the whole plan; an explicit flag still wins over it, so
    # `--profile quick --contexts 16k` means what it says.
    profile = {}
    if args.profile:
        profiles = matrix.get("profiles") or {}
        if args.profile not in profiles:
            named = ", ".join(k for k in profiles if not k.startswith("_"))
            print(f"unknown --profile {args.profile!r} (matrix.json has: {named})", file=sys.stderr)
            return 64
        profile = profiles[args.profile]
        cost = profile.get("_cost")
        print(f"profile {args.profile}: {cost}" if cost else f"profile {args.profile}")
        if not args.models:
            args.models = ",".join(profile["models"])
        if not args.contexts:
            args.contexts = ",".join(profile["contexts"])
        if not args.concurrency:
            args.concurrency = ",".join(str(w) for w in profile["concurrency"])
        if not args.concurrency_tier:
            args.concurrency_tier = ",".join(profile["concurrency_tiers"])
        if parser.get_default("runs") == args.runs:
            args.runs = int(profile["runs"])
        if parser.get_default("warmup") == args.warmup:
            args.warmup = int(profile["warmup"])

    models = [m.strip() for m in args.models.split(",") if m.strip()] or matrix["model_sets"][args.model_set]
    model_specs = {m["id"]: m for m in matrix["models"]}
    tiers = {t["name"]: t for t in matrix["context_tiers"]}
    chosen_tiers = [t.strip() for t in args.contexts.split(",") if t.strip()] or matrix["default_context_tiers"]
    # --cache-modes was parsed, printed in the plan, documented in bench/README.md
    # and never bound to anything: `cache_mode` was read four times inside the cell
    # loop and assigned nowhere, so the first run to reach that line died with a
    # NameError. It was committed at 02:34 on 2026-08-22 and executed for the first
    # time today. Every existing row is cold because no other kind could be produced.
    cache_modes = [m.strip() for m in args.cache_modes.split(",") if m.strip()] or ["cold"]
    unknown_modes = [m for m in cache_modes if m not in ("cold", "warm")]
    if unknown_modes:
        print(f"unknown --cache-modes value(s): {', '.join(unknown_modes)} (expected cold and/or warm)", file=sys.stderr)
        return 64
    widths = [int(w) for w in args.concurrency.split(",") if w.strip()] or matrix["concurrency"]["widths"]
    # Concurrency is measured at every listed tier: at a short tier the per-request
    # serial prefill dominates and aggregate throughput reads as admission latency
    # rather than batched-decode throughput, so a generation-heavy tier is required
    # to see what batching is actually worth.
    if args.concurrency_tier:
        conc_tiers = [t.strip() for t in args.concurrency_tier.split(",") if t.strip()]
    else:
        conc_tiers = matrix["concurrency"].get("tiers") or [matrix["concurrency"]["tier"]]

    overrides = {}
    for item in args.engine_url:
        name, _, url = item.partition("=")
        overrides[name] = url
    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    for name in overrides:
        if name not in engine_names:
            engine_names.append(name)

    snapshot = host_snapshot()
    swap_baseline = snapshot.get("swap_used_bytes")
    physical_memory = int(sysctl("hw.memsize") or 0)
    guard_cap = int(physical_memory * args.engine_memory_cap_pct / 100) if args.engine_memory_cap_pct > 0 else None
    swap_kill = int(args.swap_growth_kill_gb * 2 ** 30) if args.swap_growth_kill_gb > 0 else None
    swap_invalid = int(args.swap_growth_invalid_gb * 2 ** 30) if args.swap_growth_invalid_gb > 0 else None
    args._swap_baseline_bytes = swap_baseline
    args._swap_growth_invalid_bytes = swap_invalid
    args._guard_context = {
        "footprint": Footprint(),
        "cap_bytes": guard_cap,
        "swap_baseline_bytes": swap_baseline,
        "swap_growth_kill_bytes": swap_kill,
    }

    violations = host_violations(snapshot, args)

    # Waiting for quiet belongs here, not in a shell loop around this script.
    # The campaign passes used to poll `load < 5` and require three consecutive
    # readings, resetting to zero on any single blip — on a host that also runs
    # CI that is an unbounded wait, and on 2026-08-22 it burned 90 minutes and
    # produced no rows at all. One implementation, the same thresholds as the
    # guard itself, and a blip costs one interval instead of everything.
    if violations and args.wait_for_quiet > 0 and not args.allow_loaded:
        deadline = time.time() + args.wait_for_quiet
        # flush: this is the one place the harness deliberately does nothing for
        # a long time, so it is the one place a buffered stdout is indistinguishable
        # from a wedge.
        print(
            f"host is loaded; waiting up to {args.wait_for_quiet / 60:.0f} min for quiet — "
            + "; ".join(violations),
            flush=True,
        )
        while time.time() < deadline:
            time.sleep(min(30, max(5, args.wait_for_quiet / 60)))
            snapshot = host_snapshot()
            violations = host_violations(snapshot, args)
            if not violations:
                print(f"host quiet (loadavg {snapshot['loadavg_1m']}) — starting", flush=True)
                break
            print(
                f"  still loaded: {'; '.join(violations)} ({(deadline - time.time()) / 60:.0f} min left)",
                flush=True,
            )

    valid = not violations
    if violations:
        print("quiet-machine guard tripped: " + "; ".join(violations), file=sys.stderr)
        if not args.allow_loaded:
            print("refusing to benchmark a loaded host (pass --allow-loaded to record INVALID rows)", file=sys.stderr)
            return 2

    device = device_fingerprint()
    run_id = str(uuid.uuid4())[:8]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().isoformat()
    out_path = results_dir / f"{day}-{device['model'].replace(',', '-') or 'host'}-{run_id}.jsonl"
    log_dir = Path(os.environ.get("MLXCAT_BENCH_LOG_DIR", str(results_dir / ".logs")))

    print(f"run {run_id} on {device['chip']} ({device['model']}, {device['os']}) — valid_for_leaderboard={valid}")
    print(f"engines={engine_names} models={models} tiers={chosen_tiers} concurrency={widths}@{conc_tiers}")
    if args.dry_run:
        return 0

    footprint = Footprint()
    harness_commit = git_commit(REPO)
    records_written = 0
    already = recorded_cells(results_dir, device["model"]) if args.resume else set()
    if already:
        print(f"resume: {len(already)} cell(s) already recorded for {device['model']} will be skipped")

    for engine_name in engine_names:
        spec = dict(engines_spec.get(engine_name) or {})
        if not spec and engine_name not in overrides:
            print(f"[{engine_name}] unknown engine (not in engines.json) — skipped", file=sys.stderr)
            continue
        if engine_name in overrides:
            spec["url"] = overrides[engine_name]
            spec.pop("launch", None)
        quarantine = spec.get("quarantined")
        if quarantine and not args.allow_quarantined:
            print(
                f"[{engine_name}] QUARANTINED since {quarantine.get('since')} — skipped.\n"
                f"    {quarantine.get('reason')}\n"
                f"    Re-run it deliberately and alone with {quarantine.get('override')}.",
                file=sys.stderr,
            )
            continue
        if spec.get("platforms") and "macos" not in spec["platforms"]:
            print(f"[{engine_name}] not a macOS engine — skipped (rows come from its own producer)")
            continue
        if spec.get("producer"):
            stamp = {
                "run_id": run_id,
                "platform": "macos",
                "device": device,
                "host": snapshot,
                "valid_for_leaderboard": valid,
                "invalid_reason": "; ".join(violations) or None,
                "harness": {"commit": harness_commit, "tag": args.tag, "argv": sys.argv[1:]},
            }
            records_written += run_producer(engine_name, spec, models, model_specs, tiers, chosen_tiers, model_root, args, stamp, out_path)
            continue
        engine = Engine(engine_name, spec, args, model_root)
        # A sick engine gets a short leash. vllm-mlx was allowed to keep failing
        # for nine minutes on 2026-08-22 before it panicked the GPU driver and
        # rebooted the host, taking three queued passes with it.
        consecutive_failures = 0
        abandon_reason: Optional[str] = None

        for model_id in models:
            if abandon_reason:
                break
            model = model_specs.get(model_id, {"id": model_id})
            if spec.get("launch") and not (model_root / model_id).exists():
                print(f"[{engine_name}/{model_id}] model dir missing under {model_root} — skipped")
                continue
            try:
                engine.launch_for(model, log_dir)
            except EngineUnavailable as error:
                print(f"[{engine_name}] {error} — skipped")
                break
            except Exception as error:  # noqa: BLE001
                print(f"[{engine_name}/{model_id}] launch failed: {error}", file=sys.stderr)
                engine.stop()
                continue
            try:
                offered = engine.resolve_model(model_id)
                if offered is None:
                    print(f"[{engine_name}/{model_id}] model not offered by engine — skipped")
                    continue
                engine.version = engine.version or engine.detect_version()
                engine_block = {
                    "name": engine_name,
                    "version": engine.version,
                    "build_id": engine.build_id,
                    "transport": "http",
                    "url": engine.url,
                    "weights": spec.get("weights", "mlx-community safetensors (same files for every engine)"),
                    "notes": spec.get("notes"),
                    "pid": engine.pid,
                }

                # Calibrate chars/token for this model with a tiny probe.
                try:
                    probe = engine.stream_once(offered, build_prompt(0, 4.0), 8, args.request_timeout)
                    question_tokens = max(probe["prompt_tokens"], 1)
                    cal = engine.stream_once(offered, FILLER * 40 + QUESTION, 8, args.request_timeout)
                    filler_tokens = max(cal["prompt_tokens"] - question_tokens, 1)
                    chars_per_token = (len(FILLER) * 40) / filler_tokens
                except Exception as error:  # noqa: BLE001
                    hint = f" (see {engine.log_path})" if engine.log_path else ""
                    print(f"[{engine_name}/{model_id}] calibration request failed: {error}{hint} — model skipped", file=sys.stderr)
                    continue

                cells = [
                    (tier_name, width, mode)
                    for mode in cache_modes
                    for tier_name, width in (
                        [(t, 1) for t in chosen_tiers]
                        + [(t, w) for t in conc_tiers for w in widths if w > 1]
                    )
                ]
                for tier_name, width, cache_mode in cells:
                    if abandon_reason:
                        break
                    tier = tiers[tier_name]
                    max_tokens = int(tier.get("max_tokens", args.max_tokens))
                    cell_key = (
                        engine_name, engine.build_id or engine.version,
                        model_id, tier_name, width, cache_mode, max_tokens,
                    )
                    if cell_key in already:
                        print(f"[{engine_name}/{model_id}/{tier_name}/c{width}/{cache_mode}] already recorded — skipped (--resume)")
                        continue
                    prompt = build_prompt(int(tier["prompt_tokens"]), chars_per_token)
                    label = f"[{engine_name}/{model_id}/{tier_name}/c{width}/{cache_mode}]"
                    # Re-sample the host per cell — a run that starts quiet can get
                    # loud, and later rows must not inherit the opening verdict.
                    cell_snapshot = host_snapshot()
                    cell_violations = host_violations(cell_snapshot, args)
                    cell_valid = not cell_violations
                    try:
                        metrics = run_cell(
                            engine, offered, model, tier, width, prompt, args,
                            lambda: FootprintSampler(footprint, engine.pid),
                            cache_mode=cache_mode,
                        )
                    except Exception as error:  # noqa: BLE001
                        print(f"{label} FAILED: {error}", file=sys.stderr)
                        metrics = {"error": str(error)}

                    # The runaway guard has been watching this pid since launch.
                    # A breach outranks whatever the cell reported: the engine is
                    # dead, the row is not evidence, and we do not relaunch it.
                    breach = engine.guard.breach if engine.guard else None
                    if breach:
                        metrics = {"error": f"runaway guard: {breach}"}
                        abandon_reason = breach
                    if "error" in metrics:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0
                    if (
                        abandon_reason is None
                        and args.engine_failure_budget > 0
                        and consecutive_failures >= args.engine_failure_budget
                    ):
                        abandon_reason = f"{consecutive_failures} consecutive failed cells"

                    record = {
                        "schema": SCHEMA,
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "platform": "macos",
                        "device": device,
                        "engine": engine_block,
                        "model": {"id": model_id, "offered_as": offered, **{k: v for k, v in model.items() if k != "id"}},
                        "workload": {
                            "context_tier": tier_name,
                            "prompt_tokens_target": int(tier["prompt_tokens"]),
                            "max_tokens": int(tier.get("max_tokens", args.max_tokens)),
                            "concurrency": width,
                            "cache_mode": cache_mode,
                            "arrival": "burst" if not args.request_rate else f"poisson:{args.request_rate}/{args.burstiness}",
                            "temperature": 0,
                            "runs": args.runs,
                            "warmup": args.warmup,
                        },
                        "metrics": metrics,
                        "host": cell_snapshot,
                        "valid_for_leaderboard": cell_valid and "error" not in metrics,
                        "invalid_reason": "; ".join(cell_violations) or (metrics.get("error") if "error" in metrics else None),
                        "harness": {"commit": harness_commit, "tag": args.tag, "argv": sys.argv[1:]},
                    }
                    with open(out_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    records_written += 1
                    if "error" not in metrics:
                        dec = metrics["decode_tps"]["median"] if metrics["decode_tps"] else float("nan")
                        pre = metrics["prefill_tps"]["median"] if metrics["prefill_tps"] else float("nan")
                        peak = (metrics["peak_phys_footprint_bytes"] or 0) / 2**30
                        agg = f" agg {metrics['aggregate_tps']['median']:.1f} tok/s" if width > 1 else ""
                        print(f"{label} prompt {metrics['prompt_tokens']} tok · TTFT {metrics['ttft_ms']['median']:.0f} ms · prefill {pre:.0f} tok/s · decode {dec:.1f} tok/s · peak {peak:.2f} GiB{agg}")
            finally:
                engine.stop()

        sync_results(args, out_path, engine_name)

        if abandon_reason:
            print(
                f"[{engine_name}] ABANDONED: {abandon_reason}. Remaining cells for this engine were "
                f"not attempted; the run continues with the next engine. If this is a host-stability "
                f"failure rather than a bad build, quarantine it in {args.engines_file}.",
                file=sys.stderr,
            )

    print(f"wrote {records_written} record(s) → {out_path}")
    if records_written:
        print("render: python3 bench/leaderboard.py")
    return 0 if records_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
