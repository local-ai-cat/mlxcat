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
import concurrent.futures
import ctypes
import ctypes.util
import datetime as dt
import json
import os
import platform
import re
import shutil
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
        "thermal_cpu_speed_limit": thermal_cpu_speed_limit(),
    }


def quiet_machine_violations(snapshot: Dict[str, Any], max_load: float, min_free_pct: float) -> List[str]:
    problems: List[str] = []
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
        path = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
        self._lib = ctypes.CDLL(path)
        self._lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self._lib.proc_pid_rusage.restype = ctypes.c_int

    def read(self, pid: int) -> Optional[Dict[str, int]]:
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

    # -- lifecycle -------------------------------------------------------- #

    def launch_for(self, model: Dict[str, Any], log_dir: Path) -> None:
        """Start one engine process for one model (so peak memory is per model)."""
        launch = self.spec.get("launch")
        if not launch:
            if not self.url:
                raise RuntimeError(f"{self.name}: neither 'launch' nor 'url' configured")
            self.pid = pid_listening_on(self._port_from_url(self.url))
            self.wait_ready()
            return
        binary = self._resolve_binary(launch["bin"])
        if binary is None:
            raise EngineUnavailable(f"{self.name}: binary not found ({launch['bin']}); set {launch.get('bin_env', 'the path')}")
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
        self.wait_ready(timeout=float(launch.get("ready_timeout_s", 180)))

    def stop(self) -> None:
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
            return os.environ[env_name]
        candidate = spec.format(repo=str(REPO))
        if os.path.sep in candidate:
            return candidate if os.path.exists(candidate) else None
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

    def stream_once(self, model: str, prompt: str, max_tokens: int, timeout: float) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body.update(self.spec.get("extra_request_fields") or {"enable_thinking": False})
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
        decode_seconds = max(finished - first_output, 1e-9)
        ttft = first_output - started
        gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
        return {
            "ttft_ms": ttft * 1000,
            "wall_ms": (finished - started) * 1000,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prefill_tps": (prompt_tokens / ttft) if prompt_tokens and ttft > 0 else None,
            "decode_tps": (completion_tokens - 1) / decode_seconds if completion_tokens > 1 else None,
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


def run_cell(
    engine: Engine,
    offered: str,
    model: Dict[str, Any],
    tier: Dict[str, Any],
    concurrency: int,
    prompt: str,
    args: argparse.Namespace,
    sampler_factory,
) -> Dict[str, Any]:
    max_tokens = int(tier.get("max_tokens", args.max_tokens))
    timeout = float(args.request_timeout)
    # warm: first call is discarded (cold shapes/compilation), then `warmup` more.
    cold = engine.stream_once(offered, prompt, max_tokens, timeout)
    for _ in range(args.warmup):
        engine.stream_once(offered, prompt, max_tokens, timeout)

    runs: List[Dict[str, Any]] = []
    aggregate_samples: List[float] = []
    with sampler_factory() as sampler:
        for _ in range(args.runs):
            if concurrency == 1:
                runs.append(engine.stream_once(offered, prompt, max_tokens, timeout))
            else:
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows = list(pool.map(lambda _: engine.stream_once(offered, prompt, max_tokens, timeout), range(concurrency)))
                elapsed = time.perf_counter() - started
                total = sum(int(r["completion_tokens"]) for r in rows)
                aggregate_samples.append(total / max(elapsed, 1e-9))
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
    }
    if concurrency > 1:
        metrics["aggregate_tps"] = spread(aggregate_samples)
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
        }
        argv = [binary] + [a.format(**variables) for a in producer["args"]]
        flag = producer.get("prompt_tokens_flag", "--prompt-tokens")
        for tier_name in chosen_tiers:
            argv += [flag, str(int(tiers[tier_name]["prompt_tokens"]))]
        print(f"[{engine_name}/{model_id}] {' '.join(os.path.basename(a) if i == 0 else a for i, a in enumerate(argv))}")
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=float(args.request_timeout) * 4)
        except subprocess.TimeoutExpired:
            print(f"[{engine_name}/{model_id}] producer timed out", file=sys.stderr)
            continue
        if completed.returncode != 0:
            print(f"[{engine_name}/{model_id}] producer rc={completed.returncode}: {completed.stderr.strip()[-400:]}", file=sys.stderr)
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            target = int(record.get("workload", {}).get("prompt_tokens_target", 0))
            record["workload"]["context_tier"] = target_to_tier.get(target, record["workload"].get("context_tier"))
            record["model"] = {**record.get("model", {}), **{k: v for k, v in model_specs.get(model_id, {}).items() if k != "id"}, "id": model_id}
            record["engine"]["name"] = engine_name
            record.update(stamp)
            record["valid_for_leaderboard"] = bool(stamp.get("valid_for_leaderboard")) and "error" not in record.get("metrics", {})
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
    parser.add_argument("--concurrency-tier", default="", help="context tier used for the concurrency leg (default: matrix.json)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--model-root", default=os.environ.get("MLXCAT_BENCH_MODEL_ROOT", str(Path.home() / "Library/Caches/models/mlx-community")))
    parser.add_argument("--memory-ceiling-bytes", type=int, default=0, help="passed to launchers that support a ceiling (0 = engine default)")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--engines-file", default=str(DEFAULT_ENGINES))
    parser.add_argument("--matrix-file", default=str(DEFAULT_MATRIX))
    parser.add_argument("--max-load", type=float, default=8.0, help="quiet-machine guard: 1-minute load average ceiling")
    parser.add_argument("--min-free-pct", type=float, default=35.0, help="quiet-machine guard: memory_pressure free %% floor")
    parser.add_argument("--allow-loaded", action="store_true", help="run even if the guard trips; rows are marked invalid for the leaderboard")
    parser.add_argument("--tag", default="", help="free-text tag stored on every record (e.g. 'pin a481734 + ceiling')")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)

    engines_spec = load_json(Path(args.engines_file))["engines"]
    matrix = load_json(Path(args.matrix_file))
    model_root = Path(args.model_root).expanduser()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or matrix["model_sets"][args.model_set]
    model_specs = {m["id"]: m for m in matrix["models"]}
    tiers = {t["name"]: t for t in matrix["context_tiers"]}
    chosen_tiers = [t.strip() for t in args.contexts.split(",") if t.strip()] or matrix["default_context_tiers"]
    widths = [int(w) for w in args.concurrency.split(",") if w.strip()] or matrix["concurrency"]["widths"]
    conc_tier = args.concurrency_tier or matrix["concurrency"]["tier"]

    overrides = {}
    for item in args.engine_url:
        name, _, url = item.partition("=")
        overrides[name] = url
    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    for name in overrides:
        if name not in engine_names:
            engine_names.append(name)

    snapshot = host_snapshot()
    violations = quiet_machine_violations(snapshot, args.max_load, args.min_free_pct)
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
    print(f"engines={engine_names} models={models} tiers={chosen_tiers} concurrency={widths}@{conc_tier}")
    if args.dry_run:
        return 0

    footprint = Footprint()
    harness_commit = git_commit(REPO)
    records_written = 0

    for engine_name in engine_names:
        spec = dict(engines_spec.get(engine_name) or {})
        if not spec and engine_name not in overrides:
            print(f"[{engine_name}] unknown engine (not in engines.json) — skipped", file=sys.stderr)
            continue
        if engine_name in overrides:
            spec["url"] = overrides[engine_name]
            spec.pop("launch", None)
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

        for model_id in models:
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
                    "transport": "http",
                    "url": engine.url,
                    "weights": spec.get("weights", "mlx-community safetensors (same files for every engine)"),
                    "notes": spec.get("notes"),
                    "pid": engine.pid,
                }

                # Calibrate chars/token for this model with a tiny probe.
                probe = engine.stream_once(offered, build_prompt(0, 4.0), 8, args.request_timeout)
                question_tokens = max(probe["prompt_tokens"], 1)
                cal = engine.stream_once(offered, FILLER * 40 + QUESTION, 8, args.request_timeout)
                filler_tokens = max(cal["prompt_tokens"] - question_tokens, 1)
                chars_per_token = (len(FILLER) * 40) / filler_tokens

                cells = [(tier_name, 1) for tier_name in chosen_tiers] + [(conc_tier, w) for w in widths if w > 1]
                for tier_name, width in cells:
                    tier = tiers[tier_name]
                    prompt = build_prompt(int(tier["prompt_tokens"]), chars_per_token)
                    label = f"[{engine_name}/{model_id}/{tier_name}/c{width}]"
                    try:
                        metrics = run_cell(
                            engine, offered, model, tier, width, prompt, args,
                            lambda: FootprintSampler(footprint, engine.pid),
                        )
                    except Exception as error:  # noqa: BLE001
                        print(f"{label} FAILED: {error}", file=sys.stderr)
                        metrics = {"error": str(error)}
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
                            "temperature": 0,
                            "runs": args.runs,
                            "warmup": args.warmup,
                        },
                        "metrics": metrics,
                        "host": snapshot,
                        "valid_for_leaderboard": valid and "error" not in metrics,
                        "invalid_reason": "; ".join(violations) or (metrics.get("error") if "error" in metrics else None),
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

    print(f"wrote {records_written} record(s) → {out_path}")
    if records_written:
        print("render: python3 bench/leaderboard.py")
    return 0 if records_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
