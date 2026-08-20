#!/usr/bin/env python3
"""HTTP parity benchmark for MLXCat native and oMLX.

This intentionally measures both servers through the same OpenAI-compatible
/v1/chat/completions client. It is not a pass/fail test.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


PREFIX_SENTENCE = (
    "The capital of France is Paris. Swift concurrency protects shared state. "
    "GPU kernels execute matrix operations quickly. "
)
PREFIX_REPEAT = 26
SUFFIXES = [
    " The answer is",
    " Therefore",
    " In summary",
    " A careful implementation",
    " The benchmark result",
    " Swift concurrency",
    " The next step",
    " Prefix caching",
]


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def wait_ready(base_url: str, timeout_s: float = 90) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"server did not become ready: {base_url}")


def stream_chat(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_content = None
    usage: dict[str, Any] = {}
    content_chunks = 0
    text_parts: list[str] = []

    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
                continue
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_content is None:
                        first_content = time.perf_counter()
                    content_chunks += 1
                    text_parts.append(content)

    end = time.perf_counter()
    if first_content is None:
        first_content = end

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or max_tokens)
    ttft_s = first_content - start
    wall_s = end - start
    generation_wall_s = max(wall_s - ttft_s, 1e-9)
    generation_duration = float(usage.get("generation_duration") or 0)
    server_generation_tps = usage.get("generation_tokens_per_second")
    if generation_duration <= 0 or content_chunks < min(completion_tokens, 2):
        server_generation_tps = None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_s * 1000,
        "wall_ms": wall_s * 1000,
        "pp_tps": prompt_tokens / max(ttft_s, 1e-9),
        "tg_tps": completion_tokens / generation_wall_s,
        "http_output_tps": completion_tokens / max(wall_s, 1e-9),
        "server_pp_tps": usage.get("prompt_tokens_per_second"),
        "server_tg_tps": server_generation_tps,
        "generation_duration_s": generation_duration,
        "content_chunks": content_chunks,
        "text": "".join(text_parts),
        "usage": usage,
    }


def benchmark_server(
    name: str,
    base_url: str,
    model: str,
    runs: int,
    warmup: int,
    decode_tokens: int,
) -> dict[str, Any]:
    wait_ready(base_url)
    prefix = PREFIX_SENTENCE * PREFIX_REPEAT
    short_prompt = "The capital of France is"

    # Load/warm outside measured samples.
    stream_chat(base_url, model, short_prompt, 1)

    def measured_prompt() -> dict[str, Any]:
        return stream_chat(base_url, model, prefix, decode_tokens)

    for _ in range(warmup):
        measured_prompt()
    prompt_runs = [measured_prompt() for _ in range(runs)]

    def measured_ttft() -> dict[str, Any]:
        return stream_chat(base_url, model, short_prompt, 1)

    for _ in range(warmup):
        measured_ttft()
    ttft_runs = [measured_ttft() for _ in range(runs)]

    concurrency_results = []
    for concurrency in [1, 2, 4, 8]:
        prompts = [prefix + SUFFIXES[index % len(SUFFIXES)] for index in range(concurrency)]

        def run_batch() -> dict[str, Any]:
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                outputs = list(
                    pool.map(
                        lambda prompt: stream_chat(base_url, model, prompt, decode_tokens),
                        prompts,
                    )
                )
            wall_s = time.perf_counter() - start
            generated = sum(int(output["completion_tokens"]) for output in outputs)
            return {
                "wall_s": wall_s,
                "generated_tokens": generated,
                "throughput_tps": generated / max(wall_s, 1e-9),
            }

        for _ in range(warmup):
            run_batch()
        batch_runs = [run_batch() for _ in range(runs)]
        concurrency_results.append(
            {
                "concurrency": concurrency,
                "generated_tokens": int(batch_runs[0]["generated_tokens"]),
                "median_s": median([float(run["wall_s"]) for run in batch_runs]),
                "throughput_tps": median(
                    [float(run["throughput_tps"]) for run in batch_runs]
                ),
            }
        )

    return {
        "name": name,
        "base_url": base_url,
        "model": model,
        "runs": runs,
        "warmup": warmup,
        "decode_tokens": decode_tokens,
        "prompt_description": (
            f"prefix sentence repeated {PREFIX_REPEAT}x; fixed suffix literals"
        ),
        "prompt_tokens": int(median([float(run["prompt_tokens"]) for run in prompt_runs])),
        "client_pp_tps": median([float(run["pp_tps"]) for run in prompt_runs]),
        "client_tg_tps": median([float(run["tg_tps"]) for run in prompt_runs]),
        "http_output_tps": median([float(run["http_output_tps"]) for run in prompt_runs]),
        "server_pp_tps": median(
            [
                float(run["server_pp_tps"])
                for run in prompt_runs
                if run["server_pp_tps"] is not None
            ]
        ),
        "server_tg_tps": maybe_median(
            [
                float(run["server_tg_tps"])
                for run in prompt_runs
                if run["server_tg_tps"] is not None
            ]
        ),
        "ttft_ms": median([float(run["ttft_ms"]) for run in ttft_runs]),
        "wall_ms": median([float(run["wall_ms"]) for run in prompt_runs]),
        "generation_duration_s": median(
            [float(run["generation_duration_s"]) for run in prompt_runs]
        ),
        "content_chunks": int(median([float(run["content_chunks"]) for run in prompt_runs])),
        "concurrency": concurrency_results,
    }


def markdown(results: list[dict[str, Any]]) -> str:
    lines = ["# HTTP Parity Benchmark", ""]
    lines.append(
        "| Server | Server PP | Server TG | HTTP TTFT | HTTP output | Prompt Tokens | Content Chunks |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        server_tg = result["server_tg_tps"]
        server_tg_text = "n/a" if server_tg is None else f"{server_tg:.2f} tok/s"
        lines.append(
            f"| {result['name']} | {result['server_pp_tps']:.2f} tok/s | "
            f"{server_tg_text} | {result['ttft_ms']:.2f} ms | "
            f"{result['http_output_tps']:.2f} tok/s | "
            f"{result['prompt_tokens']} | {result['content_chunks']} |"
        )
    lines.append("")
    lines.append(
        "Server TG is `n/a` when the server emits one aggregated content chunk or reports zero generation duration."
    )
    lines.append("")
    lines.append("| Server | C=1 | C=2 | C=4 | C=8 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for result in results:
        values = {entry["concurrency"]: entry["throughput_tps"] for entry in result["concurrency"]}
        lines.append(
            f"| {result['name']} | {values[1]:.2f} | {values[2]:.2f} | "
            f"{values[4]:.2f} | {values[8]:.2f} |"
        )
    return "\n".join(lines)


def maybe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-url", required=True)
    parser.add_argument("--omlx-url", required=True)
    parser.add_argument("--model", default="Qwen3-0.6B-4bit")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--output-json", default="/tmp/mlxcat-http-parity.json")
    parser.add_argument("--output-md", default="/tmp/mlxcat-http-parity.md")
    args = parser.parse_args()

    results = [
        benchmark_server(
            "native-http",
            args.native_url,
            args.model,
            args.runs,
            args.warmup,
            args.decode_tokens,
        ),
        benchmark_server(
            "omlx-http",
            args.omlx_url,
            args.model,
            args.runs,
            args.warmup,
            args.decode_tokens,
        ),
    ]

    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = markdown(results)
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
