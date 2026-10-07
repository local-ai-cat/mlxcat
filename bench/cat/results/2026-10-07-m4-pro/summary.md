# Measured summary

Medians and ranges are shown with sample counts. Cache states come only from server-reported cached-token evidence; request roles are recorded separately. No p95 is inferred from three exact replays.

| Model | Case | State | n | Pass | Prompt tok | Cached tok median | TTFT median s (range) | Prefill tok/s median | Decode tok/s median | Total median s | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct-4bit | context_2k | server_reported_cached | 1 | 1/1 | 1934 | 30 | 1.914 (1.914–1.914) | 1010.7 | 84.7 | 2.032 | 1.87 |
| Llama-3.2-3B-Instruct-4bit | context_2k | unverified_no_reported_cache | 3 | 3/3 | 1934 | 0 | 1.940 (1.937–1.941) | 996.8 | 84.1 | 2.059 | 1.87 |
| Llama-3.2-3B-Instruct-4bit | context_8k | server_reported_cached | 1 | 1/1 | 7559 | 43 | 8.751 (8.751–8.751) | 863.8 | 61.9 | 8.913 | 1.87 |
| Llama-3.2-3B-Instruct-4bit | context_8k | unverified_no_reported_cache | 3 | 3/3 | 7559 | 0 | 8.794 (8.782–8.805) | 859.6 | 60.0 | 8.960 | 1.87 |
| Llama-3.2-3B-Instruct-4bit | grounded_extract | server_reported_cached | 1 | 1/1 | 78 | 30 | 0.076 (0.076–0.076) | 1027.2 | 93.0 | 0.377 | 1.87 |
| Llama-3.2-3B-Instruct-4bit | grounded_extract | unverified_no_reported_cache | 3 | 3/3 | 78 | 0 | 0.105 (0.105–0.107) | 741.3 | 92.5 | 0.410 | 1.87 |
| Llama-3.2-3B-Instruct-4bit | short_chat | unverified_no_reported_cache | 4 | 4/4 | 44 | 0 | 0.077 (0.077–0.686) | 571.7 | 125.3 | 0.101 | 1.89 |
| Qwen2.5-Coder-7B-Instruct-4bit | context_2k | server_reported_cached | 1 | 1/1 | 2430 | 24 | 5.409 (5.409–5.409) | 449.3 | 46.4 | 5.667 | 4.12 |
| Qwen2.5-Coder-7B-Instruct-4bit | context_2k | unverified_no_reported_cache | 3 | 3/3 | 2430 | 0 | 5.387 (5.387–5.390) | 451.1 | 45.2 | 5.652 | 4.12 |
| Qwen2.5-Coder-7B-Instruct-4bit | context_8k | server_reported_cached | 1 | 1/1 | 9555 | 38 | 23.469 (23.469–23.469) | 407.1 | 39.9 | 23.770 | 4.12 |
| Qwen2.5-Coder-7B-Instruct-4bit | context_8k | unverified_no_reported_cache | 3 | 3/3 | 9555 | 0 | 23.436 (23.436–23.437) | 407.7 | 39.9 | 23.736 | 4.12 |
| Qwen2.5-Coder-7B-Instruct-4bit | grounded_extract | server_reported_cached | 1 | 1/1 | 74 | 24 | 0.165 (0.165–0.165) | 449.2 | 47.0 | 1.015 | 4.12 |
| Qwen2.5-Coder-7B-Instruct-4bit | grounded_extract | unverified_no_reported_cache | 3 | 3/3 | 74 | 0 | 0.228 (0.228–0.230) | 324.0 | 46.8 | 1.083 | 4.12 |
| Qwen2.5-Coder-7B-Instruct-4bit | short_chat | unverified_no_reported_cache | 4 | 4/4 | 38 | 0 | 0.163 (0.162–0.883) | 233.8 | 67.3 | 0.207 | 4.12 |
| Qwen3-0.6B-4bit | context_2k | server_reported_cached | 1 | 0/1 | 2413 | 3 | 0.537 (0.537–0.537) | 4489.8 | 182.8 | 0.576 | 0.46 |
| Qwen3-0.6B-4bit | context_2k | unverified_no_reported_cache | 3 | 0/3 | 2413 | 0 | 0.532 (0.531–0.535) | 4531.7 | 178.7 | 0.573 | 0.46 |
| Qwen3-0.6B-4bit | context_8k | server_reported_cached | 1 | 0/1 | 9538 | 17 | 3.528 (3.528–3.528) | 2703.8 | 75.9 | 3.620 | 0.46 |
| Qwen3-0.6B-4bit | context_8k | unverified_no_reported_cache | 3 | 0/3 | 9538 | 0 | 3.503 (3.502–3.511) | 2722.8 | 78.3 | 3.592 | 0.46 |
| Qwen3-0.6B-4bit | grounded_extract | server_reported_cached | 1 | 1/1 | 57 | 3 | 0.020 (0.020–0.020) | 2821.3 | 269.8 | 0.169 | 0.45 |
| Qwen3-0.6B-4bit | grounded_extract | unverified_no_reported_cache | 3 | 3/3 | 57 | 0 | 0.020 (0.020–0.020) | 2871.7 | 264.5 | 0.171 | 0.46 |
| Qwen3-0.6B-4bit | short_chat | unverified_no_reported_cache | 4 | 4/4 | 21 | 0 | 0.015 (0.015–0.404) | 1380.7 | 333.4 | 0.024 | 0.46 |
| Qwen3.5-4B-MLX-4bit | context_2k | server_reported_cached | 3 | 3/3 | 2538 | 2048 | 0.763 (0.762–0.763) | 3327.5 | 64.6 | 0.948 | 2.65 |
| Qwen3.5-4B-MLX-4bit | context_2k | unverified_no_reported_cache | 1 | 1/1 | 2538 | 0 | 3.638 (3.638–3.638) | 697.6 | 65.4 | 3.822 | 2.65 |
| Qwen3.5-4B-MLX-4bit | context_8k | server_reported_cached | 3 | 3/3 | 10038 | 9728 | 0.582 (0.580–0.584) | 17249.2 | 58.7 | 0.785 | 2.65 |
| Qwen3.5-4B-MLX-4bit | context_8k | unverified_no_reported_cache | 1 | 1/1 | 10038 | 0 | 15.131 (15.131–15.131) | 663.4 | 59.1 | 15.334 | 2.65 |
| Qwen3.5-4B-MLX-4bit | grounded_extract | unverified_no_reported_cache | 4 | 4/4 | 56 | 0 | 0.110 (0.109–0.110) | 509.7 | 64.2 | 0.578 | 2.65 |
| Qwen3.5-4B-MLX-4bit | short_chat | unverified_no_reported_cache | 4 | 4/4 | 21 | 0 | 0.067 (0.067–0.885) | 311.2 | 87.3 | 0.102 | 3.08 |

## Coding

Pass rate: **0/4 (0%)**.

| Task | Result | Time s | Detail |
|---|---:|---:|---|
| function_completion | FAIL | 3.036 | BLOCKED: strict evaluator sandbox unavailable (signal 6); failed closed |
| bug_fix | FAIL | 3.340 | BLOCKED: strict evaluator sandbox unavailable (signal 6); failed closed |
| simple_refactor | FAIL | 1.227 | BLOCKED: strict evaluator sandbox unavailable (signal 6); failed closed |
| json_tool_call | FAIL | 0.980 | invalid strict JSON: Expecting value: line 1 column 1 (char 0) |
