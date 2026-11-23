# Inkling 2 Mini Python reference fixture

The fixture is for numerical parity with mlxcat's native Swift engine. The Python environment lives in the fixture workspace in `venv/`. The JSON records the measurement host hardware class and OS version. The pinned inference environment uses CPython 3.12.12, MLX 0.32.4, Transformers 5.22.1, PyTorch 2.14.1, and mlx-audio 0.6.7.

## Source and checkpoints

- mlx-vlm introduction commit `111ac668078e017c9db1eb3936f4d7a6e0a88f89` ([PR #2778](https://github.com/Blaizzy/mlx-vlm/pull/2778)). Git records the commit at 23:17:58 CET on 20 November, which is 21 November in Singapore. Tool parsing landed in a subsequent commit, so the fixture preserves special tokens and raw typed frames.
- netprojects checkpoint `0b437271c7982a8b7acdca011170dcc86b671b20` and baytizzel checkpoint `9e7ea8ebec04b1e1598f3460474cef11de5433f4`. Each carries four target shards and optional DSpark weights. The fixture uses standard autoregressive decode.
- The baytizzel checkpoint predates upstream's final parameter names. The harness aliases 144 per-layer loop-controller keys and one message-query key in memory at load. The mapping yields the same 1,271 keys as netprojects, and the upstream loader still checks shapes and completeness. Parameters retain their stored values and dtypes.

## Reproduce

The capture script, model-card clones, image and JSON live in the shared scratch folder. Set `INKLING_WORKSPACE` to the fixture workspace, `INKLING_MODEL_ROOT` to the directory containing both model owners, and `INKLING_LEASE_SCRIPT` to the machine lease script. Run the measured commands from the mlxcat worktree.

```sh
uv venv --python 3.12 "$INKLING_WORKSPACE/venv"  # if absent
git -C "$INKLING_WORKSPACE/refs/mlx-vlm-reference" rev-parse HEAD
uv pip install --python "$INKLING_WORKSPACE/venv/bin/python" --no-deps \
  -r "$INKLING_WORKSPACE/reference/dependencies.txt" "$INKLING_WORKSPACE/refs/mlx-vlm-reference"

models="$INKLING_MODEL_ROOT"
/bin/zsh "$INKLING_WORKSPACE/reference/bench.zsh" \
  "$models/netprojects/Inkling-2-Mini-MLX-4-8bit" \
  "$INKLING_WORKSPACE/refs/netprojects__Inkling-2-Mini-MLX-4-8bit"
/bin/zsh "$INKLING_WORKSPACE/reference/bench.zsh" \
  "$models/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb" \
  "$INKLING_WORKSPACE/refs/baytizzel__Inkling-2-Mini-MLX-M5Pro-48gb"
```

For each model, the harness verifies the index tensor coverage and payload total, the file lengths against safetensors headers and the cloned card's LFS pointers, and the SHA-256 hashes of the shards, draft and input assets. It also checks the supplied `SHA256SUMS` where available. The checkpoint template renders the prompt. For the image case, mlx-vlm's processor expands placeholders and feeds a deterministic 224×224 RGB image with a red square and a blue circle.

Each case starts from a fresh cache and uses seed 42, temperature zero, and 1,024-token prefill chunks. The first pass captures timings and MLX allocator peak after a two-token model warmup. TTFT starts at entry to the generation loop and includes embedding, model prefill and first-token evaluation. Template rendering, tokenization, CPU image preparation, file verification and weight load are outside that timer. Decode throughput counts the tokens after the first from first to final yielded token. Peak memory is MLX's process allocator peak in bytes for the timing pass, including live model allocations.

The second pass captures the ten highest raw logits for each of the first eight generated steps, before log-softmax and through EOS if it comes earlier. The observer returns the original logits untouched. The harness checks that the instrumented and timed runs emitted identical IDs before writing JSON. Output includes the original messages, expanded input IDs, generated IDs through EOS or the prompt-specific cap of 64 tokens for ordinary cases and 256 for tool and reasoning cases, and decoded text with special tokens preserved. The benchmark wrapper acquires the machine-wide exclusive lease and records macOS memory pressure and VM statistics in the run log. Each build is measured independently, with one model loaded at a time. Run `venv/bin/python reference/compare.py` after both captures to validate and compare the outputs.

The prompt set covers a short greeting, factual QA, Japanese translation, Swift, JSON, punctuation, a system instruction, a multi-turn recall, reasoning, a tool invocation, roughly 8K tokens of retrieval context, and image input. For diagnosing a divergence, feed the same prefix and compare the highest raw logits at the first differing token. Subsequent greedy steps are conditioned on different token prefixes.

## Recorded results

The two builds matched exactly on all 12 prompts for tokenized input, generated IDs and the captured top-10 logits. The tool case emitted a complete `get_weather` call for Lisbon, the reasoning case ended with 20, and the image case identified the red square and blue circle. The long-context case used 8,225 input IDs and reached its 64-token cap during reasoning.

| Build | Median decode outside the 8K case | 8K TTFT | 8K decode | 8K peak MLX memory |
| --- | ---: | ---: | ---: | ---: |
| netprojects | 23.09 tok/s | 22.65 s | 18.92 tok/s | 24.80 GiB |
| baytizzel | 23.19 tok/s | 22.89 s | 19.04 tok/s | 24.80 GiB |

The per-case comparison is in `reference/RESULTS.md` in the fixture workspace. Timing columns are single runs per case.

See [INKLING-REFERENCES.md](INKLING-REFERENCES.md) for the source audit and adoption decisions.
