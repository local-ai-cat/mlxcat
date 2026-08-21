# Contributing to mlxcat

Thanks for looking. mlxcat is a native-Swift MLX serving engine that ships on
macOS **and** embedded in an iOS app, so every change is judged on both.

## Ground rules

* **Swift + MLX only at runtime.** Python projects (oMLX, mlx-lm, vllm-mlx, …)
  are references we port from and benchmark against — never dependencies.
* **Invariant gates, not vibes.** Batched decode must match serial decode within
  dtype tolerance; a prefix-cache hit must match a fresh prefill; a memory
  budget must hold at 16k. New engine behaviour comes with a test that would
  have caught the bug.
* **Measure on a quiet machine.** Performance claims come from `bench/run.py`
  rows (see `bench/README.md`), never from a single `swift test` print.
* **Both platforms.** If a change can affect the iOS embed (memory, threading,
  Metal stream use), say so in the PR and how you checked.
* Two strings never change without a migration: the prefix-cache hash seed
  (`mlxserve-prefix-cache`) and the wire value `"owned_by": "mlxserve-native"`.

## Build

```bash
swift build                                   # library + executables, debug
swift build -c release --product mlxcat-http  # the server binary used by bench/
swift run mlxcat-http --model-dir ~/Library/Caches/models/mlx-community --port 11500
```

Models are never downloaded by the build or the tests. Point at local
`mlx-community` directories.

## Tests

```bash
swift test                                    # unit + fixture tests; model gates SKIP
MLXSERVE_TEST_MODEL=~/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit swift test
scripts/nightly-models.sh                     # every model-gated suite, wired to local models
```

Env gates (all optional; a suite skips loudly when its gate is unset):

| variable | suites |
|---|---|
| `MLXSERVE_TEST_MODEL` | batch invariance, scheduler, prefix cache, width-1, engine |
| `MLXSERVE_HYBRID_TEST_MODEL` | Qwen3.5/3.8 hybrid-cache batch integration |
| `MLXSERVE_SLIDING_TEST_MODEL` | sliding-window (Gemma) batch integration |
| `MLXSERVE_MOE_TEST_MODEL` | MoE batch integration |
| `MLXSERVE_VLM_TEST_MODEL` | VLM batch integration |
| `MLXSERVE_RERANK_TEST_MODEL` | rerank |
| `MLXCAT_MEMORY_BUDGET_MODEL` / `_BYTES` / `_PROMPT_TOKENS` | 16k memory budget regression |
| `MLXSERVE_WHISPERKIT_LIVE` | WhisperKit live transcription |

**A test that touches MLX state must guard on Metal.** Hosted CI runners have no
loaded metallib, and MLX's memory APIs (`Memory.cacheLimit`, `Memory.memoryLimit`,
anything that allocates) *abort the process* there rather than failing a test — one
unguarded `setUp` turns the whole run red with `Failed to load the default metallib`.
Call `try MLXMetalRuntime.requireAvailable()` first, and prefer splitting pure policy
out of the MLX call so the logic stays testable everywhere (see `MemoryGuard`'s
`plannedAllocatorLimits` vs `applyAllocatorLimits`).

CI (`.github/workflows/ci.yml`) runs the no-model suite on GitHub-hosted Apple
Silicon and checks that `LEADERBOARD.md` is derived from `bench/results/`.
The model-gated suites and the benchmark run on our own Macs via
`scripts/nightly-models.sh` and `bench/run.py` — a public repo cannot use our
self-hosted runners, so those results arrive as commits, not as checks.

## Benchmarks and the leaderboard

```bash
python3 bench/run.py --engines mlxcat,omlx --model-set default
python3 bench/leaderboard.py
```

Commit the new `bench/results/*.jsonl` together with the regenerated
`LEADERBOARD.md`. The leaderboard is stratified by platform and device; iOS rows
come from the in-app harness, never from `run.py`. Adding an engine = one block
in `bench/engines.json`. Read `docs/ENGINES.md` for who is on the board and why.

## Dependencies and drift

`Package.swift` pins `mlx-swift-lm` by **revision** (loader fixes land on `main`
long before a tag), `mlx-swift` by minor, `WhisperKit` and `swift-transformers`
by revision. `scripts/donor-drift.sh` reports how far upstream has moved and
what changed; the weekly `donor-drift` workflow files it as an issue here.
Bumping a pin is a PR of its own: bump, run the model-gated suites, run the
`default` bench set, and include both in the PR.

The reference clones (`guest/`) are gitignored; keep them there so the shelf
travels with the checkout.

## Commits and PRs

* Conventional commits (`feat(scheduler): …`, `fix(native): …`, `perf(generator): …`,
  `bench: …`, `docs: …`).
* One logical change per commit; explain the *why* in the body when it is not
  obvious — the history is read by people porting from it.
* PRs fill the template: what changed, how it was verified (suite, model gates,
  bench rows, device), and platform impact.
* `main` is the canonical branch; releases are tags. There is no long-lived
  integration branch.

## Code of conduct

Be kind, be specific, assume good faith. Harassment of any kind is not tolerated;
maintainers will remove contributors who do not meet that bar.
