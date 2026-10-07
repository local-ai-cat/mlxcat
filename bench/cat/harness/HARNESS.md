# Reproduction harness

This task uses the existing `mlxcat-http` source at revision
`58bed882d417b6419e8c8c90d77e731e6e5cfd89` and installed model directories.
No model or dependency download is permitted.

Build into task-owned scratch after confirming a quiet machine:

```zsh
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  swift build \
  --package-path /Users/timapple/src/forest/local-ai-cat/mlxcat \
  --scratch-path /Users/timapple/src/task-data/cat-m4-inference-20261002/build \
  --skip-update \
  -c release \
  --product mlxcat-http
```

Run the fixed suite:

```zsh
python3 benchmark.py \
  --binary build/arm64-apple-macosx/release/mlxcat-http \
  --models-root /Users/timapple/Library/Caches/models/mlx-community \
  --output run-1
```

The server runs under a macOS sandbox that denies networking except loopback and
receives a fixed minimal environment rather than the coordinator environment.
One model/request is active at a time. Temperature is 0, seed is 42, thinking is
disabled, and every completed or failed request gets a durable receipt. A
continuous watchdog terminates only the exact task-owned server PID on a safety
violation.

`request_role` distinguishes the first process request, first case request, and
exact replays. `cache_state` uses only server-reported cached-token evidence; it
does not claim a truly cold cache without reset evidence.

Coding suite (`coding_tasks.py`, 16 tasks since 2026-10-07): 10 Python functions
and bug fixes graded by fixed unit tests, 2 Swift functions compiled with
`swiftc` and graded by `precondition`s, and 4 tool-call JSON tasks compared
against an expected object. One request per task, temperature 0, pass@1.
`--coding-models M...` picks which models run it and `--coding-only` skips the
regular cases.

Code runs under a strict `sandbox-exec` profile: file reads denied except the
root directory entry, system libraries, the resolved Python runtime (or the
Xcode bundle, for `swiftc` only) and the task's own directory; writes only to
that directory; no network. Until 2026-10-07 every run reported "sandbox
unavailable": the profile did not allow reading `/` itself, which dyld needs at
process start (even `/usr/bin/true` died with SIGABRT, exit 134), and it
launched the `/usr/bin/python3` xcrun shim, which dlopens `libxcrun` from the
developer dir. The fix adds `(literal "/")` and launches the resolved
interpreter. `test_harness.py` proves the boundary: the probe reads its own
file and is refused `~/.zshenv`, a `/private/tmp` sentinel and loopback.

The AST screen in `coding_evaluator.py` is defense in depth: it rejects dunder
access, `exec`/`eval`/`open`-style names and imports outside a small stdlib
list, but accepts helpers, annotations and defaults, so correct code is graded
on behavior. JSON is parsed strictly first, then from a single fenced block;
the result detail says which (`strict JSON` / `fenced JSON`).

No-model validation:

```zsh
python3 -B -m unittest -v test_harness.py
python3 -B -m py_compile benchmark.py coding_evaluator.py summarize.py test_harness.py
```
