## What

<!-- one paragraph: the change and the why -->

## Platform impact

- [ ] macOS server (`mlxcat-http`)
- [ ] macOS embedded (in-app)
- [ ] iOS embedded — memory / threading / Metal-stream implications considered

## Verification

- [ ] `swift test` (no-model suite)
- [ ] model-gated suites run (`scripts/nightly-models.sh` or the specific `MLXSERVE_*_TEST_MODEL`): which models →
- [ ] bench rows (`python3 bench/run.py …`) on a quiet machine, if this touches performance → paste the `[engine/model/tier]` lines
- [ ] real device (iPhone) if this touches the embed path → device + result

## Checklist

- [ ] conventional commit messages, one logical change per commit
- [ ] `LEADERBOARD.md` regenerated if `bench/results/` changed
- [ ] CHANGELOG `Unreleased` updated for user-visible changes
- [ ] pin bumps are their own PR with drift report + model gates + bench
