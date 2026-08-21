# bench/results

Raw benchmark evidence: one `*.jsonl` file per `bench/run.py` invocation, one
JSON object per measured cell, schema `mlxcat-bench/1` (see `../README.md`).

* Commit these files together with the regenerated `../../LEADERBOARD.md`;
  CI fails when the two disagree.
* Rows with `valid_for_leaderboard: false` (loaded host, engine error) stay here
  for audit and are never ranked.
* Engine logs land in `.logs/` (gitignored).
* File names: `<date>-<Mac model>-<run id>.jsonl`. iOS producers use the same
  schema with `platform: "ios"` and a `device.model` like `iPhone17,2`.
