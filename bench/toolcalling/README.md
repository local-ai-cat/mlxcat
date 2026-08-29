# Tool-calling evidence bench

The end-to-end A/B that the tool-grammar lever still owed after
`TOOLGRAMMAR-RESULTS.md`: the lever shipped default-ON priced by 12 quote-heavy
prompts on one model, and its "Still open" list admits that nothing measures
the grammar's application cost or behavioral payoff end to end. This lane
measures both, across the fleet, with the raw response for every trial kept as
evidence.

## What a run measures

Per (model × arm × temperature × scenario × trial), one JSONL evidence row:

- **called / n_calls** — did a tool call come back at all, how many
- **json_valid** — every `function.arguments` string parses as JSON
- **schema_valid** — parsed arguments validate against the offered tool's
  parameter schema (required keys, types, enums, nested arrays of objects,
  `additionalProperties: false`)
- **name_correct** — the model called the tool it was offered, not a hallucinated one
- **false calls** — `auto_no_call` offers a tool the prompt gives no reason to
  use; calling it is a failure the forced-call smoke can never see
- **round trip** — after a `tool` result message, the model must answer with
  prose containing the result, not call again
- **streaming** — every call-expected cell runs twice, non-streaming and
  streaming; the streaming arm reassembles `tool_calls` deltas and applies the
  same validators, so a delta-encoding bug shows up as a streaming-only failure
- **tok/s and wall time** — the live price of the arm; `ttfc_ms` (time to
  first chunk) on streaming trials

## Arms

- `grammar-on` / `grammar-off` — mlxcat's per-request `tool_grammar` bool, so
  both arms hit ONE server process and differ by nothing else
- `default` — no field sent; use with `--engine-label lmstudio` (etc.) for the
  cross-engine comparison the original A/B ran once on 12 prompts

Models whose tool dialect the grammar does not cover should show
indistinguishable arms — that is evidence the lever is inert there, worth
having on record, not noise.

## Scenarios

`forced` (quote-heavy bash, the original A/B's territory) · `auto_call`
(unforced choice) · `auto_no_call` (false-call rate) · `parallel` (two calls
in one turn) · `nested_schema` (arrays of objects, required keys) ·
`unicode_args` (CJK/emoji/tabs/newlines through the escaping) · `long_args`
(enclosure integrity over a long free-text argument) · `round_trip` (turn 2
after a tool result).

## Running

```bash
swift build -c release --product mlxcat-http
.build/release/mlxcat-http --host 127.0.0.1 --port 11702 \
    --model-dir ~/Library/Caches/models &

python3 bench/toolcalling/run.py --base http://127.0.0.1:11702 \
    --models Qwen3-Coder-30B-A3B-Instruct-4bit,Qwen3.5-4B-MLX-4bit
```

Evidence lands in `bench/toolcalling/evidence/<date>-<host>-<runid>.jsonl`;
the summary table prints at the end and is derived from the evidence rows,
never maintained by hand. `campaign.sh` runs the default fleet.
