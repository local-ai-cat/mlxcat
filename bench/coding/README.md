# bench/coding — the coding-shaped workloads `run.py`'s tiers cannot express

`run.py`'s context tiers are one filler paragraph repeated to a token target
(`METHODOLOGY.md` § "Not adopted yet — 4"). That is a clean controlled input and
it cannot express the two things a local **coding** engine lives or dies on:

1. **A shared prefix across turns.** An agentic coding session sends the same
   9–18 K of file context every turn and changes the last few hundred tokens.
   Whether the engine reuses that prefix is the difference between a 0.1 s and a
   6 s turn, and `run.py`'s cold/warm modes measure prefix reuse only for an
   *identical* prompt, never for a growing conversation.
2. **A long structured output.** "Rewrite this file with X changed" is 1–2 K
   output tokens of syntactically constrained text, not 128 tokens of prose.

Rows are `mlxcat-bench/1`, the same schema and the same `bench/results/`
ledger, so `leaderboard.py` and `timeline.py` read them unchanged. The only
addition is `workload.kind` (`cached_prefix` | `file_rewrite`), which older rows
lack and which readers must treat as `"synthetic_filler"` when absent.

## Why a sibling script and not a `run.py` workload

`run.py` builds one prompt per cell and measures N independent repeats of it.
The cached-prefix workload is inherently a **sequence**: turn 1 must be cold and
turns 2–6 must follow it in order against the same server process, and the
per-turn TTFT series is the result — not a median over repeats. That is a
different control flow, not a different prompt corpus, so it lives beside
`run.py` rather than inside it.

## Usage

    python3 bench/coding/run_coding.py --base http://127.0.0.1:11700 \
        --engine-label mlxcat --model Qwen3.8-27B-4bit \
        --workloads cached_prefix,file_rewrite --runs 5

`--base`/`--engine-label`/`--model` are deliberately explicit: this script never
launches an engine. Start the engine the same way `run.py` would (or attach to
one already running) so the row's engine identity is something you asserted,
not something the script guessed.

## The corpus is real code, and it is pinned

The prompts are real files from the Local AI Cat repo, truncated to a token
target by line so the result is still parseable Swift:

- `LLMEvaluator.swift`, first 865 lines → the 9 K prefix (9,000 Qwen3.8 tokens)
- `LLMEvaluator.swift`, whole (1,846 lines) → the 18 K prefix (18,314 Qwen3.8 tokens)

Measured with the Qwen3.8-27B tokenizer on Local-AI-Chat `486d2d295`. An
earlier draft of this README planned `+ EnhancedChatViewModel.swift` for the
18 K prefix; that file alone is 26.5 K tokens, so it would have been a 45 K
prompt. The file-rewrite workload uses
`Packages/DevLoopStreamKit/Sources/DevLoopStreamKit/FrameRasterizer.swift`
(197 lines, 1,775 tokens), so a full rewrite lands in the 1–2 K output band; a
400-line file from that repo is 3.5–4.3 K tokens and would not.

The corpus is referenced by path + commit + sha256 and is **never copied into
this repo** (it is Local AI Cat's private source).

`--corpus-dir` points at the checkout; every row records
`workload.corpus.{files, sha256, prompt_tokens}` so a stranger can re-derive the
exact bytes measured.
