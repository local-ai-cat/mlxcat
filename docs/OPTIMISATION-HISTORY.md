# Optimisation history — what changed, what it bought, and where it came from

`PARITY-LEDGER.md` records mechanism-by-mechanism what the reference engines do
and what we do. `LEVERS.md` records every toggle and its exit. **This file is the
chronology**: each change, the number it bought, and — the part usually lost —
**where the idea actually came from**.

Provenance matters here for a specific reason. Very little of this engine's
performance was invented. Most of it was **read out of an existing engine** that
already had it, or **backported from an upstream fix** that already existed and
that we simply could not reach. Recording that keeps the team honest about which
wins were insight and which were literacy, and it credits work that is not ours.

## Provenance tags

| tag | meaning |
|---|---|
| `reference` | read out of another engine under `guest/` and adopted. The citation is `guest/<repo>/<path>:<line>`. |
| `upstream` | an existing fix in a dependency we could not reach from any release, backported into a fork. |
| `measurement` | found by measuring, not by reading — a benchmark or gate exposed it. |
| `review` | surfaced by a review pass (Codex threads, adversarial re-reads). |
| `model` | proposed by an AI model asked for analysis (Fable, Codex, Claude), where the idea itself came from the model rather than from a source it cited. |

---

## 2026-08-23 — free MLX's buffer cache between prefill chunks

**Provenance: `reference`** — mlx-lm has always done this.
`guest/mlx-lm/mlx_lm/generate.py:451`, `:586` call `mx.clear_cache()` at the end
of every prefill chunk; `:465` clears every 256 decode tokens. We did not.

Each chunk attends over a longer key range than the last, so no two chunks
allocate the same attention-scratch shape, MLX's buffer cache can never reuse
one, and it accumulated every shape for the whole prefill.

Peak at a 16k prompt, M5 Max, after load:

| model | before | after |
|---|---:|---:|
| Qwen3.8-27B | 53.02 GiB | **18.90** (−64%) |
| gemma-4-12B | 34.51 | **13.42** (−61%) |
| gpt-oss-20b | 35.17 | **14.69** (−58%) |
| Qwen3-Coder-30B-A3B | 24.75 | **19.71** (−20%) |

The single largest win in the engine's history, and it was not a new idea. It
was already in the reference, in a file we had.

## 2026-08-24 — MLX's RoPE fast path drops the batch axis

**Provenance: `upstream`** — `ml-explore/mlx` fixed this on 2026-05-11 in
`76a977ca` (#3498). We could not reach it: mlx-swift 0.31.6 (its newest release)
and mlx-swift `main` both vendor mlx at `ce45c525` from 2026-03-12, two months
before the fix. No release or branch of mlx-swift reaches it.

MLX's RoPE takes a fast path when the input is row-contiguous, sequence length
is 1, and the offset is a scalar (`mlx/backend/metal/rope.cpp:96`). That branch
dispatches a grid with **no batch axis** (`:134-139`), while the general branch
has it at `:149`. So it rotated batch row 0 and left rows 1..B-1 **untouched**,
and because the input is donated, they passed through unrotated.

Gemma's VLM attention was the only code in our stack reaching that primitive
with a scalar offset on a batched decode tensor. Batched gemma was producing
**wrong output**, and it had been granted entry to batching on the strength of
the biggest number on the board (gemma-4-E2B longgen c8 TTFT 39,346 → 1,793 ms).
The grant was revoked, then restored once both defects were carried in forks:
`atlas-open-sources/mlx-swift` (three-line backport) and
`atlas-open-sources/mlx-swift-lm` (per-row `cache?.ropeOffset` instead of the
scalar padded batch length).

**Credit: upstream MLX for the fix; the gate for catching that we shipped wrong
output for a day.** `RoPEBatchGridProbeTests` reproduces it in under a second
with no model — `rows-unrotated=[1, 2, 3]`, output bit-identical to input.

## 2026-08-26 — BenchHost could never have measured the model it exists for

**Provenance: `measurement`** — found by running it, on two phones, at 2am.

Two stacked bugs, each hiding the next:

1. **No entitlements file at all.** The shipping app carries
   `com.apple.developer.kernel.increased-memory-limit`; the bench host did not,
   so iOS capped it at 2.60 GiB while the app gets 4.77 GiB. gemma-4-E2B needs
   3.83 GiB resident — 47% more than the ceiling — so the headline model was
   jetsam-killed at load, ~25s in, on **both** phones, producing zero rows. Not
   bad luck, arithmetic.
2. **`chat_template.jinja` was never downloaded.** The snapshot glob had no
   `*.jinja`, and gemma-4 keeps its chat template in a standalone file rather
   than in `tokenizer_config.json`. The phone got every weight and no template,
   then threw `missingChatTemplate` at first use — after the full download and
   load, the most expensive possible place to discover a missing 20 KB file.

## 2026-08-26 — the benchmark was measuring queue position

**Provenance: `measurement`**, and it cost two wrong hypotheses first — see
`KNOWN-FAILURES.md` 1g.

A cell measured after other work is materially slower than the same cell
measured first, and `bench/parity.py` does not key on position. Qwen3.5-4B
`longgen/c1`: **1290.5 ms** as the 1st model of 2 (the baseline), **1652.3 ms**
as the 5th of 12 (the pass that turned the gate red), and **1291/1292/1291/1290
ms** across four isolated runs today. Six apparently-red cells were the
instrument.

Fixed by inter-model and inter-cell settles (`LEVERS.md`). **7 red cells → 2.**
No baseline was moved.

The reusable lesson is not about RoPE or about MLX. It is that a benchmark whose
cell identity omits execution order will silently compare unlike things, and
will look rigorous while doing it.

## 2026-08-26 — the first measured context ceiling on a phone

**Provenance: `measurement`** — `ios/BenchHost/run-context-ladder.sh`.

The app budgets **weights** against the device (`iOSLoadableCeilingBytes` asks
the kernel via `os_proc_available_memory()`), but nothing budgets the **KV
cache**: `generationProfile.contextWindow` is a static catalog number that never
consults memory. Qwen3-1.7B-4bit on an iPhone 17 Pro Max survives **16,384
tokens** at 4.67 GB peak against a 4.79 GiB ceiling, and is **jetsam-killed at
65,536**. The limit is memory, not the model's window, and nothing in the app
knows it.

---

## Open, with the evidence attached

- **`Qwen3.5-4B longgen/c4` is 25-35% off its baseline and unexplained.** Three
  isolated repeats cluster tightly (5206/4527/5208 ms) and do **not** contain the
  baseline's 3876 ms. Eliminated: the RoPE fork, contamination, the scalar-RoPE
  lever, thermal. Next step is a build bisect. Deliberately NOT re-baselined:
  the tight cluster is what makes re-baselining unsafe.
- **`parity.py` still does not key cells by position in the pass.** The settles
  reduce contamination; they do not make position part of cell identity.
- **iOS context is not device-aware.** Measured, documented, unchanged — making
  it adaptive is a product decision, not a bug fix.
