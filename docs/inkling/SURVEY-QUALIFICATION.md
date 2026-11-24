# Kestrel and DaisyRose qualification

## Checkpoints and licence

The measurements use the verified 4-bit affine releases of
[Kestrel-4B-Coder-4bit](https://huggingface.co/mlx-community/Kestrel-4B-Coder-4bit/tree/ba76aeaa3ecce1f7beb1473b7ff833dc31399f0c)
and
[DaisyRose-27B-4bit](https://huggingface.co/mlx-community/DaisyRose-27B-4bit/tree/895983ea8d4244e855f6fbde2c3fa8261af202ef).
The weight indexes and shard extents were rechecked before native use.
Both verified cards identify Apache-2.0, as do the
[Kestrel upstream card](https://huggingface.co/Jackrong/Kestrel-4B-Coder)
and [DaisyRose upstream card](https://huggingface.co/RadixArk/DaisyRose-27B).
A Cat entry pointing to a direct upstream download is compatible with
those terms. The catalogue gate also requires native parity.
[Apache 2.0 terms](https://www.apache.org/licenses/LICENSE-2.0)

## Parity

On the M4 Pro, Python mlx-lm 0.31.3 with MLX 0.32.2 loaded the verified
weights and used each pack's standalone Jinja template. It ran the 11
text fixtures from the Inkling suite. The image fixture requires an
image-capable reference. Text fixtures include the 25-token coding
prompt, an 8,225-token prompt, multi-turn, reasoning, tools and Unicode.
Both engines used the same prefill partition for model-level comparisons
and checked every greedy position through the reference EOS or 64 tokens.
The Swift test compares eight sampled logits per row for the first ten
rows, following the reference token stream after a mismatch so the
numeric inputs remain aligned. The standing logit tolerance is 0.25.
Greedy totals count each fixture's matching prefix.

### Kestrel

Ten of eleven rendered prompts matched. The tool fixture first differed
at input token index 20 in the template's `tool | tojson` output. Python
placed the `type` key first, and Swift placed `function` first. With reference
input IDs forced for that case, the model matched 220/330 greedy IDs
across the suite. Eight hundred fifty-six sampled scores had a maximum
absolute error of 0.6875.

The coding trace first differed in layer zero's affine quantized QKV
projection, at `MLXVLM/Models/Qwen35.swift:571–579` and
`MLXNN/Quantized.swift:367–375`. Embeddings and the input RMS norm were
bit-identical. The QKV projection's maximum delta was 0.125. The first
gated delta layer's output delta was 0.015625 with either its fused or
separate input projections in Swift. Disabling projection fusion leaves
the observed discrepancy.

The HTTP coding fixture emitted the same 64 IDs. On the long fixture it
emitted the same first six IDs and stopped at tokenizer EOS `248046`.
The Python reference continued with a line break and model EOS `248044`.
Native serving combines the tokenizer and model EOS IDs
(`NativeModelEngine.swift:59–63`).

### DaisyRose load gate

The results below predate the native tokenizer alias. The later loader and
cross-version kernel results are in [Affine projection parity](AFFINE-PARITY.md).

The unmodified checkpoint failed the Swift load and returned HTTP 500.
Its `Qwen3_5Tokenizer` class reaches the strict unknown-class check in
swift-transformers `Sources/Tokenizers/Tokenizer.swift:183–199`.
A scratch diagnostic used the original tokenizer data, template and
weights and changed only the class label to `Qwen2Tokenizer`.
All 11 Python-rendered prompt ID arrays remained identical with that
label.

With the diagnostic alias, ten of eleven rendered prompts matched.
The tool fixture showed the same JSON key-order mismatch as Kestrel.
The matching greedy prefixes summed to 292/307 IDs and the maximum
absolute error over 808 sampled scores was 1.25. On the same coding
input IDs the embeddings, input RMS norm and QKV projection matched.
The first observed difference was the quantized `in_proj_z` gate
projection, with a maximum delta of 0.125
(`MLXVLM/Models/Qwen35.swift:571–579`). The full gated delta output had
a maximum delta of 0.125 with either fused or separate projections.

## Kestrel context and 16 GiB cap

Each context run used a fresh release server, warmed it with a short
request and measured a cold prompt. The server provides TTFT and
generation throughput. Peak is the process physical footprint sampled
from model warm-up through completion. The 32-token prompt requested
128 outputs and the other prompts requested 32. All runs below returned
HTTP 200.

| Input | Decode tok/s | TTFT s | Peak GiB |
| --- | ---: | ---: | ---: |
| 32 | 69.21 | 0.064 | 3.53 |
| 4,096 | 67.79 | 5.313 | 4.55 |
| 8,192 | 63.64 | 10.973 | 4.97 |
| 8,192 with 16 GiB engine cap | 63.84 | 10.969 | 4.96 |

Kestrel cleared the same 16 GiB software-cap test used for Inkling.
All 32 capped output IDs matched the uncapped 8k arm.
The capped request left about 11 GiB below 16 GiB at the measured
process peak, evidence for a single 8k session on a 16 GB Mac.
The host used for this test has 48 GiB of physical RAM.

## DaisyRose context with the tokenizer alias

The context numbers below use the scratch alias and the same native
release server and footprint method as Kestrel. Each returned HTTP 200.
The short and long HTTP parity samples also matched 48/48 and 5/5
emitted IDs respectively, after removing the terminal tokenizer EOS.

| Input | Decode tok/s | TTFT s | Peak GiB |
| --- | ---: | ---: | ---: |
| 32 | 13.61 | 0.362 | 17.53 |
| 4,096 | 13.55 | 33.740 | 17.78 |
| 8,192 | 12.66 | 68.499 | 18.49 |

The observed 8k process peak leaves about 13.5 GiB below 32 GiB on
the test host. The original pack fails model loading before decoding.

## Whisker T2

The frozen Whisker runner made three resets of the same fixture for each
arm. Temperature was zero, seed 42, with Bash allowed inside the reset
sandbox and a 30-round limit. Completion required both pytest and the independent duplicate
and blank-label checks. The separate counts include generated Qwen tool
frames that produced no invocation on token ID `248058`.

Kestrel completed 0/3 valid runs. The totals were 0 declined, 1 malformed
and 5 tool errors. Median wall time was 74.17 s, decode 65.44 tok/s,
turn TTFT 0.773 s, peak 6.04 GiB and 10 turns. The original DaisyRose pack produced
three invalid runs because no model request completed. Those attempts
yielded zero completed turns and zero tool calls.

The DaisyRose alias completed 3/3 valid runs with 0 declined,
0 malformed and 0 tool errors. Median wall time was 256.49 s,
decode 13.18 tok/s, turn TTFT 4.711 s, peak 21.19 GiB and 9 turns.

## Catalogue decision and verification

Neither checkpoint meets the catalogue parity gate. DaisyRose's alias
also requires a tokenizer compatibility change before it can run from
its upstream download. The optional family comparison had no complete
checkpoint to use. The sole Gemma 4 candidate in the local model stores
contained only its final shard, and those stores had no GPT-OSS pack.

Python capture, native context and Whisker measurements ran on the M4
under the exclusive lease. Raw receipts are local and unpublished.
The default Swift suite passed 568 XCTest cases with 67 opt-in skips,
plus 10 Swift Testing cases. The leaderboard and benchmark parity
checks passed.

## Reproduction

In an mlx-lm development venv, capture the Python reference on the same
verified checkpoint. The JSON receipt is the input to the Swift test.
The companion safetensors file adds the coding fixture's operator trace.

```sh
"$reference_venv/bin/python3" Benchmarks/survey_reference.py "$checkpoint" "$inkling_fixtures" "$reference"
MLXCAT_SURVEY_MODEL="$checkpoint" MLXCAT_SURVEY_REFERENCE="$reference" \
  swift test --filter SurveyModelParityTests
```

Run model-backed comparisons under the machine's exclusive lease.
