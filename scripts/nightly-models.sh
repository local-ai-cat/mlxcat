#!/bin/zsh
# nightly-models.sh — run every model-gated suite with real local models.
#
# The gates exist because `swift test` must never download anything; without
# this script they skip forever ("a live test behind an unset env var is not
# coverage"). This wires each gate to a local mlx-community directory and runs
# the suites one model family at a time, serializing model loads.
#
#   scripts/nightly-models.sh                    # all gates that have a model on disk
#   MLXCAT_MODEL_ROOT=/path scripts/nightly-models.sh
#
# Writes a summary to $MLXCAT_NIGHTLY_OUT (default: .build/nightly-models.md).
# Exit code = number of failing suites. Not a CI job: a public repo cannot use
# our self-hosted Apple Silicon runners; run this on your own Mac (or the m4
# worker) and commit/attach the summary.

set -uo pipefail
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
ROOT="${MLXCAT_MODEL_ROOT:-$HOME/Library/Caches/models/mlx-community}"
OUT="${MLXCAT_NIGHTLY_OUT:-$REPO_ROOT/.build/nightly-models.md}"
mkdir -p "${OUT:h}"

# gate → preferred model dirs, in order (first that exists wins)
typeset -A GATE_CANDIDATES
GATE_CANDIDATES=(
  MLXSERVE_TEST_MODEL          "Qwen3-0.6B-4bit Qwen3-1.7B-4bit Llama-3.2-1B-Instruct-4bit"
  MLXSERVE_HYBRID_TEST_MODEL   "Qwen3.5-4B-MLX-4bit Qwen3.8-27B-4bit"
  # Qwen3.5 is the only family that anchors its RoPE in LMOutput.State rather
  # than reading it off the cache. Dropping that anchor when a second row joined
  # is what killed the benchmark's batched qwen3_5 server (KNOWN-FAILURES 1d);
  # the gate is token-exactness against serial at 4 ragged rows and at width 8.
  MLXCAT_POSITIONAL_TEST_MODEL "Qwen3.5-4B-MLX-4bit Qwen3.8-27B-4bit"
  # KV quantization A/B. Qwen3-Coder-30B is the strongest signal (all 48 layers
  # are full-attention, 96 KiB/tok at fp16); Qwen3.5-4B converts 8 of 32 and
  # still shows it clearly. gpt-oss is deliberately absent — it is excluded from
  # quantization because its quantized attention route drops attention sinks.
  MLXCAT_KV_QUANT_TEST_MODEL   "Qwen3.5-4B-MLX-4bit Qwen3-Coder-30B-A3B-Instruct-4bit"
  # Batched gemma against serial, on the factory PRODUCTION loads. Expected to
  # FAIL today — see docs/KNOWN-FAILURES.md 1f. It is here rather than in the
  # default suite because a standing red gate is how this repo already carries
  # an open numerics defect (the batch-invariance FAIL rows below), and because
  # the number needs to be in front of us every night until it is zero.
  MLXCAT_GEMMA_PROBE_MODEL     "gemma-4-12B-it-qat-4bit gemma-4-E2B-it-qat-4bit"
  # gpt-oss-20b FIRST: its sliding_window is 128, so the gate's 160 decode steps
  # actually cross it. gemma-4's window is 512 and the gate never reached it —
  # for its whole life this was a batched-vs-serial test wearing a window test's
  # name. The suite now fails loudly rather than run vacuously (2026-08-22).
  MLXSERVE_SLIDING_TEST_MODEL  "gpt-oss-20b-MXFP4-Q8 gemma-4-E2B-it-qat-4bit gemma-4-E4B-it-qat-4bit"
  MLXSERVE_MOE_TEST_MODEL      "Qwen3-Coder-30B-A3B-Instruct-4bit Qwen3.6-35B-A3B-4bit gpt-oss-20b-MXFP4-Q8"
  MLXSERVE_VLM_TEST_MODEL      "Qwen2-VL-2B-Instruct-4bit gemma-4-E2B-it-qat-4bit"
  MLXSERVE_RERANK_TEST_MODEL   "bge-reranker-v2-m3-4bit mxbai-rerank-base-v1"
  # MLXCAT_MEMORY_BUDGET_MODEL has no candidate list: it runs against EVERY
  # model in MEMORY_BUDGETS that is on disk, because "first candidate wins" is
  # the wrong shape for a per-model regression bar.
)
# which swift test --filter each gate drives (keeps one model resident at a time)
typeset -A GATE_FILTERS
GATE_FILTERS=(
  MLXSERVE_TEST_MODEL          "BatchInvarianceTests|SchedulerEngineTests|SchedulerChunkedPrefillTests|SchedulerPreemptionTests|SchedulerSpeculativeDecodingTests|PrefixSchedulerIntegrationTests|TrackAPrefixCacheTests|M1ParityIntegrationTests|SessionPrefixKVStoreTests|SingletonCachePassthroughTests|PerRequestRNGTests|BatchSamplingTests|StructuredOutputTests|ToolSelectionTests|GateProbeTests|EnginePoolTests"
  MLXSERVE_HYBRID_TEST_MODEL   "HybridBatchIntegrationTests|ContextWindowParityTests"
  MLXCAT_POSITIONAL_TEST_MODEL "PositionalStateBatchIntegrationTests"
  MLXCAT_KV_QUANT_TEST_MODEL   "KVQuantizationMemoryTests"
  MLXCAT_GEMMA_PROBE_MODEL     "GemmaRoPEOffsetProbeTests"
  MLXSERVE_SLIDING_TEST_MODEL  "SlidingWindowBatchIntegrationTests"
  MLXSERVE_MOE_TEST_MODEL      "MoEBatchIntegrationTests"
  MLXSERVE_VLM_TEST_MODEL      "ModelCacheCapabilitiesTests|ModelDiscoveryTests"
  MLXSERVE_RERANK_TEST_MODEL   "RerankTests"
  MLXCAT_MEMORY_BUDGET_MODEL   "MemoryBudgetTests"
)

# Cross-family batch invariance is not gated on ONE model — it takes a list, and
# the list is the point. NativeModelLoader.usesSerializedDecode turns batched
# decode off for gemma4, gemma4_unified, qwen3_5, qwen3_vl, mistral3, qwen2,
# qwen3 and qwen3_moe, and until 2026-08-22 the gate that would justify those
# exclusions was pinned to Qwen3-0.6B-4bit — a model on neither list. Run this
# and the exclusions have to be re-earned against real numbers.
INVARIANCE_MODELS=(
  Qwen3.5-4B-MLX-4bit
  gemma-4-E2B-it-qat-4bit
  gemma-4-12B-it-qat-4bit
  Qwen3-Coder-30B-A3B-Instruct-4bit
  gpt-oss-20b-MXFP4-Q8
)
GATE_ORDER=(MLXSERVE_TEST_MODEL MLXSERVE_HYBRID_TEST_MODEL MLXCAT_POSITIONAL_TEST_MODEL MLXCAT_KV_QUANT_TEST_MODEL MLXCAT_GEMMA_PROBE_MODEL MLXSERVE_SLIDING_TEST_MODEL MLXSERVE_VLM_TEST_MODEL MLXSERVE_RERANK_TEST_MODEL MLXCAT_MEMORY_BUDGET_MODEL MLXSERVE_MOE_TEST_MODEL)

pick_model() {
  local gate="$1" name
  for name in ${=GATE_CANDIDATES[$gate]}; do
    [[ -f "$ROOT/$name/config.json" ]] && { echo "$ROOT/$name"; return 0; }
  done
  return 1
}

wait_for_memory() {
  # Gate each load on free memory (same discipline as the fleet campaign).
  local free tries=0
  while (( tries < 60 )); do
    free=$(memory_pressure -Q 2>/dev/null | sed -nE 's/.*free percentage: *([0-9]+)%.*/\1/p')
    [[ -z "$free" || "$free" -ge 40 ]] && return 0
    (( tries++ )); sleep 10
  done
  echo "warning: memory free stayed below 40% — continuing anyway" >&2
}

echo "# nightly model gates — $(date -u +%F' '%R) UTC" > "$OUT"
echo "" >> "$OUT"
echo "| gate | model | filter | result |" >> "$OUT"
echo "|---|---|---|---|" >> "$OUT"

cd "$REPO_ROOT"
echo "building tests once…"
set -o pipefail
if ! swift build --build-tests 2>&1 | grep -E "error:|Build complete" | tail -5; then
  echo "test build FAILED — refusing to report gate results from a broken tree" | tee -a "$OUT"
  exit 90
fi
set +o pipefail

# Per-model 16k peak-footprint budgets (bytes) for MemoryBudgetTests. These PIN
# TODAY'S MEASURED BEHAVIOUR (pkg 99/102 matrices + margin) so the cliff cannot
# silently worsen; pkg 102's fix is expected to LOWER them, not the reverse.
typeset -A MEMORY_BUDGETS
MEMORY_BUDGETS=(
  Qwen3.5-4B-MLX-4bit  30064771072   # 28 GiB (measured 25.05 GiB shipping path, no ceiling)
  Qwen3-1.7B-4bit      15032385536   # 14 GiB (scaled from 0.6B's measured 10.52 GiB)
  #
  # The flagships below had NO budget at all until 2026-08-23, so the gate only
  # ever measured whichever 4B `pick_model` returned first and a regression on
  # the models people actually run could not fail it.
  #
  # Measured on an M5 Max at 16k, in-process through MLXCatEngine with no
  # ceiling configured (the same condition as the Qwen3.5-4B entry). The gate
  # reads model_type off the checkpoint and asks NativeModelLoader which prefill
  # path the family takes, so it cannot drift from what ships.
  #
  # These fell hard once the per-chunk Memory.clearCache() landed — the one
  # mlx-lm has always had (guest/mlx-lm/mlx_lm/generate.py:451, :586), which we
  # simply did not:
  #
  #     model                  before   after   peak / loaded weights
  #     Qwen3.8-27B             53.02   18.83    3.7x -> 1.32x
  #     gemma-4-12B             34.51   13.45    3.3x -> 1.30x
  #     gpt-oss-20b             35.17   14.99    3.1x -> 1.31x
  #     Qwen3-Coder-30B-A3B     24.75   19.71    1.5x -> 1.22x
  #
  # The two windowed families (gemma-4, gpt-oss) are measured with
  # `usesWindowedKVCache: true` declared, which is what production derives and
  # what the constructor default got wrong. They read 12.92 and 14.69 before
  # that was fixed — a path production never selects, because a rotating cache
  # does not take the last-token-alone prefill.
  #
  # The consistency of the after column is the tell: 1.22-1.33x of loaded weights
  # across four unrelated architectures, where before it ranged 1.5-3.7x.
  #
  # Bars are ~10% above measured. They are REGRESSION bars, not targets — the
  # target is what mlx-lm and omlx measure for the same model and context, and
  # `bench/parity.py` is the gate that scores us against them.
  #
  # gpt-oss's old 32 GiB bar came from a 30.12 GiB SERVER measurement with a
  # 24 GiB ceiling — a different condition from this gate's. The finding that
  # made it worth recording is unchanged and lives in docs/KNOWN-FAILURES.md 1c:
  # MLX's memoryLimit bounds its allocator's caching, not the process, so a
  # "ceiling" is advisory. On a Mac that is a slow machine; on iOS a jetsam kill.
  gpt-oss-20b-MXFP4-Q8              17394617549   # 16.2 GiB (measured 14.69)
  Qwen3-Coder-30B-A3B-Instruct-4bit 23300074701   # 21.7 GiB (measured 19.71)
  gemma-4-12B-it-qat-4bit           15852223037   # 14.8 GiB (measured 13.42)
  Qwen3.8-27B-4bit                  22333829939   # 20.8 GiB (measured 18.90)
)
rc_total=0

# One gate against one model. Factored out of the loop because the memory-budget
# gate runs it once PER MODEL: `pick_model` returns the first candidate on disk,
# which meant exactly one model was ever measured for peak footprint no matter
# how many had committed budgets — so a regression on the flagship passed
# silently behind a 4B that did not regress.
run_gate() {
  local gate="$1" model="$2" filter="$3" budget="${4:-}" label="${5:-}"
  local log rc xctest_line testing_line xctest_count testing_count total summary failures
  local -a budget_env
  budget_env=()
  [[ -n "$budget" ]] && budget_env=(MLXCAT_MEMORY_BUDGET_BYTES="$budget")
  # Distinct log per model, or the second model of a gate overwrites the first's
  # evidence and the summary row points at someone else's failure.
  log="$REPO_ROOT/.build/nightly-${gate}${label:+-$label}.log"
  wait_for_memory
  echo "▶ $gate → ${model:t}  ($filter)"
  # Only the gate under test is set; others stay unset so their suites skip.
  env -u MLXSERVE_TEST_MODEL -u MLXSERVE_HYBRID_TEST_MODEL -u MLXSERVE_SLIDING_TEST_MODEL -u MLXSERVE_MOE_TEST_MODEL \
      -u MLXSERVE_VLM_TEST_MODEL -u MLXSERVE_RERANK_TEST_MODEL -u MLXCAT_MEMORY_BUDGET_MODEL \
      -u MLXCAT_POSITIONAL_TEST_MODEL -u MLXCAT_KV_QUANT_TEST_MODEL -u MLXCAT_GEMMA_PROBE_MODEL \
      "${budget_env[@]}" "$gate=$model" swift test --skip-build --filter "$filter" > "$log" 2>&1
  rc=$?
  # This package runs BOTH test libraries, so `swift test` prints two summaries:
  # XCTest's "Executed N tests, with M failures" and then swift-testing's
  # "Test run with N tests in M suites". Taking `tail -1` took the swift-testing
  # line — which is 0/0 whenever the filter matches no swift-testing suite, i.e.
  # for every gate here. On 2026-08-22 that printed "PASS — 0 tests in 0 suites"
  # over a run that had executed 100 XCTest cases with 7 failures.
  #
  # Note "Executed 1 test" is SINGULAR; a `tests` regex silently misses any gate
  # with exactly one test, which is most of the integration gates.
  xctest_line=$(grep -E "Executed [0-9]+ tests?," "$log" | tail -1)
  testing_line=$(grep -E "Test run with [0-9]+ tests? in [0-9]+ suites?" "$log" | tail -1)
  xctest_count=$(sed -nE 's/.*Executed ([0-9]+) tests?,.*/\1/p' <<< "$xctest_line")
  testing_count=$(sed -nE 's/.*Test run with ([0-9]+) tests?.*/\1/p' <<< "$testing_line")
  total=$(( ${xctest_count:-0} + ${testing_count:-0} ))
  summary="${xctest_line:-$testing_line}"
  summary="${summary##[[:space:]]#}"

  if (( total == 0 )); then
    # Zero tests exits 0. A gate that ran nothing is the one result we must never
    # record as PASS — that is how a filter that stopped matching goes unnoticed.
    (( rc_total++ ))
    echo "| $gate | ${model:t} | \`$filter\` | **NO TESTS RAN** — the filter matched nothing; a gate that runs nothing is not a pass — see ${log:t} |" >> "$OUT"
  elif (( rc == 0 )); then
    echo "| $gate | ${model:t} | \`$filter\` | PASS ($total tests) — $summary |" >> "$OUT"
  else
    (( rc_total++ ))
    failures=$(grep -cE "^.*: error:" "$log")
    echo "| $gate | ${model:t} | \`$filter\` | **FAIL** (rc=$rc, $failures assertion error line(s)) — $summary — see ${log:t} |" >> "$OUT"
  fi
}

for gate in $GATE_ORDER; do
  filter="${GATE_FILTERS[$gate]}"
  if [[ "$gate" == MLXCAT_MEMORY_BUDGET_MODEL ]]; then
    # Every model with a committed budget that is on disk, not just the first.
    ran_any=0
    for name in ${(ok)MEMORY_BUDGETS}; do
      [[ -f "$ROOT/$name/config.json" ]] || continue
      ran_any=1
      run_gate "$gate" "$ROOT/$name" "$filter" "${MEMORY_BUDGETS[$name]}" "$name"
    done
    (( ran_any )) || echo "| $gate | (no budgeted model on disk) | — | SKIP |" >> "$OUT"
    continue
  fi
  model="$(pick_model "$gate")" || { echo "| $gate | (no candidate model on disk) | — | SKIP |" >> "$OUT"; continue; }
  run_gate "$gate" "$model" "$filter"
done
# --- cross-family batch invariance ------------------------------------------ #
present=()
for m in "${INVARIANCE_MODELS[@]}"; do
  [[ -d "$ROOT/$m" ]] && present+=("$ROOT/$m")
done
if (( ${#present[@]} )); then
  wait_for_memory
  inv_log="$REPO_ROOT/.build/nightly-batch-invariance.log"
  echo "▶ batch invariance across ${#present[@]} model(s)"
  MLXCAT_BATCH_INVARIANCE_MODELS="${(j:,:)present}"     swift test --skip-build --filter testBatchInvarianceAcrossModelFamilies > "$inv_log" 2>&1
  inv_rc=$?
  echo "" >> "$OUT"
  echo "### batch invariance by family" >> "$OUT"
  echo "" >> "$OUT"
  echo '```' >> "$OUT"
  grep -E "^FAMILY" "$inv_log" >> "$OUT" || echo "(no FAMILY lines — the filter matched nothing)" >> "$OUT"
  echo '```' >> "$OUT"
  if (( inv_rc != 0 )); then
    (( rc_total++ ))
    echo "" >> "$OUT"
    echo "**A family failed batch invariance — see ${inv_log:t}.**" >> "$OUT"
  elif ! grep -q "^FAMILY" "$inv_log"; then
    (( rc_total++ ))
    echo "" >> "$OUT"
    echo "**No families were checked — a gate that runs nothing is not a pass.**" >> "$OUT"
  fi
else
  echo "" >> "$OUT"
  echo "_batch invariance skipped: none of ${INVARIANCE_MODELS[*]} are under \`$ROOT\`_" >> "$OUT"
fi

echo "" >> "$OUT"
echo "_root: $ROOT · host: $(sysctl -n hw.model) $(sysctl -n machdep.cpu.brand_string) · load $(sysctl -n vm.loadavg)_" >> "$OUT"
cat "$OUT"
exit $rc_total
