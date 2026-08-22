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
GATE_ORDER=(MLXSERVE_TEST_MODEL MLXSERVE_HYBRID_TEST_MODEL MLXSERVE_SLIDING_TEST_MODEL MLXSERVE_VLM_TEST_MODEL MLXSERVE_RERANK_TEST_MODEL MLXCAT_MEMORY_BUDGET_MODEL MLXSERVE_MOE_TEST_MODEL)

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
  # gpt-oss-20b is the model with the worst long-context memory behaviour we
  # have, and it had no budget at all. Measured 2026-08-23 at 16k on an M5 Max
  # WITH a 24 GiB ceiling configured and confirmed applied by the server's own
  # startup line ("memory ceiling: 24.00GB (override); allocator: memoryLimit
  # 24.00GB, cacheLimit 4.00GB"): peak physical footprint 30.12 GiB.
  #
  # That is 25% ABOVE the configured ceiling, which is the real finding —
  # MLX's memoryLimit bounds its allocator's caching behaviour, not the
  # process, so a "ceiling" is advisory. On a Mac that is a slow machine; on
  # iOS, where mlxcat also ships, exceeding the budget is a jetsam kill.
  #
  # 32 GiB is set as a REGRESSION bar just above what we measure today, not as
  # a target. The target is at or under the configured ceiling, and getting
  # there is open work.
  gpt-oss-20b-MXFP4-Q8 34359738368   # 32 GiB (measured 30.12 GiB at 16k, ceiling 24 GiB)
  #
  # The three flagships below had NO budget at all until 2026-08-23, so the gate
  # only ever measured whichever 4B `pick_model` happened to return first and a
  # regression on the models people actually run could not fail it. Measured on
  # an M5 Max, in-process through MLXCatEngine at 16k with no ceiling configured
  # (the same condition as the Qwen3.5-4B entry), reported as
  # "after load" -> "lifetime max":
  #
  #     Qwen3-Coder-30B-A3B-Instruct-4bit  16.09 -> 24.75 GiB   x1.5
  #     gemma-4-12B-it-qat-4bit            10.34 -> 34.51 GiB   x3.3
  #     Qwen3.8-27B-4bit                   14.23 -> 53.02 GiB   x3.7
  #
  # The spread is the finding, not the absolute numbers. The MoE grows 1.5x over
  # its loaded weights; the two that grow 3.3-3.7x are exactly the two that are
  # VLM-typed (`gemma4_unified`, `qwen3_5`) and therefore run with
  # `schedulerManagedTextPrefill: false` — no chunked prefill, so admission takes
  # the single-pass path and materializes a whole-prompt logits tensor. 38 GiB of
  # transient on a 27B whose weights are 14 GiB is that tensor. mlxcat also ships
  # on iOS, where exceeding the footprint is a jetsam kill rather than a slow Mac.
  #
  # These are REGRESSION BARS just above what we measure today, not targets.
  Qwen3-Coder-30B-A3B-Instruct-4bit 28991029248   # 27 GiB (measured 24.75)
  gemma-4-12B-it-qat-4bit           40802189312   # 38 GiB (measured 34.51)
  Qwen3.8-27B-4bit                  62277025792   # 58 GiB (measured 53.02)
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
