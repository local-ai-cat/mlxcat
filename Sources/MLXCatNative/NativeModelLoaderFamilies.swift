import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXVLM
import Tokenizers

extension NativeModelLoader {
    static let vlmModelTypes: Set<String> = [
        "paligemma",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "idefics3",
        "gemma3",
        "gemma4",
        // Unified multimodal Gemma 4 (e.g. gemma-4-12B). Routes to the VLM
        // Gemma4Unified, which has the language_model.* weight sanitizer; the
        // LLM Gemma4Model expects flat model.layers.* and 500s with keyNotFound.
        "gemma4_unified",
        "smolvlm",
        "fastvlm",
        "llava_qwen2",
        "pixtral",
        "mistral3",
        "lfm2_vl",
        "lfm2-vl",
        "glm_ocr"
    ]

    static let scalarOffsetVLMModelTypes: Set<String> = [
        // These VLMs still derive MRoPE position ids, mask lengths, attention
        // scaling, or shared-KV RoPE offsets from scalar cache.offset values.
        // Membership means "needs a policy", not "must serialize": `qwen3_5` is
        // still listed and still scalar-offset, but the batch generator now
        // corrects that scalar per row, so its policy is multimodal solitude
        // rather than `.always`.
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "gemma4",
        "gemma4_unified",
        "mistral3",
        // GlmOcr builds scalar MRoPE position ids from cache.offset in
        // GlmOcr.swift:308, :356, and :383.
        "glm_ocr"
    ]

    // The stated reason for this list is WRONG, and the list is still right.
    //
    // It said: "explicit-demand two-request benchmarks showed lower aggregate
    // throughput for these model families under batched decode". Re-measured on
    // 2026-08-23 at the longgen tier, Qwen3-Coder-30B (`qwen3_moe`) c4:
    //
    //     serialized  TTFT 44,760 ms · prefill 24 tok/s  · aggregate 30.0 tok/s
    //     batched     TTFT  4,089 ms · prefill 254 tok/s · aggregate 54.5 tok/s
    //
    // Aggregate throughput is 82% HIGHER batched, not lower. The original claim
    // came from two requests at a short tier, where a ~1 s serial prefill
    // dominates the wall of a 128-token generation — so it measured admission
    // latency and called it throughput. bench/README.md documents that trap now.
    //
    // Keep the list anyway, for a reason nobody had checked: batched decode for
    // `qwen3_moe` breaks the LOGIT tolerance at width >= 4.
    //
    //     batch 2  maxLogitError 0.94   ok
    //     batch 4  maxLogitError 2.69   over the 1.25 tolerance
    //     batch 8  maxLogitError 2.69   over the 1.25 tolerance
    //
    // For scale, gpt-oss-20b — batching enabled, not on this list — measures
    // 1.7e-05 at the same widths. Correct batched decode is essentially exact;
    // this is five orders of magnitude away from it. No wide-margin token
    // actually flipped, so the output may well be fine in practice, but that is
    // not the standard the other families are held to.
    //
    // So the throughput justification is retired and a correctness one replaces
    // it. `qwen2` and `qwen3` have NOT been re-measured on either axis — they
    // inherit the old, now-discredited reason and are due the same treatment.
    static let batchDecodeRegressedModelTypes: Set<String> = [
        "qwen2",
        "qwen3"
    ]

    /// Families that batch correctly only up to a width, with the width they
    /// have actually been measured at.
    ///
    /// `qwen3_moe` is the whole list, and it is here rather than in
    /// `batchDecodeRegressedModelTypes` because the measurement above does not
    /// say "this family cannot batch" — it says width 2 is inside the 1.25
    /// tolerance (0.94) and widths 4 and 8 are not (2.69). Refusing width 2 as
    /// well threw away the half of the win that is earned: serialized decode
    /// measures 10.9x TTFT and 0.55x aggregate throughput against batched on
    /// this family. A ceiling takes what the numbers support and nothing more.
    ///
    /// Raising an entry requires re-running
    /// `BatchInvarianceTests.testBatchInvarianceAcrossModelFamilies` (logits,
    /// via `MLXCAT_BATCH_INVARIANCE_MODELS`) and `MoEBatchIntegrationTests`
    /// (emitted tokens through the serving engine, via
    /// `MLXSERVE_MOE_TEST_MODEL`) at the new width. Removing the entry entirely
    /// means the width->=4 numerics were fixed; the win waiting there is the
    /// rest of that 10.9x.
    /// Whether an IDLE admission chunks its prefill forward pass.
    ///
    /// Chunking removes a whole-prompt `[1, L, vocab]` logits tensor and its
    /// full-length attention scratch. On its own that was a 26-40% cut on three
    /// families and a 63% REGRESSION on a fourth, which looked like a trade
    /// needing a per-family exemption list — and was not. The missing piece was
    /// read out of the reference rather than benchmarked into existence:
    /// `mlx-lm` calls `mx.clear_cache()` at the end of every prefill chunk
    /// (`guest/mlx-lm/mlx_lm/generate.py:451`, `:586`). Each chunk attends over
    /// a longer key range than the last, so no two allocate the same
    /// attention-scratch shape and MLX's buffer cache can never reuse one; it
    /// simply accumulates every shape for the whole prefill. Freeing between
    /// chunks is what makes chunked prefill cheap for them, and it was the
    /// entire difference. Measured at 16k on an M5 Max, after load -> peak:
    ///
    ///     model                  single-pass   chunked   chunked+clear
    ///     Qwen3.8-27B              53.02        39.02      18.90   -64%
    ///     gemma-4-12B              34.51        20.74      13.42   -61%
    ///     gpt-oss-20b              35.17        25.37      14.69   -58%
    ///     Qwen3-Coder-30B-A3B      24.75        40.34      19.71   -20%
    ///
    /// Every family wins, so `singlePassPrefillModelTypes` is EMPTY. It is kept
    /// rather than deleted because the mechanism that would refill it is
    /// understood: a family whose chunked scratch cannot be freed between chunks
    /// would belong here. Peak is now 1.22-1.33x loaded weights across all four,
    /// against 1.5-3.7x before — the consistency is the tell that this addressed
    /// the mechanism and not a symptom.
    ///
    /// `MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES` overrides the list (`all` for
    /// every family) so the benchmark can A/B it without a rebuild.
    static func chunksIdlePrefill(
        modelType: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let normalized = modelType.lowercased()
        let raw = environment["MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES"] ?? ""
        let overrides = Set(
            raw.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                .filter { !$0.isEmpty }
        )
        if !overrides.isEmpty {
            return !(overrides.contains("all") || overrides.contains(normalized))
        }
        return !singlePassPrefillModelTypes.contains(normalized)
    }

    /// Families measured to peak HIGHER with a chunked idle prefill than with a
    /// single whole-prompt pass. Empty since the per-chunk `Memory.clearCache()`
    /// landed — see `chunksIdlePrefill` for the numbers that emptied it.
    private static let singlePassPrefillModelTypes: Set<String> = []

    static let batchDecodeWidthCeilings: [String: Int] = [
        // RAISED 2 -> 8 on 2026-08-24, on a complete evidence pack re-measured
        // that day (M4 Pro, Qwen3-Coder-30B-A3B-Instruct-4bit):
        //
        //   * `MoEBatchIntegrationTests` — emitted tokens through the engine —
        //     PASSES with strict serial-greedy equality at widths 2, 4 AND 8.
        //   * `MoEWidthNumericsProbeTests` (the discriminator that settled
        //     gemma, KNOWN-FAILURES 1f): IDENTICAL rows decoded together are
        //     token-identical to serial at widths 2/4/8 — vsSerial=0 and
        //     vsEachOther=0 everywhere, so rows are NOT crossing. Real ragged
        //     prompts: 1 late flip (step 7 of 12) in 8 rows — the near-tie
        //     batch-variance category, the same residue gemma's grant carries.
        //   * `BatchInvarianceTests` still reads maxLogitError 2.6875 at
        //     widths 4/8 (0.5 at width 2, 0.0 at width 1) against the 1.25
        //     tolerance, with zero wide-margin flips and margin 6.5 at the
        //     max-error step. With contamination excluded by the probe, that
        //     number is width-level accumulation order in the expert mix — an
        //     MoE reduction reordering with width, not a defect a per-row fix
        //     could touch.
        //
        // The old width-2 entry guarded against exactly the contamination the
        // probe now rules out; what remains over the tolerance is explained
        // and token-invisible on every end-to-end gate. 8 is the widest width
        // MEASURED, so 8 is the ceiling — "what the numbers support and
        // nothing more". This cap was ~2/3 of the whole c8 board gap (the
        // `mlxcat-moe-uncapped` arm priced it at +46-48% aggregate on the
        // coder cells); deleting the entry outright (unbounded) waits on a
        // width-16+ measurement, not on more adjudication.
        "qwen3_moe": 8,
        // gemma4_unified is here for a different reason than qwen3_moe, and the
        // difference is the whole argument.
        //
        // `multimodalOnlyModelTypes` documents entry as requiring both the logit
        // gate and the image gate. The logit evidence on record is gemma-4-E2B —
        // a `gemma4`. gemma-4-12B, the `gemma4_unified` that shares the grant,
        // had never been run through it. Run 2026-08-23 on an M5 Max, with a
        // width-1 CONTROL arm added so a path difference could be told from a
        // batching defect:
        //
        //     batch 1  maxLogitError 0.0       margin  0.0    (exact; path cleared)
        //     batch 2  maxLogitError 1.65625   margin 11.5    over 1.25
        //     batch 4  maxLogitError 1.65625   margin 11.5    over 1.25
        //     batch 8  maxLogitError 1.7402    margin  1.125  over 1.25, and OVER MARGIN
        //
        // Width 1 being bit-exact means this is rows contaminating each other,
        // not the batched path's own numerics — the defect is real. But the
        // widths are not equally dangerous. A token can only flip when the error
        // exceeds the top-1/top-2 margin; at widths 2 and 4 the margin is 7x the
        // error, and at width 8 the error EXCEEDS the margin. `mismatched` is 0
        // everywhere, so nothing has flipped yet — at width 8 that is luck, and
        // at widths 2 and 4 it is headroom.
        //
        // A ceiling of 4 removes the width where a flip is possible and keeps the
        // one the concurrency win was measured at (c4 TTFT 11,999 -> 870 ms,
        // `d630cee`). It is strictly safer than today, which is unbounded.
        //
        // It does NOT clear the 1.25 tolerance at any width >= 2, and that is
        // still an open defect: gemma-4 is the iOS flagship and `gemma4_unified`
        // is a distinct loader path from `gemma4`, which passes at 0.69. Whether
        // the answer is to fix that path or to drop to 1 (i.e. `.always`) is
        // Phil's call — see docs/KNOWN-FAILURES.md 1e.
        "gemma4_unified": 4
    ]

    static let windowedKVModelTypes: Set<String> = [
        "gemma3",
        "gemma4",
        "gemma4_unified",
        "gpt_oss"
    ]

    static let vlmProcessorClasses: Set<String> = [
        "PaliGemmaProcessor",
        "Qwen2VLProcessor",
        "Qwen2_5_VLProcessor",
        "Qwen3VLProcessor",
        "Idefics3Processor",
        "Gemma3Processor",
        "Gemma4Processor",
        "Gemma4UnifiedProcessor",
        "SmolVLMProcessor",
        "FastVLMProcessor",
        "PixtralProcessor",
        "Mistral3Processor",
        "Lfm2VlProcessor",
        "Glm46VProcessor"
    ]

}
