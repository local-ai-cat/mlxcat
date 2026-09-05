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
    /// Families whose serialization exclusion has been lifted for a measurement,
    /// comma-separated model types, or `all`.
    ///
    /// The two exclusion lists below switch batched decode off for five of the
    /// six models on the leaderboard, and that costs mlxcat its concurrency:
    /// TTFT scales 15-86x from c1 to c8 where every competitor scales 4-9x
    /// (`docs/COMPETITIVE.md`). Neither list has been re-earned. The logit-level
    /// gate that would justify them, `BatchInvarianceTests`, was pinned to
    /// Qwen3-0.6B-4bit — a model on neither list — so it had never been run
    /// against a single excluded family. Run across families on 2026-08-22,
    /// gemma-4-E2B (0.69/0.69/0.95 at batch 2/4/8) and Qwen3.5-4B
    /// (0.0/0.38/0.66) both come in UNDER the 1.25 tolerance and under
    /// Qwen3-0.6B's own 1.19, with zero mismatched wide-margin tokens.
    ///
    /// That is evidence, not a decision: this override exists so the benchmark
    /// can A/B the lists on real workloads instead of the question being settled
    /// by argument. Nothing changes by default.
    public static func serializationOverrides(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Set<String> {
        let raw = environment["MLXCAT_UNSERIALIZE_MODEL_TYPES"] ?? ""
        return Set(
            raw.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                .filter { !$0.isEmpty }
        )
    }

    public static func usesSerializedDecode(
        modelType: String,
        isVLM: Bool,
        overrides: Set<String> = serializationOverrides()
    ) -> Bool {
        serializationPolicy(modelType: modelType, isVLM: isVLM, overrides: overrides) != .never
    }

    /// How much batching a family may do.
    ///
    /// The scalar-offset families were treated as "never batch", which protects
    /// the image path by disabling batching for text chat as well. That is most
    /// of the cost: on gemma-4-E2B, serialising everything is 50x TTFT and 2.17x
    /// aggregate throughput at c4 (`docs/COMPETITIVE.md`), and its text rows are
    /// exact against serial and pass logit invariance more cleanly than the model
    /// the gate was pinned to. Its ragged IMAGE rows genuinely do break — all
    /// three rows stop at the same early token — so those keep the batch to
    /// themselves and nothing else does.
    ///
    /// `batchDecodeRegressedModelTypes` stays `.always`: what is left on that
    /// list (`qwen2`, `qwen3`) is a throughput claim, not a correctness one, and
    /// it has not been re-measured. `qwen3_moe` moved off it to
    /// `batchDecodeWidthCeilings` — its re-measurement produced a width, not a
    /// verdict. `qwen3_5` moved off too: "batched decode over its hybrid cache
    /// returns no tokens at all" was a symptom, not a cause. The cause was
    /// Qwen3.5's RoPE anchor being dropped when a second row joined, which
    /// trapped the process inside the model; with the anchor carried per row
    /// (``BatchPositionalState``) it is token-exact against serial at width 8.
    public static func serializationPolicy(
        modelType: String,
        isVLM: Bool,
        overrides: Set<String> = serializationOverrides()
    ) -> SerializationPolicy {
        let normalizedModelType = modelType.lowercased()
        if overrides.contains("all") || overrides.contains(normalizedModelType) {
            return .never
        }
        let ceiling = batchDecodeWidthCeilings[normalizedModelType]
        if batchDecodeRegressedModelTypes.contains(normalizedModelType) {
            return .always
        }
        if isVLM, scalarOffsetVLMModelTypes.contains(normalizedModelType) {
            guard multimodalOnlyModelTypes.contains(normalizedModelType) else { return .always }
            return .batched(maxWidth: ceiling, multimodalSolitude: true)
        }
        if let ceiling {
            return .maxWidth(ceiling)
        }
        return .never
    }

    /// Scalar-offset families whose TEXT rows are proven safe to batch, so only
    /// their multimodal rows are serialized. Entry here requires both gates:
    /// `BatchInvarianceTests.testBatchInvarianceAcrossModelFamilies` (logits) and
    /// `SchedulerEngineTests.testVLMBatchEqualityAcrossModels` (images). With
    /// `multimodalSolitude` the image gate is about a path that no longer
    /// batches, so what actually earns entry is the numeric one — stated here
    /// because the sentence above predates solitude and reads stricter than the
    /// policy it describes.
    ///
    /// `qwen3_5` (Qwen3.5-4B and Qwen3.8-27B) joined on 2026-08-23 with evidence
    /// stronger than the tolerance: `PositionalStateBatchIntegrationTests` holds
    /// it token-EXACT against serial at 4 ragged rows and again at width 8, on
    /// the `VLMModelFactory` path production actually loads. Its scalar offset is
    /// corrected per row by ``BatchPositionalState`` rather than tolerated.
    ///
    /// **`gemma4` and `gemma4_unified` were REMOVED on 2026-08-24.** They were
    /// granted entry on the morning of 2026-08-23 and it produced the biggest
    /// result on the board — gemma-4-E2B longgen c8 TTFT 39,346 -> 1,793 ms.
    /// The grant was wrong, and not by a tolerance: batched gemma decode
    /// produces **unrotated** queries and keys for every row but the first.
    ///
    /// MLX's RoPE has a fast path taken when the input is row-contiguous, the
    /// sequence length is 1, and the offset is a SCALAR
    /// (`mlx/backend/metal/rope.cpp:96`). That branch dispatches
    /// `MTL::Size(dims/2, N, 1)`, where `N` covers only the head axes — the
    /// batch size is absent (`rope.cpp:134-139`), while the general branch has
    /// it at `:149`. The kernel therefore rotates batch row 0 and, because the
    /// input is donated, rows 1..B-1 pass through untouched.
    ///
    /// Gemma's VLM attention is the only code in our stack that reaches that
    /// primitive with a scalar offset on a batched decode tensor
    /// (`Gemma4.swift:872,880,903`); every other family rotates through
    /// `applyRotaryPosition(..., offset: cache?.ropeOffset)`, and a per-row
    /// array offset forces the correct branch. `RoPEBatchGridProbeTests`
    /// reproduces it in under a second with no model:
    /// `rows-unrotated=[1, 2, 3]`, output bit-identical to input.
    ///
    /// So this was not a latency-versus-exactness trade — width >= 2 gemma was
    /// wrong. `.always` was a correctness floor, not conservatism, and it held
    /// until the RoPE call was fixed upstream or in a fork.
    ///
    /// **RE-GRANTED later on 2026-08-24: both defects are fixed, in forks.**
    /// The kernel: mlx-swift is pinned to atlas-open-sources/mlx-swift, whose
    /// vendored mlx carries the three-line backport of ml-explore/mlx
    /// `76a977ca` (#3498) — the fast-path grid gets its batch axis back
    /// (`RoPEBatchGridProbeTests`: `rows-unrotated=[]`). The anchor:
    /// mlx-swift-lm is pinned to atlas-open-sources/mlx-swift-lm `7b93094e`,
    /// where `Gemma4TextAttention` rotates with the per-row `cache?.ropeOffset`
    /// instead of the padded batch length. Evidence on the production factory
    /// (`GemmaRoPEOffsetProbeTests`, real chat-templated prompts, greedy):
    /// serial-exact at ragged width 2 and ragged width 4 — the arms the revert
    /// was made on. The residue is one late-step row flip on equal-length rows
    /// (unrotated gemma diverged at step 0 on 3 of 4 rows; this is the near-tie
    /// batch-variance every serving engine on non-invariant kernels shows, and
    /// the same category the logit sweep measured at `mismatched = 0`).
    private static let multimodalOnlyModelTypes: Set<String> = [
        "qwen3_5",
        "gemma4",
        "gemma4_unified"
    ]

}
