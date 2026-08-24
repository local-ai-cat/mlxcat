import MLX
import MLXLMCommon
import XCTest

@testable import MLXCat
@testable import MLXCatNative

/// `BatchKVCache` declares conformance to `BatchPositionedKVCache`, whose
/// extension supplies `ropeOffset` as `.batch(batchOffset)`. That is not enough.
///
/// `ropeOffset` is a requirement of **`KVCache`**, and `BatchPositionedKVCache`
/// refines `KVCache` without restating it. Swift therefore satisfies the
/// `KVCache` witness from `KVCache`'s own extension — `.scalar(offset)` — and the
/// refining extension is never reached through an existential. The model holds
/// `any KVCache`, so every row in a batch got one scalar RoPE position.
///
/// It compiles, it looks correct at the declaration site, and it silently
/// produces wrong tokens for every row whose left padding differs from the one
/// the scalar happens to describe. These tests pin the witness itself.
final class BatchRoPEOffsetTests: XCTestCase {

    private func offsets(_ offset: RoPEOffset?) -> [Int32]? {
        guard case .batch(let array)? = offset else { return nil }
        return array.asArray(Int32.self)
    }

    func testRopeOffsetIsPerRowThroughTheKVCacheExistential() {
        // Two rows, different left padding: true positions differ by the padding.
        let cache = BatchKVCache(leftPadding: [0, 3], idx: 10)
        let asProtocol: any KVCache = cache

        XCTAssertEqual(
            offsets(asProtocol.ropeOffset), [10, 7],
            "the model sees `any KVCache`; a scalar here is the batch decode bug"
        )
    }

    func testConcreteAndExistentialAgree() {
        let cache = BatchKVCache(leftPadding: [0, 2, 5], idx: 8)
        let viaExistential = offsets((cache as any KVCache).ropeOffset)
        let viaConcrete = offsets(cache.ropeOffset)
        XCTAssertEqual(viaExistential, viaConcrete)
        XCTAssertEqual(viaConcrete, [8, 6, 3])
    }

    func testBatchOffsetAndRopeOffsetDoNotDrift() {
        let cache = BatchKVCache(leftPadding: [1, 4], idx: 12)
        XCTAssertEqual(offsets((cache as any KVCache).ropeOffset), cache.batchOffset.asArray(Int32.self))
    }
}

/// The serialization exclusion lists decide whether batched decode runs at all,
/// and they cost five of six leaderboard models their concurrency. These pin the
/// override that lets the benchmark A/B them, and pin the default: OFF.
final class SerializationOverrideTests: XCTestCase {
    func testDefaultsAreUnchangedWithNoOverride() {
        let none: Set<String> = []
        XCTAssertTrue(NativeModelLoader.usesSerializedDecode(modelType: "gemma4", isVLM: true, overrides: none))
        XCTAssertTrue(NativeModelLoader.usesSerializedDecode(modelType: "qwen3_moe", isVLM: false, overrides: none))
        XCTAssertFalse(NativeModelLoader.usesSerializedDecode(modelType: "gpt_oss", isVLM: false, overrides: none))
    }

    func testNamedFamilyIsUnserialized() {
        let overrides: Set<String> = ["gemma4"]
        XCTAssertFalse(NativeModelLoader.usesSerializedDecode(modelType: "gemma4", isVLM: true, overrides: overrides))
        XCTAssertTrue(
            NativeModelLoader.usesSerializedDecode(modelType: "qwen3_5", isVLM: true, overrides: overrides),
            "an override must not leak to families it did not name")
    }

    func testAllLiftsEveryExclusion() {
        let overrides: Set<String> = ["all"]
        XCTAssertFalse(NativeModelLoader.usesSerializedDecode(modelType: "gemma4", isVLM: true, overrides: overrides))
        XCTAssertFalse(NativeModelLoader.usesSerializedDecode(modelType: "qwen3_moe", isVLM: false, overrides: overrides))
    }

    func testGemmaBatchesTextOnceBothRoPEDefectsAreFixedInForks() {
        // The 2026-08-23 grant was wrong (MLX's RoPE fast path dropped the
        // batch axis, so rows 1..B-1 decoded UNROTATED) and gemma fell back to
        // `.always` — a correctness floor "until the RoPE call is fixed
        // upstream or in a fork". On 2026-08-24 both defects were fixed in
        // forks (KNOWN-FAILURES 1f: atlas-open-sources/mlx-swift carries the
        // kernel backport of ml-explore/mlx 76a977ca; mlx-swift-lm 7b93094e
        // rotates with the per-row `cache?.ropeOffset`), and the grant was
        // restored on production-factory evidence: `GemmaRoPEOffsetProbeTests`
        // reads serial-exact at ragged widths 2 and 4 on real prompts, with
        // identical rows agreeing with each other and with serial.
        //
        // This pin means: whoever changes gemma's policy again must come back
        // through the probes, in either direction.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(
                modelType: "gemma4", isVLM: true, overrides: []),
            .multimodalOnly,
            "gemma4 text rows batch again — both RoPE defects are fixed in the forks")
        // gemma4_unified keeps its width-4 ceiling: that number came from a
        // logit sweep measured on the PRE-fix engine (its errors were almost
        // certainly this bug) and stays until the sweep is re-run.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(
                modelType: "gemma4_unified", isVLM: true, overrides: []),
            .batched(maxWidth: 4, multimodalSolitude: true),
            "gemma4_unified batches under its pre-fix width ceiling pending a re-sweep")

        // The measurement override still works, because pricing a policy is
        // how changes to it get justified.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(
                modelType: "gemma4", isVLM: true, overrides: ["gemma4"]),
            .never)
    }

    func testQwen35BatchesTextOnceItsRopeAnchorIsCarriedPerRow() {
        // "lifting qwen3_5 returns no tokens at all over a hybrid cache" was a
        // symptom. The cause was Qwen3.5's M-RoPE anchor being dropped when a
        // second row joined, which trapped the process inside the model itself
        // (`docs/KNOWN-FAILURES.md` §1d). With the anchor carried per row and
        // corrected for left padding, the family is token-EXACT against serial
        // at 4 ragged rows and at width 8 —
        // `PositionalStateBatchIntegrationTests`, on the VLMModelFactory path
        // production loads. That is a stronger result than the 1.25 logit
        // tolerance every other family is admitted on.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "qwen3_5", isVLM: true, overrides: []),
            .multimodalOnly,
            "qwen3_5 text rows batch; its image rows keep the batch to themselves")

        // Its sibling has no such evidence and must not ride along.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(
                modelType: "qwen3_5_moe", isVLM: true, overrides: []),
            .always)
    }

    func testFamiliesWithoutBothGatesStayFullySerialized() {
        let none: Set<String> = []
        // what is LEFT on the regression list is a throughput claim, not a
        // correctness one, and has not been re-measured — it stays blunt until
        // it is. qwen3_moe is no longer on it; see the width-ceiling tests below.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "qwen2", isVLM: false, overrides: none),
            .always)
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "qwen3", isVLM: false, overrides: none),
            .always)
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "gpt_oss", isVLM: false, overrides: none),
            .never)
    }

    func testMoEBatchesOnlyAtTheWidthItWasMeasuredAt() {
        // Raised 2 -> 8 on 2026-08-24. The 2.69-vs-1.25 logit reading at
        // widths 4/8 stands, but `MoEWidthNumericsProbeTests` rules out row
        // contamination (identical rows token-identical to serial at widths
        // 2/4/8) and `MoEBatchIntegrationTests` passes strict serial-greedy
        // token equality at all three widths — so the reading is expert-mix
        // accumulation order, not a defect. 8 is the widest width MEASURED;
        // widening further must come back through both gates.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "qwen3_moe", isVLM: false, overrides: []),
            .maxWidth(8))
        // The override still lifts it completely, so the benchmark can A/B the
        // ceiling against no ceiling at all.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(
                modelType: "qwen3_moe", isVLM: false, overrides: ["qwen3_moe"]),
            .never)
        // The legacy Bool view must keep reading "serializes to some degree",
        // because it does.
        XCTAssertTrue(
            NativeModelLoader.usesSerializedDecode(modelType: "qwen3_moe", isVLM: false, overrides: []))
    }

    func testAWidthCeilingAdmitsUpToItsWidthAndRefusesPastIt() {
        let rows = (0 ..< 3).map { index in
            Request(
                uid: "row-\(index)",
                input: .init(text: .init(tokens: MLXArray([Int32(1)]))),
                maxTokens: 4)
        }
        let policy = SerializationPolicy.maxWidth(2)
        // Nothing running: admit. One running: admit — this is the width the
        // measurement earned, and it is exactly what `.always` was throwing away.
        XCTAssertFalse(policy.refusesToJoin(running: []))
        XCTAssertFalse(policy.refusesToJoin(running: [rows[0]]))
        // At the ceiling, and past it, refuse.
        XCTAssertTrue(policy.refusesToJoin(running: [rows[0], rows[1]]))
        XCTAssertTrue(policy.refusesToJoin(running: rows))
        // A capped row never demands the batch to itself — the cap belongs to
        // the batch, not to the row.
        XCTAssertFalse(policy.requiresSolitude(rows[0]))
        // And the blunt policies are unchanged by any of this.
        XCTAssertTrue(SerializationPolicy.always.refusesToJoin(running: []))
        XCTAssertFalse(SerializationPolicy.never.refusesToJoin(running: rows))
    }

    func testATextOnlyGemmaDeploymentIsNotSerialized() {
        // isVLM is decided by the model directory, not the request. A gemma4
        // checkpoint always reports true, which is why "capability" was the wrong
        // axis to serialize on in the first place.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "gemma4", isVLM: false, overrides: []),
            .never)
    }

    func testMultimodalOnlyGivesImageRowsSolitudeAndBatchesText() {
        let text = Request(
            uid: "text", input: .init(text: .init(tokens: MLXArray([Int32(1)]))), maxTokens: 4)
        XCTAssertFalse(SerializationPolicy.multimodalOnly.requiresSolitude(text))
        XCTAssertFalse(SerializationPolicy.never.requiresSolitude(text))
        XCTAssertTrue(SerializationPolicy.always.requiresSolitude(text))
    }

    func testParsingIsWhitespaceAndCaseTolerant() {
        let parsed = NativeModelLoader.serializationOverrides(
            ["MLXCAT_UNSERIALIZE_MODEL_TYPES": " Gemma4 , QWEN3_5 ,, "])
        XCTAssertEqual(parsed, ["gemma4", "qwen3_5"])
        XCTAssertEqual(NativeModelLoader.serializationOverrides([:]), [])
    }
}

/// The idle-prefill chunking decision, which is a per-family MEASUREMENT and not
/// a preference — see `NativeModelLoader.chunksIdlePrefill` for the four-model
/// table it comes from.
final class IdlePrefillChunkingTests: XCTestCase {
    func testEveryFamilyChunksNowThatBuffersAreFreedBetweenChunks() {
        let none: [String: String] = [:]
        // qwen3_moe was the one family that peaked HIGHER chunked (+63%), which
        // is why an exemption list exists at all. Adding the per-chunk
        // Memory.clearCache() that mlx-lm has always had
        // (guest/mlx-lm/mlx_lm/generate.py:451) took it to 19.71 GiB — below
        // even its single-pass 24.75 — so the list is empty and nothing is
        // exempt. If this ever fails, a family regressed and the list is how to
        // park it while the cause is found.
        for modelType in ["gemma4_unified", "qwen3_5", "gpt_oss", "qwen3_moe", "muse_glimmer"] {
            XCTAssertTrue(
                NativeModelLoader.chunksIdlePrefill(modelType: modelType, environment: none),
                modelType)
        }
    }

    func testTheOverrideCanFlipAnyFamilyForABenchmarkArm() {
        // Naming a family forces it single-pass...
        XCTAssertFalse(
            NativeModelLoader.chunksIdlePrefill(
                modelType: "gpt_oss",
                environment: ["MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES": "gpt_oss"]))
        // ...and naming a different family leaves this one chunking, so a bench
        // arm can put exactly one family on the single-pass path.
        XCTAssertTrue(
            NativeModelLoader.chunksIdlePrefill(
                modelType: "qwen3_moe",
                environment: ["MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES": "gpt_oss"]))
        XCTAssertFalse(
            NativeModelLoader.chunksIdlePrefill(
                modelType: "gemma4",
                environment: ["MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES": "all"]))
    }

    func testParsingIsWhitespaceAndCaseTolerant() {
        XCTAssertFalse(
            NativeModelLoader.chunksIdlePrefill(
                modelType: "qwen3_moe",
                environment: ["MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES": " QWEN3_MOE , , "]))
    }
}
