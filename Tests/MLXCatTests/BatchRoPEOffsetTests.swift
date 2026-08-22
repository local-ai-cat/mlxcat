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

    func testGemmaBatchesTextAndSerializesImages() {
        // The whole point of the narrower policy: gemma-4's text rows batch (50x
        // TTFT at c4) while its ragged image rows keep the batch to themselves.
        let none: Set<String> = []
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "gemma4", isVLM: true, overrides: none),
            .multimodalOnly)
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "gemma4_unified", isVLM: true, overrides: none),
            .multimodalOnly)
    }

    func testFamiliesWithoutBothGatesStayFullySerialized() {
        let none: Set<String> = []
        // qwen3_5: lifting it was MEASURED and returns no tokens at all over a
        // hybrid cache, so it is not a candidate for the narrower policy.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "qwen3_5", isVLM: true, overrides: none),
            .always)
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
        // qwen3_moe measures max logit error 0.94 at width 2 (inside the 1.25
        // tolerance) and 2.69 at widths 4 and 8. `.always` refused both; the
        // ceiling refuses only the widths that failed.
        XCTAssertEqual(
            NativeModelLoader.serializationPolicy(modelType: "qwen3_moe", isVLM: false, overrides: []),
            .maxWidth(2))
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
