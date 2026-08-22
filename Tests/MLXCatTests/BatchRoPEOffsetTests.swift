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

    func testParsingIsWhitespaceAndCaseTolerant() {
        let parsed = NativeModelLoader.serializationOverrides(
            ["MLXCAT_UNSERIALIZE_MODEL_TYPES": " Gemma4 , QWEN3_5 ,, "])
        XCTAssertEqual(parsed, ["gemma4", "qwen3_5"])
        XCTAssertEqual(NativeModelLoader.serializationOverrides([:]), [])
    }
}
