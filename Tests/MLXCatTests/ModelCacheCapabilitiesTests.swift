import MLX
import MLXLMCommon
import MLXNN
import MLXCat
@testable import MLXCatNative
import XCTest

final class ModelCacheCapabilitiesTests: XCTestCase {
    func testExplicitWindowedKVCapabilityDisablesPrefixCache() async throws {
        let store = SessionPrefixKVStore()
        let engine = MLXCatEngine(
            model: FixedLogitPrefixModel(cacheFactory: Self.simpleCache),
            parameters: GenerateParameters(maxTokens: 1, temperature: 0),
            maxConcurrentRequests: 1,
            prefixStore: store,
            cacheCapabilities: ModelCacheCapabilities(usesWindowedKVCache: true)
        )

        _ = try await engine.generate([Self.request(uid: "windowed")])

        XCTAssertEqual(store.stats.storeCount, 0)
    }

    func testPrefixCacheDoesNotDisableFromCacheTypeName() async throws {
        let store = SessionPrefixKVStore()
        let engine = MLXCatEngine(
            model: FixedLogitPrefixModel(cacheFactory: { RotatingNameOnlyCache() }),
            parameters: GenerateParameters(maxTokens: 1, temperature: 0),
            maxConcurrentRequests: 1,
            prefixStore: store
        )

        _ = try await engine.generate([Self.request(uid: "name-only")])

        XCTAssertEqual(store.stats.storeCount, 1)
    }

    func testMaxKVSizeAloneDoesNotDisablePrefixCache() async throws {
        let store = SessionPrefixKVStore()
        let engine = MLXCatEngine(
            model: FixedLogitPrefixModel(cacheFactory: Self.simpleCache),
            parameters: GenerateParameters(maxTokens: 1, maxKVSize: 4, temperature: 0),
            maxConcurrentRequests: 1,
            prefixStore: store,
            cacheCapabilities: ModelCacheCapabilities(usesWindowedKVCache: false)
        )

        _ = try await engine.generate([Self.request(uid: "max-kv-size")])

        XCTAssertEqual(store.stats.storeCount, 1)
    }

    func testNativeLoaderDerivesWindowedKVCapabilityFromModelConfig() throws {
        let loader = NativeModelLoader(maxConcurrentRequests: 1)

        XCTAssertTrue(
            try loader.cacheCapabilities(in: Self.modelDirectory(configJSON: #"{"model_type":"gemma4"}"#))
                .usesWindowedKVCache
        )
        XCTAssertTrue(
            try loader.cacheCapabilities(in: Self.modelDirectory(configJSON: #"{"model_type":"gpt_oss","sliding_window":128}"#))
                .usesWindowedKVCache
        )
        XCTAssertFalse(
            try loader.cacheCapabilities(
                in: Self.modelDirectory(
                    configJSON: #"{"model_type":"qwen3","sliding_window":4096,"use_sliding_window":false}"#
                )
            ).usesWindowedKVCache
        )
    }

    func testNativeLoaderDerivesKVCacheProfileFromModelConfig() throws {
        let loader = NativeModelLoader(maxConcurrentRequests: 1)

        let capabilities = try loader.cacheCapabilities(
            in: Self.modelDirectory(
                configJSON: #"""
                {
                  "model_type": "qwen3",
                  "num_hidden_layers": 32,
                  "num_attention_heads": 32,
                  "num_key_value_heads": 8,
                  "head_dim": 128,
                  "torch_dtype": "bfloat16"
                }
                """#
            )
        )

        let profile = try XCTUnwrap(capabilities.kvCacheProfile)
        XCTAssertEqual(profile.bytesPerToken, 131_072)
        XCTAssertEqual(profile.estimatedBytes(promptTokens: 10, maxGeneratedTokens: 5), 1_966_080)
    }

    func testNativeLoaderDerivesKVCacheProfileFromNestedTextConfig() throws {
        let loader = NativeModelLoader(maxConcurrentRequests: 1)

        let capabilities = try loader.cacheCapabilities(
            in: Self.modelDirectory(
                configJSON: #"""
                {
                  "model_type": "qwen2_vl",
                  "text_config": {
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "hidden_size": 32,
                    "dtype": "float32"
                  }
                }
                """#
            )
        )

        let profile = try XCTUnwrap(capabilities.kvCacheProfile)
        XCTAssertEqual(profile.bytesPerToken, 256)
    }

    /// Pins the CURRENT policy, which is under review — see
    /// `BatchDecodeThroughputTests`. The families below are serialized because
    /// aggregate two-in-flight throughput measured *over HTTP* looked worse than
    /// serial. Measuring the decode loop directly contradicts that: Qwen3-0.6B
    /// (`qwen3`) scales 1.53x at batch 2 and 3.36x at batch 4, and the same
    /// requests through `MLXCatEngine` scale 2.11x. The loss is above the
    /// engine, so this allowlist is expected to be REVERTED rather than extended
    /// once the serving-path defect is fixed. Do not add families to it on the
    /// strength of an HTTP measurement alone.
    func testNativeLoaderSerializesMeasuredBatchDecodeRegressions() {
        for modelType in ["qwen2", "qwen3", "qwen3_moe"] {
            XCTAssertTrue(
                NativeModelLoader.usesSerializedDecode(modelType: modelType, isVLM: false),
                modelType
            )
        }

        // These are the literal `model_type` strings from the config.json of the
        // models the campaign actually benchmarked. "gemma4" alone was not one of
        // them: gemma-4-12B-it-qat-4bit reports `gemma4_unified` (the e2b/e4b
        // variants are the ones that report `gemma4`), so the original assertion
        // never covered the Gemma the evidence came from.
        for modelType in ["gemma4", "gemma4_unified", "gpt_oss", "qwen3_5"] {
            XCTAssertFalse(
                NativeModelLoader.usesSerializedDecode(modelType: modelType, isVLM: false),
                modelType
            )
        }

        // Unmeasured families keep batching — the policy is opt-in, so anything
        // new (e.g. muse_glimmer) defaults to batched rather than serialized.
        XCTAssertFalse(
            NativeModelLoader.usesSerializedDecode(modelType: "muse_glimmer", isVLM: false)
        )

        // The VLM scalar-offset rule is independent of the batch-decode allowlist
        // and must survive its removal.
        XCTAssertTrue(NativeModelLoader.usesSerializedDecode(modelType: "qwen3_vl", isVLM: true))
    }

    private static func request(uid: String) -> Request {
        Request(
            uid: uid,
            input: LMInput(text: LMInput.Text(tokens: MLXArray([Int32(1), 2, 3]))),
            maxTokens: 1,
            sampling: SamplingParameters(temperature: 0)
        )
    }

    private static func simpleCache() -> KVCacheSimple {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray.zeros([1, 1, 1, 1]),
            MLXArray.zeros([1, 1, 1, 1]),
        ]
        return cache
    }

    private static func modelDirectory(configJSON: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "mlxcat-model-capabilities-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(configJSON.utf8).write(to: directory.appending(component: "config.json"))
        return directory
    }
}

private final class FixedLogitPrefixModel: Module, LanguageModel {
    private let cacheFactory: () -> any KVCache

    init(cacheFactory: @escaping () -> any KVCache) {
        self.cacheFactory = cacheFactory
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        LMOutput(logits: MLXArray.zeros([input.tokens.dim(0), 1, 8]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [cacheFactory()]
    }
}

private final class RotatingNameOnlyCache: KVCache {
    var state: [MLXArray] = [
        MLXArray.zeros([1, 1, 1, 1]),
        MLXArray.zeros([1, 1, 1, 1]),
    ]
    var metaState: [String] = []
    var offset: Int { 0 }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] {
        state
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        state = [newKeys, newValues]
        return (newKeys, newValues)
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        0
    }

    func copy() -> any KVCache {
        let copy = RotatingNameOnlyCache()
        copy.state = state.map { $0[.ellipsis] }
        copy.metaState = metaState
        return copy
    }
}
