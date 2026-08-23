import MLX
import MLXNN
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
@testable import MLXCat
@testable import MLXCatNative
import Tokenizers
import XCTest

/// The policy, its exclusions, and its gate — no weights, runs everywhere.
final class KVQuantizationPolicyTests: XCTestCase {

    func testOffIsTheDefaultAndTheReferenceDefaults() {
        // Nobody ships this on. mlx-lm's --kv-bits defaults to None with group
        // size 64 and --quantized-kv-start 5000; omlx defaults its own scheme
        // off; mlx-serve has no KV quantization at all.
        XCTAssertFalse(KVQuantizationPolicy.off.isEnabled)
        XCTAssertNil(KVQuantizationPolicy().bits)
        XCTAssertEqual(KVQuantizationPolicy().groupSize, 64)
        XCTAssertEqual(KVQuantizationPolicy().startTokens, 5000)
        XCTAssertEqual(KVQuantizationPolicy.fromEnvironment([:]), .off)
    }

    func testOnlyFourAndEightBitAreAccepted() {
        // A typo must leave the cache in fp16 rather than pick some other width.
        XCTAssertEqual(KVQuantizationPolicy.fromEnvironment(["MLXCAT_KV_BITS": "4"]).bits, 4)
        XCTAssertEqual(KVQuantizationPolicy.fromEnvironment(["MLXCAT_KV_BITS": "8"]).bits, 8)
        for bad in ["3", "16", "0", "-4", "yes", ""] {
            XCTAssertNil(
                KVQuantizationPolicy.fromEnvironment(["MLXCAT_KV_BITS": bad]).bits,
                "MLXCAT_KV_BITS=\(bad) must not enable quantization")
        }
    }

    func testGroupSizeAndStartAreIndependentlyOverridable() {
        let policy = KVQuantizationPolicy.fromEnvironment([
            "MLXCAT_KV_BITS": "4",
            "MLXCAT_KV_GROUP_SIZE": "32",
            "MLXCAT_QUANTIZED_KV_START": "0",
        ])
        XCTAssertEqual(policy, KVQuantizationPolicy(bits: 4, groupSize: 32, startTokens: 0))

        // Set without bits, they change nothing that matters.
        XCTAssertFalse(
            KVQuantizationPolicy.fromEnvironment(["MLXCAT_QUANTIZED_KV_START": "0"]).isEnabled)
    }

    func testAttentionSinkModelsAreExcluded() {
        // gpt-oss's quantized attention route does not pass `sinks:`, and
        // `quantizedScaledDotProductAttention` has no parameter to pass them
        // to — so quantizing it would silently drop them. That is a correctness
        // change wearing quantization's clothes, and it is refused rather than
        // documented.
        let asked = KVQuantizationPolicy(bits: 4, startTokens: 0)
        XCTAssertEqual(KVQuantizationPolicy.resolved(asked, modelType: "gpt_oss"), .off)
        XCTAssertEqual(KVQuantizationPolicy.resolved(asked, modelType: "GPT_OSS"), .off)
        // Everything else keeps what it asked for.
        for allowed in ["qwen3_moe", "qwen3_5", "gemma4", "gemma4_unified", "qwen3"] {
            XCTAssertEqual(
                KVQuantizationPolicy.resolved(asked, modelType: allowed), asked, allowed)
        }
        // An unknown model type is not silently excluded — only the named list is.
        XCTAssertEqual(KVQuantizationPolicy.resolved(asked, modelType: nil), asked)
    }
}

/// The single-stream gate, asserted through BEHAVIOUR rather than a test-only
/// accessor — the same discipline `SchedulerAdmissionShapeTests` uses.
///
/// It matters because a quantized row that reaches `BatchLayerCache` is
/// misrouted by its state array count: a quantized layer's state is 4 or 6
/// arrays, not 2. Rows of different lengths throw; two rows of the SAME length
/// pass shape validation and then have `[step, offset, groupSize, bits]` read as
/// slot metadata. The second case is silent wrong output, which is why this is a
/// gate and not a warning.
final class KVQuantizationGateTests: XCTestCase {

    private func request(_ uid: String, promptTokens: Int) -> Request {
        Request(
            uid: uid,
            input: LMInput(text: .init(tokens: MLXArray(Array(repeating: Int32(7), count: promptTokens)))),
            maxTokens: 4,
            sampling: SamplingParameters(temperature: 0)
        )
    }

    private func admittedInOneStep(kvBits: Int?, policy: SerializationPolicy) async throws -> Int {
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(QuantGateModel(vocabularySize: 32)),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0),
            maxConcurrentRequests: 8,
            serializationPolicy: policy,
            kvQuantization: KVQuantizationPolicy(bits: kvBits, startTokens: 0)
        )
        for index in 0 ..< 8 {
            try await scheduler.submit(request("r\(index)", promptTokens: 12))
        }
        // Each admitted request emits its first token in the step that admits
        // it, so the responses ARE the admission count.
        return try await scheduler.step().filter { $0.token >= 0 }.count
    }

    func testQuantizationForcesEveryRowToRunAlone() async throws {
        let quantized = try await admittedInOneStep(kvBits: 4, policy: .never)
        XCTAssertEqual(
            quantized, 1,
            "a quantized row must never be able to join a batch — 8 were admitted at once")
    }

    func testTheGateIsOffWhenQuantizationIsOff() async throws {
        // The gate must not quietly serialize an engine that asked for nothing.
        let unquantized = try await admittedInOneStep(kvBits: nil, policy: .never)
        XCTAssertEqual(
            unquantized, 8,
            "batching regressed for everyone, not just for quantized engines")
    }

    func testPrefixCachingIsTurnedOffRatherThanLeftToThrow() async throws {
        // Both prefix stores require the 2-array keys/values schema and would
        // throw on every publish. The scheduler catches that, so nothing
        // crashes — the cache just silently stops working while filling the log.
        // It is disabled explicitly instead, and this is how you can tell.
        func storeCount(kvBits: Int?) async throws -> Int {
            let store = SessionPrefixKVStore(maxBytes: 1 << 22)
            let scheduler = Scheduler(
                modelBox: LanguageModelBox(QuantGateModel(vocabularySize: 32)),
                parameters: GenerateParameters(maxTokens: 2, temperature: 0),
                maxConcurrentRequests: 1,
                prefixStore: store,
                kvQuantization: KVQuantizationPolicy(bits: kvBits, startTokens: 0)
            )
            try await scheduler.submit(request("only", promptTokens: 24))
            for _ in 0 ..< 8 { _ = try await scheduler.step() }
            return store.stats.storeCount
        }

        // The control has to actually store something, or the treatment proves
        // nothing.
        let unquantized = try await storeCount(kvBits: nil)
        XCTAssertGreaterThan(unquantized, 0, "the control never published a prefix")
        let quantized = try await storeCount(kvBits: 8)
        XCTAssertEqual(quantized, 0, "a quantized engine still published prefixes")
    }
}

/// Minimal `LanguageModel` that grows a plain `KVCacheSimple`, so admission
/// shape and prefix publishing are observable with no weights and no Metal.
private final class QuantGateModel: Module, LanguageModel {
    private let vocabularySize: Int

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        let batch = input.tokens.dim(0)
        let sequence = input.tokens.shape.count > 1 ? input.tokens.dim(1) : 1
        cache?.forEach { layerCache in
            guard let simple = layerCache as? KVCacheSimple else { return }
            let cached = simple.state.first?.dim(2) ?? 0
            let total = cached + max(1, sequence)
            simple.state = [
                MLXArray.zeros([batch, 1, total, 1]),
                MLXArray.zeros([batch, 1, total, 1]),
            ]
        }
        return LMOutput(logits: MLXArray.zeros([batch, 1, vocabularySize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }
}

/// The claim, measured. Everything above is policy and gating; this is the only
/// test that shows the KV cache actually gets smaller.
///
/// The bar is deliberately below the arithmetic. Affine group-64 stores an fp16
/// scale and bias per group, so 4-bit is 0.5625 B/element against fp16's 2 —
/// a 3.56x reduction of the FULL-ATTENTION layers only. Qwen3.5 converts 8 of
/// its 32 layers (the other 24 are linear-attention `MambaCache`, which upstream
/// declines to quantize), and the measurement also carries weights and
/// activations that do not shrink at all. So this asserts a clear directional
/// drop, not a ratio it would have to keep re-deriving.
///
///     MLXCAT_KV_QUANT_TEST_MODEL=~/Library/Caches/models/mlx-community/Qwen3.5-4B-MLX-4bit \
///       swift test --filter KVQuantizationMemoryTests
final class KVQuantizationMemoryTests: XCTestCase {

    /// Long enough that the KV dominates the difference between the arms.
    private static let promptTokens = 16384

    private static func residentBytesAfterPrefill(
        model: any LanguageModel, policy: KVQuantizationPolicy
    ) async throws -> Int {
        Memory.clearCache()
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            maxConcurrentRequests: 1,
            kvQuantization: policy
        )
        let baseline = Memory.activeMemory
        try await scheduler.submit(
            Request(
                uid: "kvq",
                input: LMInput(
                    text: .init(
                        tokens: MLXArray((0 ..< promptTokens).map { Int32(1000 + ($0 % 2048)) }))),
                maxTokens: 8,
                sampling: SamplingParameters(temperature: 0)
            )
        )
        // Step until the prompt has been prefilled — the first real token is
        // emitted by the step that finishes admission.
        var emitted = false
        for _ in 0 ..< 400 where !emitted {
            emitted = try await scheduler.step().contains { $0.token >= 0 }
        }
        XCTAssertTrue(emitted, "prefill never completed, so nothing was measured")
        let resident = Memory.activeMemory - baseline
        return resident
    }

    /// What quantization COSTS. Stage 1 shipped with the price documented as
    /// "real and unmeasured", which is a promissory note, not a number.
    ///
    /// Quantized attention is an unfused `quantizedMatmul` + softmax rather than
    /// a fused SDPA kernel, and `qwen3_5` additionally loses its compiled decode
    /// route (`Qwen35.swift:707-712` gates that on a plain attention path). So a
    /// decode slowdown is expected; the question is how much, and the answer
    /// belongs on the board rather than in a caveat.
    ///
    /// The assertion is a catastrophe guard, not a target — the printed ratio is
    /// the result, and a bar tight enough to be a target would just encode
    /// today's hardware.
    private static func decodeRate(model: any LanguageModel, policy: KVQuantizationPolicy)
        async throws -> Double
    {
        Memory.clearCache()
        let generated = 64
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: GenerateParameters(maxTokens: generated, temperature: 0),
            maxConcurrentRequests: 1,
            kvQuantization: policy
        )
        try await scheduler.submit(
            Request(
                uid: "kvq-rate",
                input: LMInput(
                    text: .init(
                        tokens: MLXArray((0 ..< promptTokens).map { Int32(1000 + ($0 % 2048)) }))),
                maxTokens: generated,
                sampling: SamplingParameters(temperature: 0)
            )
        )

        // Prefill first, and do NOT time it — the prefill cost differs between
        // the arms for its own reasons (conversion allocates), and mixing it in
        // would report a blend of two different questions.
        var emitted = false
        for _ in 0 ..< 400 where !emitted {
            emitted = try await scheduler.step().contains { $0.token >= 0 }
        }
        XCTAssertTrue(emitted, "prefill never completed")

        let start = Date()
        var decoded = 0
        for _ in 0 ..< (generated * 4) where decoded < generated - 1 {
            decoded += try await scheduler.step().filter { $0.token >= 0 }.count
        }
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertGreaterThan(decoded, 0, "no tokens were decoded, so nothing was timed")
        return Double(decoded) / elapsed
    }

    func testTheThroughputPriceOfQuantizedAttentionIsMeasured() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let path = ProcessInfo.processInfo.environment["MLXCAT_KV_QUANT_TEST_MODEL"] else {
            throw XCTSkip("Set MLXCAT_KV_QUANT_TEST_MODEL to run the KV-quantization A/B.")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: url, using: #huggingFaceTokenizerLoader())

        // A/B/B/A. Running each arm once in a fixed order cannot tell a real
        // difference from an ordering artifact — the second arm inherits a warm
        // allocator and warm kernels. On Qwen3-Coder-30B the fixed-order run
        // reported quantized decode as FASTER, which is surprising enough to be
        // worth disbelieving until the reversed order agrees.
        let kv4 = KVQuantizationPolicy(bits: 4, groupSize: 64, startTokens: 0)
        let (forwardFP16, forwardKV4, reverseKV4, reverseFP16) =
            try await container.perform { context in
                let a = try await Self.decodeRate(model: context.model, policy: .off)
                let b = try await Self.decodeRate(model: context.model, policy: kv4)
                let c = try await Self.decodeRate(model: context.model, policy: kv4)
                let d = try await Self.decodeRate(model: context.model, policy: .off)
                return (a, b, c, d)
            }
        let fp16 = (forwardFP16 + reverseFP16) / 2
        let quantized = (forwardKV4 + reverseKV4) / 2

        print(
            String(
                format: "KVQUANTRATE prompt=%d fp16=%.1f tok/s kv4=%.1f tok/s ratio=%.2fx"
                    + " (forward %.1f/%.1f = %.2fx, reverse %.1f/%.1f = %.2fx)",
                Self.promptTokens, fp16, quantized, quantized / fp16,
                forwardFP16, forwardKV4, forwardKV4 / forwardFP16,
                reverseFP16, reverseKV4, reverseKV4 / reverseFP16))

        XCTAssertGreaterThan(fp16, 0)
        // Both orders must agree on the SIGN of the effect, or the number is an
        // ordering artifact and must not be quoted at all.
        XCTAssertEqual(
            forwardKV4 > forwardFP16, reverseKV4 > reverseFP16,
            "the two orders disagree on whether quantized decode is faster"
                + " (forward \(forwardFP16) -> \(forwardKV4),"
                + " reverse \(reverseFP16) -> \(reverseKV4));"
                + " this measurement is an ordering artifact, not a result")
        XCTAssertGreaterThan(
            quantized, fp16 / 4,
            "quantized decode is more than 4x slower than fp16 — that is a defect, not a price")
    }

    func testFourBitKVIsMateriallySmallerThanFP16() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let path = ProcessInfo.processInfo.environment["MLXCAT_KV_QUANT_TEST_MODEL"] else {
            throw XCTSkip("Set MLXCAT_KV_QUANT_TEST_MODEL to run the KV-quantization A/B.")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: url, using: #huggingFaceTokenizerLoader())

        let (fp16, quantized) = try await container.perform { context in
            let fp16 = try await Self.residentBytesAfterPrefill(
                model: context.model, policy: .off)
            let quantized = try await Self.residentBytesAfterPrefill(
                model: context.model,
                policy: KVQuantizationPolicy(bits: 4, groupSize: 64, startTokens: 0))
            return (fp16, quantized)
        }

        // Printed unconditionally: the number is the result.
        print(
            "KVQUANT prompt=\(Self.promptTokens) fp16=\(fp16 / 1_048_576) MiB "
                + "kv4=\(quantized / 1_048_576) MiB saved=\((fp16 - quantized) / 1_048_576) MiB")

        // The control has to hold something, or the treatment means nothing.
        XCTAssertGreaterThan(
            fp16, 64 * 1_048_576,
            "the fp16 arm held almost nothing — this prompt is too short to measure anything")
        XCTAssertLessThan(
            quantized, fp16,
            "quantizing the KV cache did not reduce resident memory")
    }
}
