import Foundation
import MLX
import MLXCat
import MLXCatHTTP
import MLXLMCommon
import XCTest

@testable import MLXCatNative

final class InklingModelTests: XCTestCase {
    func testLoopedDecoderPrefillAndContinuationKeepIndependentVisitState() throws {
        try MLXMetalRuntime.requireAvailable()
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 3, 2, 4, 5, 4])
        let wholeCache = try model.newCache(parameters: nil)
        let whole = model(tokens, cache: wholeCache)[0, -1, 0...].asType(.float32)
        XCTAssertEqual(whole.shape, [8])
        XCTAssertEqual(wholeCache.count, 8)
        XCTAssertTrue(wholeCache.allSatisfy { $0.offset == 6 && $0.state.count == 6 })

        let chunkCache = try model.newCache(parameters: nil)
        let input = LMInput(tokens: tokens[0..<5])
        let prepared = try model.prepare(input, cache: chunkCache, state: nil, prefill: .init(stepSize: 2))
        guard case .logits = prepared else { return XCTFail("Expected prefill logits") }
        let continued = model(tokens[5...], cache: chunkCache)[0, -1, 0...].asType(.float32)
        XCTAssertTrue(chunkCache.allSatisfy { $0.offset == 6 && $0.state.count == 6 })
        XCTAssertLessThanOrEqual(abs(whole - continued).max().item(Float.self), 0.02)

        let copy = chunkCache[1].copy()
        _ = model(MLXArray([Int32(2)]), cache: chunkCache)
        XCTAssertEqual(copy.offset, 6)
        XCTAssertEqual(chunkCache[1].offset, 7)
        XCTAssertEqual(copy.state[0].dim(2), 3)
    }

    func testCompiledExpertsPreserveSingletonLogitsAcrossLoopVisits() throws {
        try MLXMetalRuntime.requireAvailable()
        let plain = try makeModel()
        let compiled = try makeModel(compileExperts: true)
        let plainCache = try plain.newCache(parameters: nil)
        let compiledCache = try compiled.newCache(parameters: nil)
        let prefix = MLXArray([Int32(1), 3, 2, 4])
        eval(plain(prefix, cache: plainCache), compiled(prefix, cache: compiledCache))
        for id: Int32 in [5, 4, 1, 6] {
            let token = MLXArray([id])
            XCTAssertEqual(
                plain(token, cache: plainCache).asArray(Float.self),
                compiled(token, cache: compiledCache).asArray(Float.self))
        }
    }

    func testNativeChatStreamsTokensFromTheLoopedModel() async throws {
        try MLXMetalRuntime.requireAvailable()
        let model = try makeModel()
        let tokenizer = TestTokenizer()
        let context = ModelContext(
            configuration: .init(directory: FileManager.default.temporaryDirectory),
            model: model, processor: InklingInputProcessor(
                tokenizer: tokenizer, imageTokenID: 200054, imagePatchSize: nil),
            tokenizer: tokenizer)
        let engine = NativeModelEngine(
            context: context, modelID: "Inkling-2-Mini-test", maxContextTokens: 32,
            maxConcurrentRequests: 2,
            cacheCapabilities: .init(usesWindowedKVCache: true),
            serializationPolicy: .always, schedulerManagedTextPrefill: true, chunkIdlePrefill: true,
            windowedLayerDecoder: InklingVisitCache.restore, prefill: .init(stepSize: 2))
        func generate(_ messages: [OpenAIChatMessage], session: String) async throws
            -> (count: Int, chunks: [OpenAIChatChunk])
        {
            let stream = try await engine.startChatCompletion(.init(
                model: "Inkling-2-Mini-test", messages: messages,
                maxTokens: 2, stream: true, enableThinking: false, cacheSession: session))
            var chunks: [OpenAIChatChunk] = []
            for try await chunk in stream.chunks { chunks.append(chunk) }
            return (stream.promptTokens, chunks)
        }

        let initial: [OpenAIChatMessage] = [.init(role: "user", content: "Hi")]
        let cold = try await generate(initial, session: "conversation")
        XCTAssertEqual(cold.count, 6)
        XCTAssertEqual(cold.chunks.count, 2)
        XCTAssertTrue(cold.chunks.allSatisfy { (0..<8).contains($0.tokenID) && !$0.text.isEmpty })
        XCTAssertEqual(cold.chunks.last?.finishReason, "length")
        XCTAssertEqual(cold.chunks.first?.cachedPromptTokens, 0)

        let next = initial + [
            .init(role: "assistant", content: cold.chunks.map(\.text).joined()),
            .init(role: "user", content: "Follow up"),
        ]
        let warm = try await generate(next, session: "conversation")
        let uncached = try await generate(next, session: "separate")
        XCTAssertEqual(warm.count, 9)
        XCTAssertEqual(warm.chunks.map(\.tokenID), uncached.chunks.map(\.tokenID))
        XCTAssertEqual(warm.chunks.first?.cachedPromptTokens, 4)
        XCTAssertEqual(uncached.chunks.first?.cachedPromptTokens, 0)
        XCTAssertEqual(engine.prefixCacheStats().fetchHitCount, 1)
    }

    private struct TestTokenizer: MLXLMCommon.Tokenizer {
        let bosToken: String? = nil
        let eosToken: String? = nil
        let unknownToken: String? = nil

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            let initial = [1, 3, 2, 4, 5, 4]
            return text.contains("Follow up") ? initial + [5, 3, 2] : initial
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined()
        }
        func convertTokenToId(_ token: String) -> Int? { Int(token) }
        func convertIdToToken(_ id: Int) -> String? { String(id) }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { encode(text: "", addSpecialTokens: false) }
    }

    private func makeModel(compileExperts: Bool = false) throws -> InklingLanguageModel {
        let json = """
            {"hidden_size":64,"num_hidden_layers":4,"n_layers_per_attn_res_block":1,
             "rms_norm_eps":0.000001,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
             "d_rel":4,"qk_l2_clip_threshold":22,"rel_l2_clip_threshold":8,
             "log_scaling_n_floor":4,"log_scaling_alpha":0.1,"attn_res_normalize_values":true,
             "num_experts_per_tok":2,"route_scale":2,"mlp_up_softcap":4,"mlp_gate_softcap":4,
             "final_logit_softcapping":30,"unpadded_vocab_size":8,"local_layer_ids":[0,1,2],
             "sliding_window_size":4,"sconv_kernel_size":2,
             "loop_config":{"core_start":1,"core_end":3,"num_passes":3,"gate_rank":2}}
            """
        let config = try JSONDecoder().decode(InklingConfiguration.Text.self, from: Data(json.utf8))
        let width = config.hiddenSize
        func signal(_ rows: Int, _ columns: Int) -> MLXArray {
            (sin(arange(rows * columns).asType(.float32)) * 0.02)
                .reshaped(rows, columns).asType(.bfloat16)
        }
        let norm = MLXArray.ones([width], dtype: .bfloat16)
        let packed = quantized(
            stacked([signal(width, width), signal(width, width)], axis: 0),
            groupSize: 32, bits: 8, mode: .mxfp8)
        let projection = InklingExperts.Projection(weight: packed.wq, scales: packed.scales, format: .mxfp8)
        let controller = InklingTextModel.Controller(
            down: signal(2, width * 2), forget: signal(width, 2), gain: signal(width, 2))
        let layers = (0..<config.layerCount).map { index in
            let local = config.localLayerIDs.contains(index)
            let convolution: InklingDecoderAttention.ShortWeights = local
                ? .fixed(MLXArray.zeros([32, 2, 1], dtype: .bfloat16))
                : .dynamic(staticWeight: signal(2, 32), projection: signal(2, width))
            let attention = InklingDecoderAttention(
                config: config, window: local ? 4 : nil,
                queryWeight: .dense(signal(64, width)), keyWeight: .dense(signal(32, width)), valueWeight: .dense(signal(32, width)),
                relativeWeight: .dense(signal(8, width)), outputWeight: .dense(signal(width, 64)),
                gateWeight: .dense(signal(64, width)), sinkWeight: .dense(signal(2, width)),
                relativeProjection: signal(4, 4), queryScale: MLXArray.ones([2, 32]),
                relativeScale: MLXArray.ones([2, 4]), keyConvolution: convolution,
                valueConvolution: convolution)
            var layer = InklingTextModel.Layer(
                attention: attention,
                experts: .init(router: signal(2, width), routingBias: MLXArray.zeros([2]),
                               up: projection, gate: projection, down: projection,
                               topK: 2, routeScale: 2, upCap: 4, gateCap: 4),
                inputNorms: local ? [norm] : [norm, norm, norm], postAttentionNorm: norm,
                attentionQuery: index == 0 ? nil : signal(local ? 1 : 3, width).asType(.float32),
                mlpQuery: signal(1, width).asType(.float32),
                attentionConvolution: MLXArray.zeros([width, 2, 1], dtype: .bfloat16),
                mlpConvolution: MLXArray.zeros([width, 2, 1], dtype: .bfloat16),
                controllers: [controller, controller, controller, controller])
            if compileExperts {
                let experts = layer.experts
                layer.compiledExperts = compile { (input: MLXArray) in experts(input) }
            }
            return layer
        }
        let model = try InklingTextModel(
            config: config, loop: config.makeLoop(), layers: layers,
            embeddings: .dense(signal(8, width)), embeddingNorm: norm,
            messageQueries: signal(8, width).reshaped(2, 4, width).asType(.float32),
            finalQuery: signal(1, width).asType(.float32), finalNorm: norm,
            vocabulary: .dense(signal(8, width).asType(.float32)))
        return InklingLanguageModel(textModel: model, visionModel: nil, imageTokenID: 200054)
    }
}

extension InklingModelTests {
    func testMixedStorageVocabularyGEMVPreservesFP32LaneOrder() throws {
        try MLXMetalRuntime.requireAvailable()
        let rows = 4_096
        let width = 384
        let x = cos(arange(width).asType(.float32)).asType(.bfloat16)
            .asType(.float32).reshaped(1, 1, width)
        let weight = sin(arange(rows * width).asType(.float32))
            .reshaped(rows, width).asType(.bfloat16)
        let expected = matmul(x, weight.asType(.float32).transposed())
        let actual = InklingVocabularyGEMV.run(x, weight: weight)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testVocabularyProjectionAB() throws {
        guard ProcessInfo.processInfo.environment["MLXCAT_INKLING_VOCAB_AB"] == "1" else {
            throw XCTSkip("Set MLXCAT_INKLING_VOCAB_AB=1 under an exclusive machine lease.")
        }
        try MLXMetalRuntime.requireAvailable()
        let weight = MLXArray.ones([201_024, 2_304], dtype: .bfloat16)
        let input = sin(arange(2_304).asType(.float32)).reshaped(1, 1, 2_304)
        eval(weight, input)
        let iterations = 12
        for cached in [false, true, true, false] {
            let projection = cached ? weight.asType(.float32) : weight
            eval(matmul(input, projection.asType(.float32).transposed()))
            Memory.clearCache()
            let retained = Memory.activeMemory
            Memory.peakMemory = 0
            let start = Date()
            for _ in 0..<iterations {
                eval(matmul(input, projection.asType(.float32).transposed()))
            }
            let elapsed = Date().timeIntervalSince(start) * 1000 / Double(iterations)
            print("Inkling vocabulary cached=\(cached): \(elapsed) ms, retained \(retained) bytes, peak \(Memory.peakMemory) bytes")
        }
    }
}
