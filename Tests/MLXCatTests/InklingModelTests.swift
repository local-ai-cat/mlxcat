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
            serializationPolicy: .always, schedulerManagedTextPrefill: false, chunkIdlePrefill: true)
        let stream = try await engine.startChatCompletion(.init(
            model: "Inkling-2-Mini-test", messages: [.init(role: "user", content: "Hi")],
            maxTokens: 2, stream: true, enableThinking: false))
        XCTAssertEqual(stream.promptTokens, 6)
        var chunks: [OpenAIChatChunk] = []
        for try await chunk in stream.chunks { chunks.append(chunk) }
        XCTAssertEqual(chunks.count, 2)
        XCTAssertTrue(chunks.allSatisfy { (0..<8).contains($0.tokenID) && !$0.text.isEmpty })
        XCTAssertEqual(chunks.last?.finishReason, "length")
    }

    private struct TestTokenizer: MLXLMCommon.Tokenizer {
        let bosToken: String? = nil
        let eosToken: String? = nil
        let unknownToken: String? = nil

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 3, 2, 4, 5, 4] }
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

    private func makeModel() throws -> InklingLanguageModel {
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
                queryWeight: signal(64, width), keyWeight: signal(32, width), valueWeight: signal(32, width),
                relativeWeight: signal(8, width), outputWeight: signal(width, 64),
                gateWeight: signal(64, width), sinkWeight: signal(2, width),
                relativeProjection: signal(4, 4), queryScale: MLXArray.ones([2, 32]),
                relativeScale: MLXArray.ones([2, 4]), keyConvolution: convolution,
                valueConvolution: convolution)
            return InklingTextModel.Layer(
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
        }
        let model = try InklingTextModel(
            config: config, loop: config.makeLoop(), layers: layers,
            embeddings: signal(8, width), embeddingNorm: norm,
            messageQueries: signal(8, width).reshaped(2, 4, width).asType(.float32),
            finalQuery: signal(1, width).asType(.float32), finalNorm: norm, vocabulary: signal(8, width))
        return InklingLanguageModel(textModel: model, visionModel: nil, imageTokenID: 200054)
    }
}
