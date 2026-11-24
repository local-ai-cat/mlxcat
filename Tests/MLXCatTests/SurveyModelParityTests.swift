import Foundation
import MLX
import MLXCatNative
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SurveyModelParityTests: XCTestCase {
    private struct Case: Decodable {
        struct Sample: Decodable {
            let ids: [Int]
            let scores: [Float]
        }
        let name: String
        let messages: [[String: MLXLMCommon.JSONValue]]
        let tools: [[String: MLXLMCommon.JSONValue]]?
        let thinking: Bool
        let prompt: [Int]
        let generated: [Int]
        let sampled: [Sample]
        let prefillChunking: String?
    }

    func testSurveyModelMatchesPython() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let directory = env["MLXCAT_SURVEY_MODEL"], let fixtures = env["MLXCAT_SURVEY_REFERENCE"] else {
            throw XCTSkip("Set MLXCAT_SURVEY_MODEL and MLXCAT_SURVEY_REFERENCE to verified local data.")
        }
        if let metallib = env["MLXCAT_SURVEY_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: metallib)
        }
        try MLXMetalRuntime.requireAvailable()
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: fixtures)))
        XCTAssertEqual(cases.count, 11)
        let model = try await VLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: directory), using: NativeTokenizerLoader())
        try await model.perform { context in
            for fixture in cases {
                if fixture.name == "code",
                   let qwen = context.model as? Qwen35 {
                    let trace = URL(fileURLWithPath: fixtures)
                        .deletingPathExtension().appendingPathExtension("safetensors")
                    if FileManager.default.fileExists(atPath: trace.path) {
                        try Self.trace(qwen, fixture: fixture, reference: loadArrays(url: trace))
                    }
                }
                let messages = fixture.messages.map { $0.mapValues(Self.sendableValue) }
                let tools = fixture.tools?.map { $0.mapValues(Self.sendableValue) }
                let input: LMInput?
                do {
                    input = try await context.processor.prepare(input: UserInput(
                        messages: messages, tools: tools, additionalContext: ["enable_thinking": fixture.thinking]))
                } catch {
                    input = nil
                    print("SURVEY \(fixture.name) template error \(error)")
                }
                let rendered = input?.text.tokens.asArray(Int.self) ?? []
                let promptMatch = rendered == fixture.prompt
                let promptDiff = zip(rendered, fixture.prompt).prefix(while: { $0 == $1 }).count
                if !promptMatch {
                    print("SURVEY \(fixture.name) first prompt difference \(promptDiff)"
                        + " Python \(Array(fixture.prompt.dropFirst(promptDiff).prefix(8)))"
                        + " Swift \(Array(rendered.dropFirst(promptDiff).prefix(8)))")
                }
                // Hold input IDs fixed when diagnosing arithmetic after a template mismatch.
                let targetInput = promptMatch ? input! : LMInput(tokens: MLXArray(fixture.prompt))
                let cache = try context.model.newCache(parameters: nil)
                // Packed projections can round differently with a different
                // number of rows. Reproduce the reference's window boundaries.
                let chunking: PrefillParameters.Chunking
                switch fixture.prefillChunking {
                case nil, "remainder":
                    chunking = .remainder
                case "balanced":
                    chunking = .balanced
                default:
                    XCTFail("Unknown reference chunking for \(fixture.name)")
                    continue
                }
                let prepared = try context.model.prepare(
                    targetInput, cache: cache, state: nil,
                    prefill: .init(stepSize: 1024, chunking: chunking))
                guard case .logits(var output) = prepared else {
                    XCTFail("Expected prepared logits for \(fixture.name)")
                    continue
                }
                var firstDifference: Int?
                var maximum: Float = 0
                var scores = 0
                for (index, token) in fixture.generated.enumerated() {
                    let row = output.logits[0, -1].asType(.float32)
                    if firstDifference == nil {
                        let native = argMax(row).item(Int.self)
                        if native != token {
                            firstDifference = index
                            print("SURVEY \(fixture.name) first greedy difference \(index) Python \(token) Swift \(native)")
                        }
                    }
                    if index < fixture.sampled.count {
                        let sample = fixture.sampled[index]
                        let actual = row[MLXArray(sample.ids)].asArray(Float.self)
                        scores += actual.count
                        for (a, b) in zip(actual, sample.scores) {
                            maximum = max(maximum, abs(a - b))
                        }
                    }
                    if index + 1 < fixture.generated.count {
                        output = context.model(
                            LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)), cache: cache, state: output.state)
                    }
                }
                print("SURVEY \(fixture.name) prompt \(promptDiff)/\(fixture.prompt.count) native \(rendered.count)"
                    + " greedy \(firstDifference ?? fixture.generated.count)/\(fixture.generated.count)"
                    + " logits \(scores) max \(maximum)")
                XCTAssertTrue(promptMatch, fixture.name)
                XCTAssertNil(firstDifference, fixture.name)
                XCTAssertLessThanOrEqual(maximum, 0.25, fixture.name)
            }
        }
    }

    private static func trace(
        _ model: Qwen35, fixture: Case, reference: [String: MLXArray]
    ) throws {
        func compare(_ name: String, _ array: MLXArray) {
            guard let expected = reference[name], expected.shape == array.shape else {
                print("TRACE \(name) shape mismatch")
                return
            }
            let maximum = abs(expected.asType(.float32) - array.asType(.float32)).max().item(Float.self)
            print("TRACE \(name) \(maximum)")
        }

        let tokens = MLXArray(fixture.prompt).reshaped(1, -1)
        let text = model.languageModel.model
        let layers = model.loraLayers.compactMap { $0 as? Qwen35Language.DecoderLayer }
        let caches = try model.newCache(parameters: nil)
        var hidden = text.embedTokens(tokens)
        compare("embed_tokens", hidden)
        if let first = layers.first, let linear = first.linearAttn {
            let normed = first.inputLayerNorm(hidden)
            compare("input_layernorm", normed)
            compare("in_proj_qkv", linear.inProjQKV(normed))
            compare("in_proj_z", linear.inProjZ(normed))
            compare("in_proj_b", linear.inProjB(normed))
            compare("in_proj_a", linear.inProjA(normed))
            let actual = linear(normed)
            compare("linear_attn", actual)
            let fused = linear.fusedInputProjectionEnabled
            linear.fusedInputProjectionEnabled = false
            if let a = reference["linear_attn"] {
                let separate = linear(normed)
                print("TRACE linear_attn separate "
                      + "\(abs(a.asType(.float32) - separate.asType(.float32)).max().item(Float.self))")
            }
            linear.fusedInputProjectionEnabled = fused
        }
        let faMask: MLXArray?
        if case .array(let mask) = createAttentionMask(h: hidden, cache: caches[text.faIdx], returnArray: true) {
            faMask = mask
        } else {
            faMask = nil
        }
        let ssmMask = createSSMMask(h: hidden, cache: caches[text.ssmIdx] as? MambaCache)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, attentionMask: faMask, ssmMask: ssmMask, cache: caches[index], positionIds: nil)
            compare("layer_\(index)", hidden)
        }
    }

    private static func sendableValue(_ value: MLXLMCommon.JSONValue) -> any Sendable {
        switch value {
        case .null: NSNull()
        case .bool(let value): value
        case .int(let value): value
        case .double(let value): value
        case .string(let value): value
        case .array(let values): values.map(sendableValue)
        case .object(let values): values.mapValues(sendableValue)
        }
    }
}
