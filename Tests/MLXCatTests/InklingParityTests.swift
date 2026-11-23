import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXCatNative

final class InklingParityTests: XCTestCase {
    private struct Fixture: Decodable {
        struct TopLogit: Decodable {
            let id: Int
            let logit: Float
        }

        let inputIDs: [Int]
        let generatedIDs: [Int]
        let prefillChunk: Int
        let topLogits: [[TopLogit]]
        let imageSHA256: String?
        let messages: MLXLMCommon.JSONValue
        let settings: [String: MLXLMCommon.JSONValue]

        enum CodingKeys: String, CodingKey {
            case inputIDs = "input_ids"
            case generatedIDs = "generated_ids"
            case prefillChunk = "prefill_chunk_tokens"
            case topLogits = "top_logits"
            case imageSHA256 = "image_sha256"
            case messages
            case settings
        }
    }

    func testGreedyTokensAndReferenceLogits() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MLXCAT_INKLING_VERIFIED_MODEL"],
            let fixturePath = environment["MLXCAT_INKLING_REFERENCE"]
        else {
            throw XCTSkip("Set the verified checkpoint and reference fixture directory to run Inkling parity.")
        }
        try MLXMetalRuntime.requireAvailable()
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let fixtures = URL(fileURLWithPath: fixturePath, isDirectory: true)
        let paths = try FileManager.default.contentsOfDirectory(at: fixtures, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix(directory.lastPathComponent + "-") && $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertEqual(paths.count, 12, "Expected all twelve fixture cases")
        guard !paths.isEmpty else { return }

        let decoder = JSONDecoder()
        let config = try decoder.decode(
            InklingConfiguration.self, from: Data(contentsOf: directory.appending(component: "config.json")))
        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: directory)
        let imageTokenID = config.imageTokenID ?? 200054
        let processor = InklingInputProcessor(
            tokenizer: tokenizer, imageTokenID: imageTokenID,
            imagePatchSize: config.vision.map { $0.patchSize * $0.merge })
        let weights = try InklingCheckpoint(directory: directory)
        let vision = try config.vision.map { try weights.visionModel(config: $0) }
        let model = try InklingLanguageModel(
            textModel: weights.textModel(config: config.text),
            visionModel: vision, imageTokenID: imageTokenID)
        let tolerance: Float = 0.25

        for path in paths {
            let fixture = try decoder.decode(Fixture.self, from: Data(contentsOf: path))
            let cache = try model.newCache(parameters: nil)
            let imageSource: [UserInput.Image] = fixture.imageSHA256 == nil ? []
                : [.url(fixtures.appending(component: "shapes.png"))]
            let settings = fixture.settings.mapValues(Self.sendableValue)
            let tools = settings["tools"] as? [[String: any Sendable]]
            let userInput: UserInput
            switch fixture.messages {
            case .string(let prompt):
                userInput = UserInput(
                    chat: [.user(prompt, images: imageSource)], tools: tools, additionalContext: settings)
            case .array(let messages):
                userInput = UserInput(
                    messages: messages.compactMap { Self.sendableValue($0) as? [String: any Sendable] },
                    images: imageSource, tools: tools, additionalContext: settings)
            default:
                return XCTFail("Unexpected fixture message format")
            }
            let input = try processor.prepare(input: userInput)
            let inputIDs = input.text.tokens.asArray(Int.self)
            XCTAssertEqual(inputIDs, fixture.inputIDs, path.lastPathComponent)
            guard inputIDs == fixture.inputIDs else { continue }
            let result = try model.prepare(
                input, cache: cache, state: nil, prefill: .init(stepSize: fixture.prefillChunk))
            guard case .logits(let output) = result else { return XCTFail("Expected prefill logits") }
            var logits = output.logits
            var errors: [Float] = []
            for (step, reference) in fixture.generatedIDs.prefix(64).enumerated() {
                let row = logits[0, -1, 0...]
                if step < fixture.topLogits.count {
                    for score in fixture.topLogits[step] {
                        let actual = row[score.id].item(Float.self)
                        errors.append(abs(actual - score.logit))
                        XCTAssertEqual(actual, score.logit, accuracy: tolerance,
                                       "\(path.lastPathComponent) step \(step), token \(score.id)")
                    }
                }
                let token = argMax(row).item(Int.self)
                XCTAssertEqual(token, reference, "\(path.lastPathComponent) step \(step)")
                if token != reference { break }
                logits = model(MLXArray([token]), cache: cache)
            }
            let sorted = errors.sorted()
            if let maximum = sorted.last {
                print("Inkling parity \(path.lastPathComponent): max \(maximum), median \(sorted[sorted.count / 2])")
            }
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
