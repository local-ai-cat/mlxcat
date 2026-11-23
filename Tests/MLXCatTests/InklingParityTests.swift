import Foundation
import MLX
import MLXLMCommon
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

        enum CodingKeys: String, CodingKey {
            case inputIDs = "input_ids"
            case generatedIDs = "generated_ids"
            case prefillChunk = "prefill_chunk_tokens"
            case topLogits = "top_logits"
            case imageSHA256 = "image_sha256"
        }
    }

    func testGreedyTokensAndReferenceLogits() throws {
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
        let weights = try InklingCheckpoint(directory: directory)
        let vision = try config.vision.map { try weights.visionModel(config: $0) }
        let model = try InklingLanguageModel(
            textModel: weights.textModel(config: config.text),
            visionModel: vision, imageTokenID: config.imageTokenID ?? 200054)
        let tolerance: Float = 0.25

        for path in paths {
            let fixture = try decoder.decode(Fixture.self, from: Data(contentsOf: path))
            let cache = try model.newCache(parameters: nil)
            let image: LMInput.ProcessedImage?
            if fixture.imageSHA256 != nil, let config = config.vision {
                let tiles = try InklingImageProcessor.prepare(
                    .url(fixtures.appending(component: "shapes.png")), patch: config.patchSize * config.merge)
                image = .init(pixels: tiles.0, frames: [tiles.1])
            } else {
                image = nil
            }
            let input = LMInput(text: .init(tokens: MLXArray(fixture.inputIDs)), image: image)
            let result = try model.prepare(
                input, cache: cache, state: nil, prefill: .init(stepSize: fixture.prefillChunk))
            guard case .logits(let output) = result else { return XCTFail("Expected prefill logits") }
            var logits = output.logits
            var maximumError: Float = 0
            for (step, reference) in fixture.generatedIDs.prefix(64).enumerated() {
                let row = logits[0, -1, 0...]
                if step < fixture.topLogits.count {
                    for score in fixture.topLogits[step] {
                        let actual = row[score.id].item(Float.self)
                        maximumError = max(maximumError, abs(actual - score.logit))
                        XCTAssertEqual(actual, score.logit, accuracy: tolerance,
                                       "\(path.lastPathComponent) step \(step), token \(score.id)")
                    }
                }
                let token = argMax(row).item(Int.self)
                XCTAssertEqual(token, reference, "\(path.lastPathComponent) step \(step)")
                if token != reference { break }
                logits = model(MLXArray([token]), cache: cache)
            }
            print("Inkling parity \(path.lastPathComponent) maximum observed logit error \(maximumError)")
        }
    }
}
