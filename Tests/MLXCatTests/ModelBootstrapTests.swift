import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

private struct GreedyBaselineFixture: Decodable {
    let model: String
    let prompt: String
    let maxTokens: Int
    let tokenIds: [Int]
}

final class ModelBootstrapTests: XCTestCase {
    func testLocalModelResolutionSkipsCleanlyWhenAbsent() throws {
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip(
                "No pinned local MLX test model found. Set MLXSERVE_TEST_MODEL to run model-dependent baselines."
            )
        }

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: resolution.url.appendingPathComponent("config.json").path
            ),
            "resolved model from \(resolution.source) must contain config.json"
        )
    }

    func testQwenGreedyGenerationMatchesFixture() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip(
                "No pinned local MLX test model found. Set MLXSERVE_TEST_MODEL to run model-dependent baselines."
            )
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("Greedy baseline fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let fixture = try FixtureLoader.loadJSON(
            "qwen3_0_6b_greedy_baseline",
            as: GreedyBaselineFixture.self
        )
        let generated = try await generateTokenIds(
            modelDirectory: resolution.url,
            prompt: fixture.prompt,
            maxTokens: fixture.maxTokens
        )

        XCTAssertEqual(generated, fixture.tokenIds)
    }

    private func generateTokenIds(
        modelDirectory: URL,
        prompt: String,
        maxTokens: Int
    ) async throws -> [Int] {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDirectory,
            using: #huggingFaceTokenizerLoader()
        )

        return try await container.perform { context in
            let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
            var iterator = try TokenIterator(
                input: input,
                model: context.model,
                parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0)
            )

            var tokenIds: [Int] = []
            while let tokenId = iterator.next() {
                tokenIds.append(tokenId)
            }
            return tokenIds
        }
    }
}
