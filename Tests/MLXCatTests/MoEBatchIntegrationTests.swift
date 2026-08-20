import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers
import XCTest

final class MoEBatchIntegrationTests: XCTestCase {
    func testMoEBatchMatchesSerialGreedyTokens() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_MOE_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_MOE_TEST_MODEL to a MoE model to run this gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let tokenCount = 4
            let parameters = GenerateParameters(maxTokens: tokenCount, temperature: 0)
            let prompts = [
                "Write a Swift function name for parsing JSON:",
                "Complete this sentence: A router selects experts by",
            ]
            var inputs: [LMInput] = []
            for prompt in prompts {
                inputs.append(
                    try await context.processor.prepare(input: UserInput(prompt: prompt))
                )
            }
            let serial = try inputs.map {
                try SerialGreedyTokenHelper.tokens(
                    model: context.model,
                    input: $0,
                    parameters: parameters,
                    steps: tokenCount
                )
            }
            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 2
            )
            let batched = try await engine.generate(
                inputs.enumerated().map { index, input in
                    Request(
                        uid: "moe-batch-\(index)",
                        input: input,
                        maxTokens: tokenCount,
                        sampling: SamplingParameters(temperature: 0)
                    )
                }
            )

            for index in inputs.indices {
                XCTAssertEqual(batched["moe-batch-\(index)"], serial[index])
            }
        }
    }
}
