import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLX
import MLXCat
import Tokenizers
import XCTest

final class HybridBatchIntegrationTests: XCTestCase {
    func testHybridFixedStateInsertRemoveExtractMidBatch() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_HYBRID_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_HYBRID_TEST_MODEL to Qwen3.6-27B-4bit to run this gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 2, temperature: 0)
            let generator = ContinuousBatchGenerator(
                model: context.model,
                parameters: parameters
            )
            let prompts = [
                "The capital of France is",
                "In Swift, an actor protects",
                "List three colors:",
            ]
            var inputs: [LMInput] = []
            for prompt in prompts {
                inputs.append(
                    try await context.processor.prepare(input: UserInput(prompt: prompt))
                )
            }

            try generator.insert(uid: "hybrid-0", input: inputs[0], sampling: SamplingParameters(temperature: 0))
            try generator.insert(uid: "hybrid-1", input: inputs[1], sampling: SamplingParameters(temperature: 0))
            XCTAssertEqual(generator.next().count, 2)
            try generator.insert(uid: "hybrid-2", input: inputs[2], sampling: SamplingParameters(temperature: 0))
            generator.remove(uid: "hybrid-1")

            XCTAssertEqual(generator.uids, ["hybrid-0", "hybrid-2"])
            XCTAssertNotNil(generator.extractCache(uid: "hybrid-0"))
            XCTAssertNotNil(generator.extractCache(uid: "hybrid-2"))
            XCTAssertEqual(generator.next().count, 2)
        }
    }

    func testHybridBatchMatchesSerialGreedyTokens() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_HYBRID_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_HYBRID_TEST_MODEL to Qwen3.6-27B-4bit to run this gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 2, temperature: 0)
            let prompts = [
                "The capital of France is",
                "In Swift, an actor protects",
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
                    steps: 2
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
                        uid: "hybrid-batch-\(index)",
                        input: input,
                        maxTokens: 2,
                        sampling: SamplingParameters(temperature: 0)
                    )
                }
            )

            for index in inputs.indices {
                XCTAssertEqual(batched["hybrid-batch-\(index)"], serial[index])
            }
        }
    }
}
