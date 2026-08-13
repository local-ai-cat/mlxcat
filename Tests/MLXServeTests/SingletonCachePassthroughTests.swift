import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import Tokenizers
import XCTest

final class SingletonCachePassthroughTests: XCTestCase {

    func testSingleRequestGeneratesTokens() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run singleton cache tests.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
            let generator = ContinuousBatchGenerator(
                model: context.model,
                parameters: parameters
            )
            let input = try await context.processor.prepare(
                input: UserInput(prompt: "The capital of France is")
            )
            try generator.insert(
                uid: "single-0",
                input: input,
                sampling: SamplingParameters(temperature: 0)
            )

            var tokens: [Int] = []
            for _ in 0 ..< 8 {
                let responses = generator.next()
                XCTAssertEqual(responses.count, 1)
                if let response = responses.first {
                    tokens.append(response.token)
                    if response.finishReason != nil { break }
                }
            }
            XCTAssertFalse(tokens.isEmpty, "Expected at least one token from singleton decode")
        }
    }

    func testSingletonMatchesBatchedOutput() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run singleton cache tests.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 16, temperature: 0)
            let prompt = "The capital of France is"
            let input = try await context.processor.prepare(
                input: UserInput(prompt: prompt)
            )

            let singletonGen = ContinuousBatchGenerator(
                model: context.model,
                parameters: parameters
            )
            try singletonGen.insert(
                uid: "solo",
                input: input,
                sampling: SamplingParameters(temperature: 0)
            )
            var singletonTokens: [Int] = []
            for _ in 0 ..< 16 {
                let responses = singletonGen.next()
                guard let response = responses.first else { break }
                singletonTokens.append(response.token)
                if response.finishReason != nil { break }
            }

            let batchedGen = ContinuousBatchGenerator(
                model: context.model,
                parameters: parameters
            )
            let dummyInput = try await context.processor.prepare(
                input: UserInput(prompt: "Dummy padding prompt for batch")
            )
            try batchedGen.insert(
                uid: "target",
                input: input,
                sampling: SamplingParameters(temperature: 0)
            )
            try batchedGen.insert(
                uid: "dummy",
                input: dummyInput,
                sampling: SamplingParameters(temperature: 0)
            )
            batchedGen.remove(uid: "dummy")

            var batchedTokens: [Int] = []
            for _ in 0 ..< 16 {
                let responses = batchedGen.next()
                guard let response = responses.first else { break }
                batchedTokens.append(response.token)
                if response.finishReason != nil { break }
            }

            XCTAssertEqual(
                singletonTokens, batchedTokens,
                "Singleton and force-batched paths must produce identical greedy tokens"
            )
        }
    }

    func testPromotionFromSingletonToBatched() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run singleton cache tests.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 16, temperature: 0)
            let generator = ContinuousBatchGenerator(
                model: context.model,
                parameters: parameters
            )
            let input1 = try await context.processor.prepare(
                input: UserInput(prompt: "The capital of France is")
            )
            let input2 = try await context.processor.prepare(
                input: UserInput(prompt: "List three colors:")
            )

            try generator.insert(
                uid: "first",
                input: input1,
                sampling: SamplingParameters(temperature: 0)
            )
            XCTAssertEqual(generator.count, 1)

            let step1 = generator.next()
            XCTAssertEqual(step1.count, 1, "Width-1 step should produce 1 response")

            try generator.insert(
                uid: "second",
                input: input2,
                sampling: SamplingParameters(temperature: 0)
            )
            XCTAssertEqual(generator.count, 2, "After promotion, should have 2 rows")

            var sawBothRespond = false
            for _ in 0 ..< 4 {
                let step = generator.next()
                if step.count == 2 {
                    sawBothRespond = true
                    for response in step {
                        XCTAssertGreaterThanOrEqual(response.token, 0)
                    }
                    break
                }
            }
            XCTAssertTrue(sawBothRespond, "Width-2 should eventually produce 2 responses per step")
        }
    }

    func testRemoveAfterPromotionContinues() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run singleton cache tests.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 16, temperature: 0)
            let generator = ContinuousBatchGenerator(
                model: context.model,
                parameters: parameters
            )
            let input1 = try await context.processor.prepare(
                input: UserInput(prompt: "The capital of France is")
            )
            let input2 = try await context.processor.prepare(
                input: UserInput(prompt: "List three colors:")
            )

            try generator.insert(
                uid: "a", input: input1,
                sampling: SamplingParameters(temperature: 0)
            )
            _ = generator.next()

            try generator.insert(
                uid: "b", input: input2,
                sampling: SamplingParameters(temperature: 0)
            )
            XCTAssertEqual(generator.count, 2)
            _ = generator.next()

            generator.remove(uid: "b")
            XCTAssertEqual(generator.count, 1)
            XCTAssertEqual(generator.uids, ["a"])

            let step = generator.next()
            XCTAssertEqual(step.count, 1)
            XCTAssertGreaterThanOrEqual(step[0].token, 0)
        }
    }

    func testExtractCacheInSingletonMode() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run singleton cache tests.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            let generator = ContinuousBatchGenerator(
                model: context.model,
                parameters: parameters
            )
            let input = try await context.processor.prepare(
                input: UserInput(prompt: "The capital of France is")
            )
            try generator.insert(
                uid: "extract-test",
                input: input,
                sampling: SamplingParameters(temperature: 0)
            )
            _ = generator.next()

            let extracted = generator.extractCache(uid: "extract-test")
            XCTAssertNotNil(extracted, "extractCache should work in singleton mode")
            if let layers = extracted {
                XCTAssertFalse(layers.isEmpty, "Extracted cache should have layers")
                for layer in layers {
                    XCTAssertFalse(layer.state.isEmpty, "Each layer should have state")
                }
            }
        }
    }
}
