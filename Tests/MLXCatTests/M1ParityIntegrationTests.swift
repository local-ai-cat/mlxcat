import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
@testable import MLXCatHTTP
import Tokenizers
import XCTest

final class M1ParityIntegrationTests: XCTestCase {
    func testQwenSchedulerMatchesUpstreamGreedyForFocusedPrompt() async throws {
        try MLXMetalRuntime.requireAvailable()
        let environment = ProcessInfo.processInfo.environment
        let prompt: String
        if let inlinePrompt = environment["MLXSERVE_PARITY_PROMPT"] {
            prompt = inlinePrompt
        } else if let promptPath = environment["MLXSERVE_PARITY_PROMPT_FILE"] {
            prompt = try String(contentsOfFile: promptPath, encoding: .utf8)
        } else {
            throw XCTSkip("Set MLXSERVE_PARITY_PROMPT or MLXSERVE_PARITY_PROMPT_FILE to run the focused scheduler parity gate.")
        }
        let resolution = try qwenResolution()
        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let tokenCount = 128
            let parameters = GenerateParameters(maxTokens: tokenCount, temperature: 0)
            let input = try await context.processor.prepare(
                input: UserInput(
                    chat: [.user(prompt)],
                    additionalContext: ["enable_thinking": false]
                )
            )
            let legacyStream = try MLXLMCommon.generateTokens(
                input: input,
                parameters: parameters,
                context: context
            )
            var legacy: [Int] = []
            for await event in legacyStream {
                if case .token(let tokenID) = event {
                    legacy.append(tokenID)
                }
            }
            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 1
            )
            var eosTokenIds = context.configuration.eosTokenIds
            if let tokenizerEosTokenId = context.tokenizer.eosTokenId {
                eosTokenIds.insert(tokenizerEosTokenId)
            }
            let generated = try await engine.generate([
                Request(
                    uid: "focused-mmlu",
                    input: input,
                    maxTokens: tokenCount,
                    sampling: SamplingParameters(temperature: 0),
                    eosTokenIds: eosTokenIds
                )
            ])["focused-mmlu", default: []]

            var schedulerTokens = generated
            if let lastToken = schedulerTokens.last, eosTokenIds.contains(lastToken) {
                schedulerTokens.removeLast()
            }
            XCTAssertEqual(schedulerTokens, legacy)
        }
    }

    func testQwenGeneratedOutputCanBeTruncatedByStopSequence() async throws {
        try MLXMetalRuntime.requireAvailable()
        let resolution = try qwenResolution()
        let chunks = try await generatedDecodedChunks(modelDirectory: resolution.url)
        let output = chunks.joined()
        let stop = try XCTUnwrap(chunks.first { !$0.isEmpty })
        var matcher = StreamingStopSequenceMatcher(stopSequences: [stop])
        var truncated = ""
        var stopped = false

        for chunk in chunks {
            let result = matcher.feed(chunk)
            truncated += result.text
            if result.stopped {
                stopped = true
                break
            }
        }
        if !stopped {
            let result = matcher.finish()
            truncated += result.text
            stopped = result.stopped
        }

        XCTAssertFalse(output.isEmpty)
        XCTAssertTrue(stopped)
        XCTAssertFalse(truncated.contains(stop))
        let stopRange = try XCTUnwrap(output.range(of: stop))
        XCTAssertEqual(truncated, String(output[..<stopRange.lowerBound]))
    }

    func testQwenEnableThinkingChangesPreparedPromptTokenIds() async throws {
        try MLXMetalRuntime.requireAvailable()
        let resolution = try qwenResolution()
        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        let tokenIds = try await container.perform { context in
            let chat: [Chat.Message] = [.user("Give a one word answer: blue or green?")]
            let thinkingInput = try await context.processor.prepare(
                input: UserInput(chat: chat, additionalContext: ["enable_thinking": true])
            )
            let noThinkingInput = try await context.processor.prepare(
                input: UserInput(chat: chat, additionalContext: ["enable_thinking": false])
            )
            return (
                thinking: thinkingInput.text.tokens.asArray(Int.self),
                noThinking: noThinkingInput.text.tokens.asArray(Int.self)
            )
        }

        XCTAssertNotEqual(tokenIds.thinking, tokenIds.noThinking)
    }

    private func qwenResolution() throws -> TestModelResolution {
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to Qwen3-0.6B-4bit to run M1 parity integration tests.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M1 parity integration tests are pinned to Qwen3-0.6B-4bit.")
        }
        return resolution
    }

    private func generatedDecodedChunks(modelDirectory: URL) async throws -> [String] {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDirectory,
            using: #huggingFaceTokenizerLoader()
        )

        return try await container.perform { context in
            let input = try await context.processor.prepare(input: UserInput(prompt: "The capital of France is"))
            var iterator = try TokenIterator(
                input: input,
                model: context.model,
                parameters: GenerateParameters(maxTokens: 8, temperature: 0)
            )

            var chunks: [String] = []
            while let tokenId = iterator.next() {
                chunks.append(context.tokenizer.decode(tokenIds: [tokenId], skipSpecialTokens: true))
            }
            return chunks
        }
    }
}
