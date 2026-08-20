import Foundation
@testable import MLXCatHTTP
import XCTest

final class CompletionsTests: XCTestCase {
    func testCompletionRequestParsesStringPromptAndSamplingFields() throws {
        let request = try OpenAICompletionRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "prompt": "hello",
                  "max_tokens": 32,
                  "temperature": 0.7,
                  "top_p": 0.8,
                  "top_k": 42,
                  "min_p": 0.05,
                  "repetition_penalty": 1.2,
                  "presence_penalty": 0.5,
                  "frequency_penalty": 0.6,
                  "min_tokens": 6,
                  "logit_bias": {"7": 4.25},
                  "stop": "END",
                  "seed": 1234,
                  "stream": true,
                  "stream_options": { "include_usage": true }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.prompt, .string("hello"))
        XCTAssertEqual(request.maxTokens, 32)
        XCTAssertEqual(request.temperature, 0.7, accuracy: 0.0001)
        XCTAssertEqual(request.topP, 0.8, accuracy: 0.0001)
        XCTAssertEqual(request.topK, 42)
        XCTAssertEqual(request.minP, 0.05, accuracy: 0.0001)
        XCTAssertEqual(request.repetitionPenalty, 1.2, accuracy: 0.0001)
        XCTAssertEqual(request.presencePenalty, 0.5, accuracy: 0.0001)
        XCTAssertEqual(request.frequencyPenalty, 0.6, accuracy: 0.0001)
        XCTAssertEqual(request.minTokens, 6)
        XCTAssertEqual(try XCTUnwrap(request.logitBias[7]), 4.25, accuracy: 0.0001)
        XCTAssertEqual(request.stop, ["END"])
        XCTAssertEqual(request.seed, 1234)
        XCTAssertTrue(request.stream)
        XCTAssertTrue(request.includeUsage)
    }

    func testCompletionRequestParsesArrayPromptWithDefaults() throws {
        let request = try OpenAICompletionRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "prompt": ["one", "two"],
                  "stop": ["END", "STOP"]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.prompt, .strings(["one", "two"]))
        XCTAssertEqual(request.maxTokens, 16)
        XCTAssertEqual(request.temperature, 0)
        XCTAssertEqual(request.topP, 0)
        XCTAssertEqual(request.topK, 0)
        XCTAssertEqual(request.minP, 0)
        XCTAssertEqual(request.repetitionPenalty, 1)
        XCTAssertEqual(request.presencePenalty, 0)
        XCTAssertEqual(request.frequencyPenalty, 0)
        XCTAssertEqual(request.minTokens, 0)
        XCTAssertTrue(request.logitBias.isEmpty)
        XCTAssertEqual(request.stop, ["END", "STOP"])
        XCTAssertNil(request.seed)
        XCTAssertFalse(request.stream)
        XCTAssertFalse(request.includeUsage)
    }

    func testCompletionResponseShapeUsesTextCompletionObjectAndTextChoices() throws {
        let request = OpenAICompletionRequest(
            model: "test-model",
            prompt: .string("hello"),
            maxTokens: 16
        )
        let response = buildCompletionResponse(
            request: request,
            id: "cmpl_test",
            created: 123,
            choices: [
                buildCompletionChoice(index: 0, text: "answer", finishReason: "stop")
            ],
            promptTokens: 5,
            completionTokens: 3
        )

        XCTAssertEqual(response["id"] as? String, "cmpl_test")
        XCTAssertEqual(response["object"] as? String, "text_completion")
        XCTAssertEqual(response["created"] as? Int, 123)
        XCTAssertEqual(response["model"] as? String, "test-model")
        let choices = try XCTUnwrap(response["choices"] as? [[String: Any]])
        XCTAssertEqual(choices[0]["index"] as? Int, 0)
        XCTAssertEqual(choices[0]["text"] as? String, "answer")
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop")
        XCTAssertTrue(choices[0]["logprobs"] is NSNull)

        let usage = try XCTUnwrap(response["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 5)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 3)
        XCTAssertEqual(usage["total_tokens"] as? Int, 8)
    }

    func testCompletionStreamingFormatterEmitsTextCompletionChunks() throws {
        var formatter = CompletionStreamFormatter(
            id: "cmpl_test",
            model: "test-model",
            created: 123,
            stopSequences: []
        )

        let first = formatter.feed(OpenAIChatChunk(text: "hel", tokenID: 1))
        let second = formatter.feed(OpenAIChatChunk(text: "lo", tokenID: 2))
        let final = formatter.finish()

        let chunks = first + second + final
        XCTAssertEqual(chunks.count, 3)
        XCTAssertEqual(chunks.compactMap { $0["object"] as? String }, Array(repeating: "text_completion", count: 3))
        XCTAssertEqual(chunkText(chunks[0]), "hel")
        XCTAssertEqual(chunkText(chunks[1]), "lo")
        XCTAssertEqual(chunkText(chunks[2]), "")
        XCTAssertTrue(chunkFinishReason(chunks[0]) is NSNull)
        XCTAssertTrue(chunkFinishReason(chunks[1]) is NSNull)
        XCTAssertEqual(chunkFinishReason(chunks[2]) as? String, "length")
    }

    func testCompletionStreamingFormatterHonorsStopSequences() {
        var formatter = CompletionStreamFormatter(
            id: "cmpl_test",
            model: "test-model",
            created: 123,
            stopSequences: ["STOP"]
        )

        let first = formatter.feed(OpenAIChatChunk(text: "hello ST", tokenID: 1))
        let second = formatter.feed(OpenAIChatChunk(text: "OP hidden", tokenID: 2))
        let final = formatter.finish()

        XCTAssertEqual(first.count, 1)
        XCTAssertEqual(chunkText(first[0]), "hello")
        XCTAssertEqual(second.count, 1)
        XCTAssertEqual(chunkText(second[0]), " ")
        XCTAssertEqual(final.count, 1)
        XCTAssertEqual(chunkText(final[0]), "")
        XCTAssertEqual(chunkFinishReason(final[0]) as? String, "stop")
        XCTAssertTrue(formatter.isStopped)
    }

    private func chunkText(_ chunk: [String: Any]) -> String? {
        let choices = chunk["choices"] as? [[String: Any]]
        return choices?.first?["text"] as? String
    }

    private func chunkFinishReason(_ chunk: [String: Any]) -> Any? {
        let choices = chunk["choices"] as? [[String: Any]]
        return choices?.first?["finish_reason"]
    }
}
