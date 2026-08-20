import Foundation
@testable import MLXCatHTTP
import XCTest

final class AnthropicDialectTests: XCTestCase {
    func testMessagesRequestRequiresMaxTokens() {
        let body = Data(
            """
            {
              "model": "test-model",
              "messages": [{"role": "user", "content": "hello"}]
            }
            """.utf8
        )

        XCTAssertThrowsError(try AnthropicMessagesRequest.parse(body)) { error in
            XCTAssertEqual(error as? OpenAIServerError, .invalidJSON)
        }
    }

    func testCountTokensRequestAllowsMissingMaxTokensAndReturnsInputTokens() throws {
        let request = try AnthropicCountTokensRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hi"}]
                }
                """.utf8
            )
        )

        let response = buildAnthropicCountTokensResponse(request: request)

        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages[0].content, "hi")
        XCTAssertEqual(response["input_tokens"], 2)
    }

    func testCountTokensRequestCarriesToolsIntoOpenAIRequest() throws {
        let request = try AnthropicCountTokensRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hi"}],
                  "tools": [{"name": "lookup", "description": "search"}],
                  "tool_choice": {"type": "tool", "name": "lookup"}
                }
                """.utf8
            )
        )
        let openAIRequest = request.openAIRequest()

        XCTAssertEqual(openAIRequest.tools?.count, 1)
        XCTAssertEqual(openAIRequest.toolChoice, .function("lookup"))
        XCTAssertEqual(openAIRequest.chatTemplateKwargs?["tools"], .array([.object(["name": .string("lookup"), "description": .string("search")])]))
    }

    func testCountTokensHandlerUsesExactBackendWhenAvailable() async {
        let handler = AnthropicMessagesHandler(backend: ExactCountBackend(inputTokens: 42))
        let response = await handler.countTokensResponseForTesting(
            HTTPRequest(
                method: "POST",
                path: "/v1/messages/count_tokens",
                headers: [:],
                body: Self.countTokensBody
            )
        )

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(response.body["input_tokens"] as? Int, 42)
        XCTAssertNil(response.body["estimated"])
    }

    func testCountTokensHandlerFallbackMatchesReferenceShape() async {
        let handler = AnthropicMessagesHandler(backend: EstimateOnlyBackend())
        let response = await handler.countTokensResponseForTesting(
            HTTPRequest(
                method: "POST",
                path: "/v1/messages/count_tokens",
                headers: [:],
                body: Self.countTokensBody
            )
        )

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(response.body["input_tokens"] as? Int, 2)
        XCTAssertNil(response.body["estimated"])
        XCTAssertEqual(Set(response.body.keys), ["input_tokens"])
    }

    func testMessagesRequestParsesSystemStringAndTranslationFields() throws {
        let request = try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 64,
                  "system": "You are concise.",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stop_sequences": ["END"],
                  "stream": true,
                  "temperature": 0.4,
                  "top_p": 0.8,
                  "top_k": 20,
                  "thinking": {"type": "enabled", "budget_tokens": 16},
                  "tools": [{"name": "lookup", "description": "search"}],
                  "tool_choice": {"type": "tool", "name": "lookup"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.maxTokens, 64)
        XCTAssertEqual(request.messages.map(\.role), ["system", "user"])
        XCTAssertEqual(request.messages[0].content, "You are concise.")
        XCTAssertEqual(request.stopSequences, ["END"])
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.temperature, 0.4, accuracy: 0.0001)
        XCTAssertEqual(request.topP, 0.8, accuracy: 0.0001)
        XCTAssertEqual(request.topK, 20)
        XCTAssertEqual(request.enableThinking, true)
        XCTAssertEqual(request.thinkingBudget, 16)
        XCTAssertEqual(request.chatTemplateKwargs?["tools"], .array([.object(["name": .string("lookup"), "description": .string("search")])]))
        XCTAssertEqual(request.chatTemplateKwargs?["tool_choice"], .object(["type": .string("tool"), "name": .string("lookup")]))

        let chatRequest = request.openAIRequest()
        XCTAssertEqual(chatRequest.model, "test-model")
        XCTAssertEqual(chatRequest.maxTokens, 64)
        XCTAssertEqual(chatRequest.stop, ["END"])
        XCTAssertEqual(chatRequest.enableThinking, true)
        XCTAssertEqual(chatRequest.thinkingBudget, 16)
        XCTAssertNotNil(chatRequest.tools)
        XCTAssertEqual(chatRequest.toolChoice, .function("lookup"))
    }

    func testMessagesRequestConvertsAnthropicToolsAndToolChoice() throws {
        let toolRequest = try anthropicRequestWithToolChoice(
            """
            {"type": "tool", "name": "get_weather"}
            """
        )
        let toolChatRequest = toolRequest.openAIRequest()

        XCTAssertEqual(toolChatRequest.toolChoice, .function("get_weather"))
        let tools = try XCTUnwrap(toolChatRequest.tools)
        XCTAssertEqual(tools.count, 1)
        XCTAssertEqual(
            tools[0],
            .object(
                [
                    "type": .string("function"),
                    "function": .object(
                        [
                            "name": .string("get_weather"),
                            "description": .string("Get weather"),
                            "parameters": .object(
                                [
                                    "type": .string("object"),
                                    "properties": .object(
                                        [
                                            "city": .object(["type": .string("string")])
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            )
        )

        let anyRequest = try anthropicRequestWithToolChoice(#"{"type": "any"}"#)
        XCTAssertEqual(anyRequest.openAIRequest().toolChoice, .required)

        let autoRequest = try anthropicRequestWithToolChoice(#"{"type": "auto"}"#)
        XCTAssertEqual(autoRequest.openAIRequest().toolChoice, .auto)
    }

    func testMessagesRequestParsesSystemArrayAndContentBlocks() throws {
        let request = try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 16,
                  "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "image", "source": {"type": "base64", "data": "..."}},
                        {"type": "tool_result", "content": [{"type": "text", "text": "result"}]}
                      ]
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.messages[0].content, "AB")
        XCTAssertEqual(request.messages[1].content, "hello result")
    }

    func testBuildAnthropicMessageResponseSplitsThinkingAndText() throws {
        let request = try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hello"}]
                }
                """.utf8
            )
        )
        let completion = AnthropicBufferedCompletion(
            text: "<think>reasoning</think>answer",
            completionTokens: 3,
            finishReason: "stop",
            stoppedByTextStop: false,
            stopSequence: nil
        )

        let response = buildAnthropicMessageResponse(
            request: request,
            completion: completion,
            promptTokens: 5,
            id: "msg_test"
        )

        XCTAssertEqual(response["id"] as? String, "msg_test")
        XCTAssertEqual(response["type"] as? String, "message")
        XCTAssertEqual(response["role"] as? String, "assistant")
        XCTAssertEqual(response["model"] as? String, "test-model")
        XCTAssertEqual(response["stop_reason"] as? String, "end_turn")
        let content = try XCTUnwrap(response["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 2)
        XCTAssertEqual(content[0]["type"] as? String, "thinking")
        XCTAssertEqual(content[0]["thinking"] as? String, "reasoning")
        XCTAssertEqual(content[1]["type"] as? String, "text")
        XCTAssertEqual(content[1]["text"] as? String, "answer")
        let usage = try XCTUnwrap(response["usage"] as? [String: Int])
        XCTAssertEqual(usage["input_tokens"], 5)
        XCTAssertEqual(usage["output_tokens"], 3)
    }

    func testBuildAnthropicMessageResponseEmitsToolUseBlock() throws {
        let request = try anthropicRequestWithToolChoice(#"{"type": "auto"}"#)
        let completion = AnthropicBufferedCompletion(
            text: #"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
            completionTokens: 3,
            finishReason: "stop",
            stoppedByTextStop: false,
            stopSequence: nil
        )

        let response = buildAnthropicMessageResponse(
            request: request,
            completion: completion,
            promptTokens: 5,
            id: "msg_test"
        )

        XCTAssertEqual(response["stop_reason"] as? String, "tool_use")
        let content = try XCTUnwrap(response["content"] as? [[String: Any]])
        XCTAssertEqual(content.count, 1)
        XCTAssertEqual(content[0]["type"] as? String, "tool_use")
        XCTAssertEqual(content[0]["name"] as? String, "get_weather")
        let input = try XCTUnwrap(content[0]["input"] as? [String: Any])
        XCTAssertEqual(input["city"] as? String, "Paris")
    }

    func testStreamingFormatterEventOrderForThinkingThenText() {
        var formatter = AnthropicStreamFormatter(
            id: "msg_test",
            model: "test-model",
            promptTokens: 5,
            stopSequences: []
        )

        var events = formatter.startEvents()
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "<think>reason", tokenID: 1)))
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "ing</think>answer", tokenID: 2, finishReason: "stop")))
        events.append(contentsOf: formatter.finishEvents())

        XCTAssertEqual(
            events.map(\.name),
            [
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        )

        let firstBlock = events[2].payload["content_block"] as? [String: Any]
        XCTAssertEqual(firstBlock?["type"] as? String, "thinking")
        let secondBlock = events[6].payload["content_block"] as? [String: Any]
        XCTAssertEqual(secondBlock?["type"] as? String, "text")
    }

    func testStreamingFormatterReportsStopSequence() throws {
        var formatter = AnthropicStreamFormatter(
            id: "msg_test",
            model: "test-model",
            promptTokens: 5,
            stopSequences: ["END"]
        )

        var events = formatter.startEvents()
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "answer EN", tokenID: 1)))
        events.append(contentsOf: formatter.feed(OpenAIChatChunk(text: "D hidden", tokenID: 2)))
        events.append(contentsOf: formatter.finishEvents())

        let delta = try XCTUnwrap(events.first { $0.name == "message_delta" }?.payload["delta"] as? [String: Any])
        XCTAssertEqual(delta["stop_reason"] as? String, "stop_sequence")
        XCTAssertEqual(delta["stop_sequence"] as? String, "END")
    }

    func testStreamingFormatterEmitsToolUseBlock() throws {
        var formatter = AnthropicStreamFormatter(
            id: "msg_test",
            model: "test-model",
            promptTokens: 5,
            stopSequences: [],
            toolsRequested: true
        )

        var events = formatter.startEvents()
        events.append(
            contentsOf: formatter.feed(
                OpenAIChatChunk(
                    text: #"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
                    tokenID: 1,
                    finishReason: "stop"
                )
            )
        )
        events.append(contentsOf: formatter.finishEvents())

        XCTAssertEqual(
            events.map(\.name),
            [
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        )

        let toolBlock = try XCTUnwrap(events[2].payload["content_block"] as? [String: Any])
        XCTAssertEqual(toolBlock["type"] as? String, "tool_use")
        XCTAssertEqual(toolBlock["name"] as? String, "get_weather")

        let delta = try XCTUnwrap(events[3].payload["delta"] as? [String: Any])
        XCTAssertEqual(delta["type"] as? String, "input_json_delta")
        XCTAssertEqual(delta["partial_json"] as? String, #"{"city":"Paris"}"#)

        let messageDelta = try XCTUnwrap(events[5].payload["delta"] as? [String: Any])
        XCTAssertEqual(messageDelta["stop_reason"] as? String, "tool_use")
    }

    private func anthropicRequestWithToolChoice(_ toolChoice: String) throws -> AnthropicMessagesRequest {
        try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 16,
                  "messages": [{"role": "user", "content": "weather"}],
                  "tools": [
                    {
                      "name": "get_weather",
                      "description": "Get weather",
                      "input_schema": {
                        "type": "object",
                        "properties": {
                          "city": {"type": "string"}
                        }
                      }
                    }
                  ],
                  "tool_choice": \(toolChoice)
                }
                """.utf8
            )
        )
    }

    private static let countTokensBody = Data(
        """
        {
          "model": "test-model",
          "messages": [{"role": "user", "content": "hi"}]
        }
        """.utf8
    )
}

private final class ExactCountBackend: OpenAIChatBackend, AnthropicTokenCountingBackend, @unchecked Sendable {
    let models = [OpenAIModelInfo(id: "test-model")]
    private let inputTokens: Int

    init(inputTokens: Int) {
        self.inputTokens = inputTokens
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        OpenAIChatStream(promptTokens: 0, chunks: AsyncThrowingStream { $0.finish() })
    }

    func countTokens(_ request: AnthropicCountTokensRequest) async throws -> AnthropicCountTokensResult {
        AnthropicCountTokensResult(inputTokens: inputTokens)
    }
}

private final class EstimateOnlyBackend: OpenAIChatBackend, @unchecked Sendable {
    let models = [OpenAIModelInfo(id: "test-model")]

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        OpenAIChatStream(promptTokens: 0, chunks: AsyncThrowingStream { $0.finish() })
    }
}
