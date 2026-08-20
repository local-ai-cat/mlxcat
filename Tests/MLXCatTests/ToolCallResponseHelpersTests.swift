@testable import MLXCatHTTP
import XCTest

final class ToolCallResponseHelpersTests: XCTestCase {
    func testAssistantMessagePureToolCallUsesNullContentAndToolCalls() throws {
        let message = buildAssistantMessageWithToolCalls(
            content: #"<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>"#,
            reasoning: "Need weather.",
            parsed: ToolCallParseResult(
                content: "",
                toolCalls: [
                    ParsedToolCall(
                        id: "call_0",
                        name: "get_weather",
                        arguments: #"{"location":"Paris"}"#
                    )
                ]
            )
        )

        XCTAssertEqual(message["role"] as? String, "assistant")
        XCTAssertTrue(message["content"] is NSNull)
        XCTAssertEqual(message["reasoning_content"] as? String, "Need weather.")
        let toolCalls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]])
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls[0]["id"] as? String, "call_0")
        XCTAssertEqual(toolCalls[0]["type"] as? String, "function")
        let function = try XCTUnwrap(toolCalls[0]["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "get_weather")
        XCTAssertEqual(function["arguments"] as? String, #"{"location":"Paris"}"#)
    }

    func testAssistantMessageProseAndToolCallPreservesProseContent() throws {
        let message = buildAssistantMessageWithToolCalls(
            content: "raw content",
            reasoning: "",
            parsed: ToolCallParseResult(
                content: "I'll check.",
                toolCalls: [
                    ParsedToolCall(id: "call_0", name: "get_weather", arguments: "{}")
                ]
            )
        )

        XCTAssertEqual(message["content"] as? String, "I'll check.")
        XCTAssertNil(message["reasoning_content"])
        let toolCalls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]])
        XCTAssertEqual(toolCalls.count, 1)
    }

    func testAssistantMessageWithoutToolCallsIsPlainContent() {
        let message = buildAssistantMessageWithToolCalls(
            content: "plain answer",
            reasoning: "",
            parsed: ToolCallParseResult(content: "plain answer", toolCalls: [])
        )

        XCTAssertEqual(message["role"] as? String, "assistant")
        XCTAssertEqual(message["content"] as? String, "plain answer")
        XCTAssertNil(message["tool_calls"])
        XCTAssertNil(message["reasoning_content"])
    }

    func testToolCallDeltaDictionariesIncludeIndicesAndFunctionShape() throws {
        let delta = toolCallDeltaDictionaries(
            from: ToolCallParseResult(
                content: "",
                toolCalls: [
                    ParsedToolCall(id: "call_0", name: "first", arguments: #"{"a":1}"#),
                    ParsedToolCall(id: "call_1", name: "second", arguments: #"{"b":2}"#),
                ]
            )
        )

        XCTAssertEqual(delta.count, 2)
        XCTAssertEqual(delta[0]["index"] as? Int, 0)
        XCTAssertEqual(delta[0]["id"] as? String, "call_0")
        XCTAssertEqual(delta[0]["type"] as? String, "function")
        let firstFunction = try XCTUnwrap(delta[0]["function"] as? [String: Any])
        XCTAssertEqual(firstFunction["name"] as? String, "first")
        XCTAssertEqual(firstFunction["arguments"] as? String, #"{"a":1}"#)

        XCTAssertEqual(delta[1]["index"] as? Int, 1)
        XCTAssertEqual(delta[1]["id"] as? String, "call_1")
        let secondFunction = try XCTUnwrap(delta[1]["function"] as? [String: Any])
        XCTAssertEqual(secondFunction["name"] as? String, "second")
        XCTAssertEqual(secondFunction["arguments"] as? String, #"{"b":2}"#)
    }

    func testFinishReasonForToolCallsOverridesWhenToolCallsExist() {
        let parsed = ToolCallParseResult(
            content: "",
            toolCalls: [ParsedToolCall(id: "call_0", name: "lookup", arguments: "{}")]
        )

        XCTAssertEqual(finishReasonForToolCalls(defaultFinishReason: "stop", parsed: parsed), "tool_calls")
    }

    func testFinishReasonForToolCallsKeepsDefaultWithoutToolCalls() {
        let parsed = ToolCallParseResult(content: "answer", toolCalls: [])

        XCTAssertEqual(finishReasonForToolCalls(defaultFinishReason: "stop", parsed: parsed), "stop")
    }
}
