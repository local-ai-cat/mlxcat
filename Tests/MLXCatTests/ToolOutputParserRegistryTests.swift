@testable import MLXCatHTTP
import XCTest

/// Per-family tool-call parsing. Each test feeds a representative RAW model
/// output string for one family's native format and asserts the extracted call.
final class ToolOutputParserRegistryTests: XCTestCase {

    // MARK: Family selection

    func testFamilySelectionByModelID() {
        XCTAssertEqual(toolCallModelFamily(forModel: "gpt-oss-20b"), .harmony)
        XCTAssertEqual(toolCallModelFamily(forModel: "mlx-community/gemma-4-E4B-it"), .gemma4)
        XCTAssertEqual(toolCallModelFamily(forModel: "deepseek-r1-distill-qwen-7b"), .deepseek)
        XCTAssertEqual(toolCallModelFamily(forModel: "Qwen3-8B"), .generic)
    }

    func testFamilySelectionByOutputMarkersWhenIDUnknown() {
        let harmony = "<|channel|>commentary to=functions.f<|message|>{}<|call|>"
        XCTAssertEqual(toolCallModelFamily(forModel: "mystery-model", output: harmony), .harmony)

        let deepseek = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>f\n```json\n{}\n```<｜tool▁call▁end｜>"
        XCTAssertEqual(toolCallModelFamily(forModel: "mystery-model", output: deepseek), .deepseek)

        let gemma = "<|tool_call>call:f{}<tool_call|>"
        XCTAssertEqual(toolCallModelFamily(forModel: "mystery-model", output: gemma), .gemma4)
    }

    // MARK: Harmony (gpt-oss)

    func testHarmonyCommentaryToolCall() {
        let text = #"<|channel|>commentary to=functions.get_weather<|message|>{"location": "Seoul"}<|call|>"#
        let result = parseModelOutput(text, model: "gpt-oss-20b", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"location":"Seoul"}"#)
        ])
        XCTAssertEqual(result.content, "")
    }

    func testHarmonyRoleHeaderRecipientWithAnalysisAndFinal() {
        // analysis -> reasoning, final -> content, tool call recipient in the
        // role header before <|channel|>.
        let text = "<|channel|>analysis<|message|>thinking<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Here you go<|end|>"
            + #"<|start|>assistant to=functions.Read<|channel|>commentary<|message|>{"path":"f.py"}<|end|>"#
        let result = parseModelOutput(text, model: "gpt-oss-20b", idGenerator: idGenerator())

        XCTAssertEqual(result.reasoning, "thinking")
        XCTAssertEqual(result.content, "Here you go")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "Read", arguments: #"{"path":"f.py"}"#)
        ])
    }

    func testHarmonyPlainFinalNoToolCall() {
        let text = "<|channel|>final<|message|>Hello there<|return|>"
        let result = parseModelOutput(text, model: "gpt-oss-20b", idGenerator: idGenerator())

        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertEqual(result.content, "Hello there")
    }

    // MARK: Gemma 4

    func testGemma4MarkerBlockBareValue() {
        let text = "<|tool_call>\ncall:get_weather{location: Tokyo}\n<tool_call|>"
        let result = parseModelOutput(text, model: "gemma-4-E4B-it", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"location":"Tokyo"}"#)
        ])
        XCTAssertFalse(result.content.contains("<|tool_call>"))
    }

    func testGemma4MultipleArgsAndTurnEnd() {
        // Real Gemma grammar: <|"|>-delimited strings + bare value, ended by <turn|>.
        let text = "<|tool_call>call:get_weather{city:<|\"|>San Francisco<|\"|>,days:3}<turn|>"
        let result = parseModelOutput(text, model: "gemma-4-E4B-it", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"city":"San Francisco","days":"3"}"#)
        ])
    }

    func testGemma4BareFallbackCall() {
        // Tier-2 fallback: bare `call:name{...}` with no <|tool_call> marker.
        let text = "call:lookup{q:<|\"|>swift<|\"|>}"
        let result = parseModelOutput(text, model: "gemma-4-E4B-it", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"q":"swift"}"#)
        ])
    }

    func testGemma4DelimitedStringValue() {
        let text = "<|tool_call>call:search{query: <|\"|>test<|\"|>}<tool_call|>"
        let result = parseModelOutput(text, model: "gemma-4-E4B-it", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "search", arguments: #"{"query":"test"}"#)
        ])
    }

    func testGemma4EmptyArgs() {
        let text = "<|tool_call>call:get_time{}<tool_call|>"
        let result = parseModelOutput(text, model: "gemma-4-E4B-it", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_time", arguments: "{}")
        ])
    }

    // MARK: DeepSeek

    func testDeepSeekR1NativeToolCall() {
        let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            + "```json\n{\"city\": \"Seoul\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        let result = parseModelOutput(text, model: "deepseek-r1-distill-qwen-7b", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"city":"Seoul"}"#)
        ])
    }

    func testDeepSeekR1WithLeadingThink() {
        let text = "<think>Let me check the weather.</think>"
            + "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            + "```json\n{\"city\": \"Tokyo\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        let result = parseModelOutput(text, model: "deepseek-r1-distill-qwen-7b", idGenerator: idGenerator())

        XCTAssertEqual(result.reasoning, "Let me check the weather.")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"city":"Tokyo"}"#)
        ])
    }

    func testDeepSeekV31ToolCall() {
        // V3.1 form: name<｜tool▁sep｜>{args}, no ```json fence.
        let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
            + "{\"city\": \"Seoul\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        let result = parseModelOutput(text, model: "deepseek-v3.1", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"city":"Seoul"}"#)
        ])
    }

    // MARK: Generic regression (must keep working)

    func testGenericHermesToolCallUnchanged() {
        let text = #"<tool_call>{"name":"lookup","arguments":{"query":"swift"}}</tool_call>"#
        let result = parseModelOutput(text, model: "Qwen3-8B", idGenerator: idGenerator())

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"query":"swift"}"#)
        ])
        XCTAssertEqual(result.content, "")
    }

    func testGenericThinkingSplitPreserved() {
        let text = "<think>reason</think>The answer is 42."
        let result = parseModelOutput(text, model: "Qwen3-8B", idGenerator: idGenerator())

        XCTAssertEqual(result.reasoning, "reason")
        XCTAssertEqual(result.content, "The answer is 42.")
        XCTAssertTrue(result.toolCalls.isEmpty)
    }

    func testToolsNotRequestedSkipsExtraction() {
        let text = #"<tool_call>{"name":"lookup","arguments":{}}</tool_call>"#
        let result = parseModelOutput(text, model: "Qwen3-8B", includeToolCalls: false, idGenerator: idGenerator())

        XCTAssertTrue(result.toolCalls.isEmpty)
        // Content preserved verbatim (markers not stripped) when tools are off.
        XCTAssertEqual(result.content, text)
    }

    // MARK: Streaming finish-time extraction

    func testStreamingHarmonyRecoversToolCallFromRawText() {
        // The visible content channel is empty; the tool call lives in the raw
        // harmony commentary stream only.
        let raw = #"<|channel|>commentary to=functions.ping<|message|>{"n":1}<|call|>"#
        let parsed = streamingToolCallParse(model: "gpt-oss-20b", rawText: raw, bufferedContent: "", idGenerator: idGenerator())

        XCTAssertEqual(parsed.toolCalls, [
            ParsedToolCall(id: "call_0", name: "ping", arguments: #"{"n":1}"#)
        ])
    }

    func testStreamingGenericUsesBufferedContent() {
        let raw = #"<tool_call>{"name":"lookup","arguments":{}}</tool_call>"#
        let parsed = streamingToolCallParse(model: "Qwen3-8B", rawText: raw, bufferedContent: raw, idGenerator: idGenerator())

        XCTAssertEqual(parsed.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: "{}")
        ])
    }

    // MARK: Helpers

    private func idGenerator() -> () -> String {
        var nextID = 0
        return {
            defer { nextID += 1 }
            return "call_\(nextID)"
        }
    }
}
