@testable import MLXCatHTTP
import XCTest

/// Parser-level tests for the vLLM-shaped `ToolParser` family parsers: buffered
/// `extractToolCalls` per family + Hermes incremental streaming + the manager.
final class ToolParsersTests: XCTestCase {

    // MARK: Manager

    func testManagerRegistersAllFamilies() {
        for name in ["hermes", "llama3_json", "pythonic", "gemma4", "deepseek"] {
            XCTAssertNotNil(ToolParserManager.makeParser(named: name), "missing \(name)")
        }
        XCTAssertNil(ToolParserManager.makeParser(named: "nope"))
    }

    // MARK: Hermes buffered

    func testHermesBuffered() {
        let parser = HermesToolParser(idGenerator: seq())
        let info = parser.extractToolCalls(
            #"Sure. <tool_call>{"name":"lookup","arguments":{"q":"swift"}}</tool_call>"#
        )
        XCTAssertTrue(info.toolsCalled)
        XCTAssertEqual(info.content, "Sure. ")
        XCTAssertEqual(info.toolCalls, [ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"q":"swift"}"#)])
    }

    // MARK: Hermes streaming (canonical incremental algorithm)

    func testHermesStreamingEmitsNameThenArgDiffs() {
        let parser = HermesToolParser(idGenerator: seq())
        let chunks = [
            "<tool_call>",
            #"{"name": "get"#,
            #"_weather", "arguments": {"city": "#,
            #""London"}}"#,
            "</tool_call>",
        ]

        var previous = ""
        var streamedName: String?
        var streamedArgs = ""
        for chunk in chunks {
            let current = previous + chunk
            if let delta = parser.extractToolCallsStreaming(previousText: previous, currentText: current, deltaText: chunk) {
                for call in delta.toolCalls {
                    if let name = call.name { streamedName = name }
                    if let args = call.argumentsDelta { streamedArgs += args }
                }
            }
            previous = current
        }
        XCTAssertEqual(streamedName, "get_weather")
        // Argument fragments reassemble to the arguments object (sans the outer
        // brace that closes the tool-call object).
        XCTAssertTrue(streamedArgs.contains(#""city""#))
        XCTAssertTrue(streamedArgs.contains(#""London""#))
    }

    func testHermesStreamingLeadingContent() {
        let parser = HermesToolParser(idGenerator: seq())
        var previous = ""
        var content = ""
        for chunk in ["Let me check. ", "<tool_call>", #"{"name":"f","arguments":{}}"#, "</tool_call>"] {
            let current = previous + chunk
            if let delta = parser.extractToolCallsStreaming(previousText: previous, currentText: current, deltaText: chunk),
                let piece = delta.content {
                content += piece
            }
            previous = current
        }
        XCTAssertEqual(content, "Let me check. ")
    }

    // MARK: Llama JSON — `parameters` alias

    func testLlamaJSONParametersAlias() {
        let parser = Llama3JsonToolParser(idGenerator: seq())
        let info = parser.extractToolCalls(#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#)
        XCTAssertTrue(info.toolsCalled)
        XCTAssertEqual(info.toolCalls, [ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"city":"Paris"}"#)])
    }

    func testLlamaJSONWithPythonTag() {
        let parser = Llama3JsonToolParser(idGenerator: seq())
        let info = parser.extractToolCalls(#"<|python_tag|>{"name": "ping", "arguments": {"n": 1}}"#)
        XCTAssertTrue(info.toolsCalled)
        XCTAssertEqual(info.toolCalls, [ParsedToolCall(id: "call_0", name: "ping", arguments: #"{"n":1}"#)])
    }

    // MARK: Pythonic — `[func(k=v)]`

    func testPythonicSingleCall() {
        let parser = PythonicToolParser(idGenerator: seq())
        let info = parser.extractToolCalls(#"[get_weather(city="London", unit="celsius")]"#)
        XCTAssertTrue(info.toolsCalled)
        XCTAssertEqual(info.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"city":"London","unit":"celsius"}"#)
        ])
    }

    func testPythonicTypedAndListValues() {
        let parser = PythonicToolParser(idGenerator: seq())
        let info = parser.extractToolCalls(#"[search(limit=5, verbose=True, tags=["a", "b"])]"#)
        XCTAssertTrue(info.toolsCalled)
        XCTAssertEqual(info.toolCalls, [
            ParsedToolCall(id: "call_0", name: "search", arguments: #"{"limit":5,"tags":["a","b"],"verbose":true}"#)
        ])
    }

    func testPythonicMultipleCalls() {
        let parser = PythonicToolParser(idGenerator: seq())
        let info = parser.extractToolCalls(#"[a(x=1), b(y="z")]"#)
        XCTAssertTrue(info.toolsCalled)
        XCTAssertEqual(info.toolCalls, [
            ParsedToolCall(id: "call_0", name: "a", arguments: #"{"x":1}"#),
            ParsedToolCall(id: "call_1", name: "b", arguments: #"{"y":"z"}"#),
        ])
    }

    func testPythonicPythonStartWrapper() {
        let parser = PythonicToolParser(idGenerator: seq())
        let info = parser.extractToolCalls(#"<|python_start|>[f(a=1)]<|python_end|>"#)
        XCTAssertTrue(info.toolsCalled)
        XCTAssertEqual(info.toolCalls, [ParsedToolCall(id: "call_0", name: "f", arguments: #"{"a":1}"#)])
    }

    func testPythonicRejectsProse() {
        let parser = PythonicToolParser(idGenerator: seq())
        let info = parser.extractToolCalls("I cannot call that tool.")
        XCTAssertFalse(info.toolsCalled)
    }

    // MARK: DeepSeek — R1/V3 fenced

    func testDeepSeekR1Fenced() {
        let parser = DeepSeekToolParser(idGenerator: seq())
        let text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            + "```json\n{\"city\": \"Seoul\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        let info = parser.extractToolCalls(text)
        XCTAssertTrue(info.toolsCalled)
        XCTAssertEqual(info.toolCalls, [ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"city":"Seoul"}"#)])
    }

    func testDeepSeekMultipleCalls() {
        let parser = DeepSeekToolParser(idGenerator: seq())
        let text = "<｜tool▁calls▁begin｜>"
            + "<｜tool▁call▁begin｜>function<｜tool▁sep｜>a\n```json\n{\"x\": 1}\n```<｜tool▁call▁end｜>"
            + "<｜tool▁call▁begin｜>function<｜tool▁sep｜>b\n```json\n{\"y\": 2}\n```<｜tool▁call▁end｜>"
            + "<｜tool▁calls▁end｜>"
        let info = parser.extractToolCalls(text)
        XCTAssertEqual(info.toolCalls, [
            ParsedToolCall(id: "call_0", name: "a", arguments: #"{"x":1}"#),
            ParsedToolCall(id: "call_1", name: "b", arguments: #"{"y":2}"#),
        ])
    }

    // MARK: Base-class streaming default (emit-complete-call)

    func testGemmaStreamingEmitsOnCompletion() {
        let parser = Gemma4ToolParser(idGenerator: seq())
        var previous = ""
        var name: String?
        for chunk in ["<|tool_call>call:ping{", "n:<|\"|>1<|\"|>}", "<tool_call|>"] {
            let current = previous + chunk
            if let delta = parser.extractToolCallsStreaming(previousText: previous, currentText: current, deltaText: chunk) {
                for call in delta.toolCalls where call.name != nil { name = call.name }
            }
            previous = current
        }
        XCTAssertEqual(name, "ping")
    }

    // MARK: Helpers

    private func seq() -> () -> String {
        var n = 0
        return { defer { n += 1 }; return "call_\(n)" }
    }
}
