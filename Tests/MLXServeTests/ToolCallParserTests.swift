@testable import MLXServeHTTP
import XCTest

final class ToolCallParserTests: XCTestCase {
    func testSingleHermesBlock() {
        let result = parseToolCalls(
            from: #"<tool_call>{"name":"lookup","arguments":{"query":"swift"}}</tool_call>"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"query":"swift"}"#)
        ])
    }

    func testMultipleHermesBlocks() {
        let result = parseToolCalls(
            from: #"""
            <tool_call>{"name":"first","arguments":{"a":1}}</tool_call>
            <tool_call>{"name":"second","arguments":{"b":2}}</tool_call>
            """#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "first", arguments: #"{"a":1}"#),
            ParsedToolCall(id: "call_1", name: "second", arguments: #"{"b":2}"#),
        ])
    }

    func testHermesBlockWithLeadingProsePreservesProse() {
        let result = parseToolCalls(
            from: #"I'll check that. <tool_call>{"name":"lookup","arguments":{"query":"swift"}}</tool_call>"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "I'll check that.")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"query":"swift"}"#)
        ])
    }

    func testFunctionCallTagQwenCoder() {
        // Qwen2.5-Coder wraps the Hermes payload in <function_call> instead.
        let result = parseToolCalls(
            from: "<function_call>\n  {\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\", \"unit\": \"celsius\"}}\n</function_call>",
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"city":"London","unit":"celsius"}"#)
        ])
    }

    func testFunctionCallTagWithLeadingProse() {
        let result = parseToolCalls(
            from: #"On it. <function_call>{"name":"lookup","arguments":{"q":"x"}}</function_call>"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "On it.")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"q":"x"}"#)
        ])
    }

    func testBareJSONObject() {
        let result = parseToolCalls(
            from: #"{"name":"lookup","arguments":{"query":"swift"}}"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"query":"swift"}"#)
        ])
    }

    func testBareLlamaJSONObjectPreservesParametersAlias() {
        let result = parseToolCalls(
            from: #"{"name":"get_weather","parameters":{"city":"London","unit":"celsius"}}"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(
                id: "call_0",
                name: "get_weather",
                arguments: #"{"city":"London","unit":"celsius"}"#
            )
        ])
    }

    func testBareJSONArray() {
        let result = parseToolCalls(
            from: #"[{"name":"first","arguments":{"a":1}},{"name":"second","arguments":{"b":2}}]"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "first", arguments: #"{"a":1}"#),
            ParsedToolCall(id: "call_1", name: "second", arguments: #"{"b":2}"#),
        ])
    }

    func testLlamaPythonTagPrefix() {
        let result = parseToolCalls(
            from: #"<|python_tag|>{"name":"lookup","arguments":{"query":"swift"}}"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"query":"swift"}"#)
        ])
    }

    func testArgumentsObjectIsReserializedWithSortedKeys() {
        let result = parseToolCalls(
            from: #"{"name":"lookup","arguments":{"z":1,"a":2}}"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"a":2,"z":1}"#)
        ])
    }

    func testArgumentsStringIsPassedThrough() {
        let result = parseToolCalls(
            from: #"{"name":"lookup","arguments":"{\"z\":1,\"a\":2}"}"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: #"{"z":1,"a":2}"#)
        ])
    }

    func testMissingArgumentsDefaultsToEmptyObjectString() {
        let result = parseToolCalls(
            from: #"{"name":"lookup"}"#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "lookup", arguments: "{}")
        ])
    }

    func testQwenXMLInsideToolCallParsesSingleStringParameter() {
        let result = parseToolCalls(
            from: #"""
            <tool_call>
            <function=get_weather>
            <parameter=location>
            Paris
            </parameter>
            </function>
            </tool_call>
            """#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"location":"Paris"}"#)
        ])
    }

    func testQwenXMLInsideToolCallCoercesNumericParameter() {
        let result = parseToolCalls(
            from: #"""
            <tool_call>
            <function=get_weather>
            <parameter=count>
            3
            </parameter>
            </function>
            </tool_call>
            """#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"count":3}"#)
        ])
    }

    func testQwenXMLInsideToolCallSortsMultipleParameters() {
        let result = parseToolCalls(
            from: #"""
            <tool_call>
            <function=get_weather>
            <parameter=units>
            celsius
            </parameter>
            <parameter=location>
            Paris
            </parameter>
            </function>
            </tool_call>
            """#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(
                id: "call_0",
                name: "get_weather",
                arguments: #"{"location":"Paris","units":"celsius"}"#
            )
        ])
    }

    func testBareFunctionWithoutToolCallWrapper() {
        let result = parseToolCalls(
            from: #"""
            Let me check.
            <function=get_weather>
            <parameter=location>Paris</parameter>
            </function>
            """#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "Let me check.")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"location":"Paris"}"#)
        ])
    }

    func testToolsWrappedJSONObject() {
        let result = parseToolCalls(
            from: #"""
            <tools>
              { "name": "get_weather", "arguments": { "location": "Paris" } }
            </tools>
            """#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "get_weather", arguments: #"{"location":"Paris"}"#)
        ])
    }

    func testToolsWrappedJSONWithoutNameIsContent() {
        let text = #"<tools>{"arguments":{"location":"Paris"}}</tools>"#
        let result = parseToolCalls(from: text, idGenerator: idGenerator())

        XCTAssertEqual(result.content, text)
        XCTAssertEqual(result.toolCalls, [])
    }

    func testMultipleBareFunctionSections() {
        let result = parseToolCalls(
            from: #"""
            <function=first>
            <parameter=a>1</parameter>
            </function>
            <function=second>
            <parameter=b>2</parameter>
            </function>
            """#,
            idGenerator: idGenerator()
        )

        XCTAssertEqual(result.content, "")
        XCTAssertEqual(result.toolCalls, [
            ParsedToolCall(id: "call_0", name: "first", arguments: #"{"a":1}"#),
            ParsedToolCall(id: "call_1", name: "second", arguments: #"{"b":2}"#),
        ])
    }

    func testMalformedJSONInsideHermesBlockIsContent() {
        let text = #"<tool_call>{"name":"lookup","arguments":</tool_call>"#
        let result = parseToolCalls(from: text, idGenerator: idGenerator())

        XCTAssertEqual(result.content, text)
        XCTAssertEqual(result.toolCalls, [])
    }

    func testPlainTextWithoutToolCallIsReturnedVerbatim() {
        let text = "  plain text\nwith spacing  "
        let result = parseToolCalls(from: text, idGenerator: idGenerator())

        XCTAssertEqual(result.content, text)
        XCTAssertEqual(result.toolCalls, [])
    }

    func testJSONObjectWithoutNameIsContent() {
        let text = #"{"arguments":{"query":"swift"}}"#
        let result = parseToolCalls(from: text, idGenerator: idGenerator())

        XCTAssertEqual(result.content, text)
        XCTAssertEqual(result.toolCalls, [])
    }

    func testHermesBlockWithoutNameIsContent() {
        let text = #"<tool_call>{"arguments":{"query":"swift"}}</tool_call>"#
        let result = parseToolCalls(from: text, idGenerator: idGenerator())

        XCTAssertEqual(result.content, text)
        XCTAssertEqual(result.toolCalls, [])
    }

    func testToolCallBlockWithoutJSONOrFunctionIsContent() {
        let text = #"<tool_call>please call lookup</tool_call>"#
        let result = parseToolCalls(from: text, idGenerator: idGenerator())

        XCTAssertEqual(result.content, text)
        XCTAssertEqual(result.toolCalls, [])
    }

    private func idGenerator() -> () -> String {
        var nextID = 0
        return {
            defer { nextID += 1 }
            return "call_\(nextID)"
        }
    }
}
