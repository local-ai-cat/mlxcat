import MLXCatHTTP
import Testing

struct InklingOutputParserTests {
    @Test func streamsThinkingAndTextAcrossEveryMarkerBoundary() {
        var parser = ThinkingParser()
        let raw = "<|message_model|><|content_thinking|>working<|end_message|>"
            + "<|message_model|><|content_text|>answer<|end_message|>"
        var reasoning = ""
        var content = ""
        for character in raw {
            let delta = parser.feed(String(character))
            reasoning += delta.reasoning
            content += delta.content
        }
        let tail = parser.finish()
        #expect(reasoning + tail.reasoning == "working")
        #expect(content + tail.content == "answer")
    }

    @Test func extractsArgumentsWithoutInterpretingMarkersInsideJSON() {
        let raw = """
            <|message_model|>lookup<|content_invoke_tool_json|>{"name":"lookup","args":{"id":9007199254740993,"text":"<|end_message|>"}}<|end_message|>
            """
        let parsed = parseModelOutput(raw, model: "Inkling-2-Mini", idGenerator: { "tool-1" })
        #expect(parsed.content.isEmpty)
        #expect(parsed.toolCalls == [
            ParsedToolCall(id: "tool-1", name: "lookup", arguments: """
                {"id":9007199254740993,"text":"<|end_message|>"}
                """)
        ])
    }
}
