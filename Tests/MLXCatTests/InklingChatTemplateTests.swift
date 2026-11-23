import Testing

@testable import MLXCatNative

struct InklingChatTemplateTests {
    @Test func rendersToolDeclarationAndTurnInThePublisherFormat() throws {
        let function: [String: any Sendable] = ["name": "find", "arguments": ["id": 42]]
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "Use tools."],
            ["role": "user", "content": "Find it."],
            ["role": "assistant", "content": "", "reasoning_content": "I can look it up.",
             "tool_calls": [["id": "call-1", "function": function] as [String: any Sendable]]],
            ["role": "tool", "tool_call_id": "call-1", "content": "Done"],
        ]
        let result = try InklingChatTemplate.render(
            messages: messages, tools: [["name": "find"]], reasoningEffort: "medium")
        #expect(result ==
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"\",\"name\":\"find\",\"parameters\":{},\"type\":\"function\"}]<|end_message|>"
            + "<|message_system|><|content_text|>Use tools.<|end_message|>"
            + "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>"
            + "<|message_user|><|content_text|>Find it.<|end_message|>"
            + "<|message_model|><|content_thinking|>I can look it up.<|end_message|>"
            + "<|message_model|>find<|content_invoke_tool_json|>{\"name\":\"find\",\"args\":{\"id\":42}}<|end_message|>"
            + "<|content_model_end_sampling|><|message_tool|>find<|content_text|>Done<|end_message|>")
    }
}
