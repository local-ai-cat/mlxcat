import Foundation
import Testing
import MLXCatHTTP
import MLXLMCommon
import MLXVLM
@testable import MLXCatNative

@Suite("Native structured tool history")
struct NativeChatHistoryTests {
    @Test("Generation failures retain an actionable localized description")
    func failureDescription() {
        #expect(NativeModelEngineError.generationFailed("image preparation failed").localizedDescription == "image preparation failed")
    }

    @Test("Cancellation remains distinct from normal completion")
    func cancellationIsNotStop() {
        #expect(openAIFinishReason(.cancelled) == "cancelled")
        #expect(openAIFinishReason(.length) == "length")
        #expect(openAIFinishReason(.stop) == "stop")
    }

    @Test("HTTP preserves null-content calls, reasoning and linked results")
    func parsesStructuredHistory() throws {
        let body = Data(#"{"model":"gemma-4-E2B","messages":[{"role":"user","content":"go"},{"role":"assistant","content":null,"reasoning_content":"inspect","tool_calls":[{"id":"a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"index.html\"}"}}]},{"role":"tool","tool_call_id":"a","content":"source"}]}"#.utf8)
        let request = try OpenAIChatRequest.parse(body)
        #expect(request.messages.count == 3)
        #expect(request.messages[1].reasoningContent == "inspect")
        #expect(request.messages[1].toolCalls.first?.id == "a")
        #expect(request.messages[2].toolCallID == "a")

        let assistant = Chat.Message(role: .assistant, content: "", tool: try NativeChatHistory.toolMetadata(for: request.messages[1]))
        let result = Chat.Message(role: .tool, content: "source", tool: try NativeChatHistory.toolMetadata(for: request.messages[2]))
        let rendered = Gemma4MessageGenerator().generate(messages: [assistant, result])
        let calls = try #require(rendered[0]["tool_calls"] as? [[String: any Sendable]])
        #expect(calls[0]["id"] as? String == "a")
        #expect(rendered[1]["tool_call_id"] as? String == "a")
    }

    @Test("Gemma keeps current tool reasoning and media, without restoring old-turn thoughts")
    func preservesCurrentReasoningAndMedia() throws {
        let call = OpenAIChatToolCall(id: "a", name: "read_file", arguments: "{}")
        let messages = [
            OpenAIChatMessage(role: "assistant", content: "", reasoningContent: "old", toolCalls: [call]),
            OpenAIChatMessage(role: "user", content: "go"),
            OpenAIChatMessage(role: "assistant", content: "", reasoningContent: "current", toolCalls: [call])
        ]
        var chat = try messages.map {
            Chat.Message(role: try #require(Chat.Message.Role(rawValue: $0.role)), content: $0.content,
                         tool: try NativeChatHistory.toolMetadata(for: $0))
        }
        chat[1].images = [.url(URL(fileURLWithPath: "/fixture/frame.png"))]
        let input = UserInput(chat: chat)
        let enriched = try NativeChatHistory.preservingReasoning(in: input, messages: messages, modelID: "gemma-4-E2B")
        guard case .messages(let rendered) = enriched.prompt else {
            Issue.record("Expected model-formatted messages")
            return
        }
        #expect(rendered[0]["reasoning_content"] == nil)
        #expect(rendered[2]["reasoning_content"] as? String == "current")
        #expect(rendered[2]["tool_calls"] != nil)
        #expect(enriched.images.count == 1)
        let unchanged = try NativeChatHistory.preservingReasoning(in: input, messages: messages, modelID: "Qwen3.5-4B")
        guard case .chat = unchanged.prompt else {
            Issue.record("Qwen must retain its own formatter")
            return
        }
    }

    @Test("Malformed historical arguments fail instead of silently becoming an empty call")
    func rejectsMalformedArguments() {
        let message = OpenAIChatMessage(role: "assistant", content: "", toolCalls: [.init(id: "a", name: "edit", arguments: "{")])
        #expect(throws: (any Error).self) { try NativeChatHistory.toolMetadata(for: message) }
    }
}
