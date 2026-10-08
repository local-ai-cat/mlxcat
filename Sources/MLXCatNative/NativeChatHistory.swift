import Foundation
import MLXCatHTTP
import MLXLMCommon
import MLXVLM

/// Keeps tool history structured until the model's own message generator renders it.
enum NativeChatHistory {
    enum HistoryError: Error, LocalizedError {
        case messageCountMismatch
        case assistantMediaWithToolCalls

        var errorDescription: String? {
            switch self {
            case .messageCountMismatch: return "Model history formatting changed the message count."
            case .assistantMediaWithToolCalls:
                return "Gemma cannot safely combine assistant images with tool calls in the same message."
            }
        }
    }

    static func toolMetadata(for message: OpenAIChatMessage) throws -> Chat.Message.Tool? {
        if !message.toolCalls.isEmpty {
            return .calls(try message.toolCalls.map { call in
                let arguments = try JSONDecoder().decode([String: JSONValue].self, from: Data(call.arguments.utf8))
                return ToolCall(function: .init(name: call.name, arguments: arguments), id: call.id)
            })
        }
        if let id = message.toolCallID {
            return .result(id: id, name: message.toolName)
        }
        return nil
    }

    static func preservingReasoning(
        in input: UserInput, messages: [OpenAIChatMessage], modelID: String
    ) throws -> UserInput {
        let family = modelID.lowercased().replacingOccurrences(of: "_", with: "-")
        guard family.contains("gemma-4") || family.contains("gemma4") else { return input }
        // The Gemma template renders linked tool results before assistant content,
        // while tensors follow message order. Reject this unsupported mixed shape
        // rather than silently assigning screenshots to the wrong placeholders.
        guard !messages.contains(where: {
            $0.role == "assistant" && !$0.toolCalls.isEmpty && !$0.imageReferences.isEmpty
        }) else { throw HistoryError.assistantMediaWithToolCalls }
        let hasReasoning = messages.contains { !$0.toolCalls.isEmpty && $0.reasoningContent != nil }
        let hasToolImages = messages.contains { $0.role == "tool" && !$0.imageReferences.isEmpty }
        guard hasReasoning || hasToolImages else { return input }

        // Use Gemma's existing formatter, including its media placeholders and tool
        // metadata. Enrich only the reasoning field that Chat.Message cannot carry.
        var formatted = Gemma4MessageGenerator().generate(from: input)
        guard formatted.count == messages.count else { throw HistoryError.messageCountMismatch }
        // Gemma's OpenAI tool-response template concatenates only text parts.
        // Keep image placeholders in those text parts so the processor's media
        // tensors still have matching prompt positions. No image is moved into
        // a synthetic user turn (which would discard current tool reasoning).
        for index in formatted.indices where messages[index].role == "tool" {
            guard let parts = formatted[index]["content"] as? [[String: any Sendable]] else { continue }
            formatted[index]["content"] = parts.map { part -> [String: any Sendable] in
                if part["type"] as? String == "image" {
                    return ["type": "text", "text": "<|image|>"]
                }
                return part
            }
        }
        let lastUser = messages.lastIndex(where: { $0.role == "user" }) ?? -1
        for index in messages.indices where index > lastUser {
            let message = messages[index]
            if message.role == "assistant", !message.toolCalls.isEmpty,
               let reasoning = message.reasoningContent, !reasoning.isEmpty {
                formatted[index]["reasoning_content"] = reasoning
            }
        }
        var result = input
        // Changing .chat to .messages preserves input.images/videos/audios; the
        // model processor still owns media decoding and tensor construction.
        result.prompt = .messages(formatted)
        return result
    }
}
