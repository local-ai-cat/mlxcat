import Foundation
import MLXCatHTTP
import MLXLMCommon
import MLXVLM

/// Keeps tool history structured until the model's own message generator renders it.
enum NativeChatHistory {
    enum HistoryError: Error { case messageCountMismatch }

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
        guard family.contains("gemma-4") || family.contains("gemma4"),
              messages.contains(where: { !$0.toolCalls.isEmpty && $0.reasoningContent != nil }) else { return input }

        // Use Gemma's existing formatter, including its media placeholders and tool
        // metadata. Enrich only the reasoning field that Chat.Message cannot carry.
        var formatted = Gemma4MessageGenerator().generate(from: input)
        guard formatted.count == messages.count else { throw HistoryError.messageCountMismatch }
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
