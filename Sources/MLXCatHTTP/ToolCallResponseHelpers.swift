import Foundation

public func buildAssistantMessageWithToolCalls(
    content: String,
    reasoning: String,
    parsed: ToolCallParseResult
) -> [String: Any] {
    var message: [String: Any] = [
        "role": "assistant",
        "content": content,
    ]
    if !parsed.toolCalls.isEmpty {
        message["content"] = parsed.content.isEmpty ? NSNull() : parsed.content
        message["tool_calls"] = toolCallResponseDictionaries(parsed.toolCalls)
    }
    if !reasoning.isEmpty {
        message["reasoning_content"] = reasoning
    }
    return message
}

public func toolCallResponseDictionaries(_ toolCalls: [ParsedToolCall]) -> [[String: Any]] {
    toolCalls.map { toolCall in
        let function: [String: Any] = [
            "name": toolCall.name,
            "arguments": toolCall.arguments,
        ]
        return [
            "id": toolCall.id,
            "type": "function",
            "function": function,
        ]
    }
}

public func toolCallDeltaDictionaries(from parsed: ToolCallParseResult) -> [[String: Any]] {
    parsed.toolCalls.enumerated().map { index, toolCall in
        let function: [String: Any] = [
            "name": toolCall.name,
            "arguments": toolCall.arguments,
        ]
        return [
            "index": index,
            "id": toolCall.id,
            "type": "function",
            "function": function,
        ]
    }
}

public func finishReasonForToolCalls(
    defaultFinishReason: String,
    parsed: ToolCallParseResult
) -> String {
    parsed.toolCalls.isEmpty ? defaultFinishReason : "tool_calls"
}
