import Foundation

private let toolCallOpenTag = "<tool_call>"
private let toolCallCloseTag = "</tool_call>"
// Qwen2.5-Coder emits the same `{name, arguments}` payload as Hermes but wraps
// it in `<function_call>…</function_call>` instead of `<tool_call>`.
private let functionCallOpenTag = "<function_call>"
private let functionCallCloseTag = "</function_call>"
private let functionOpenTagPrefix = "<function="
private let functionCloseTag = "</function>"
private let parameterOpenTagPrefix = "<parameter="
private let parameterCloseTag = "</parameter>"
private let toolsOpenTag = "<tools>"
private let toolsCloseTag = "</tools>"
private let llamaPythonTagPrefix = "<|python_tag|>"
private let toolCallsPrefix = "[TOOL_CALLS]"

public struct ParsedToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    public let arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public struct ToolCallParseResult: Sendable, Equatable {
    public let content: String
    public let toolCalls: [ParsedToolCall]

    public init(content: String, toolCalls: [ParsedToolCall]) {
        self.content = content
        self.toolCalls = toolCalls
    }
}

public func parseToolCalls(
    from text: String,
    idGenerator: () -> String = { "call_" + String(UUID().uuidString.prefix(8)).lowercased() }
) -> ToolCallParseResult {
    if text.contains(toolCallOpenTag) {
        return parseTaggedToolCalls(
            from: text, openTag: toolCallOpenTag, closeTag: toolCallCloseTag, idGenerator: idGenerator
        )
    }
    if text.contains(functionCallOpenTag) {
        return parseTaggedToolCalls(
            from: text, openTag: functionCallOpenTag, closeTag: functionCallCloseTag, idGenerator: idGenerator
        )
    }

    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if text.contains(functionOpenTagPrefix) {
        let result = parseBareXMLFunctionToolCalls(from: text, idGenerator: idGenerator)
        if !result.toolCalls.isEmpty {
            return result
        }
    }

    if trimmed.hasPrefix(llamaPythonTagPrefix) {
        let remainder = String(trimmed.dropFirst(llamaPythonTagPrefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return parseBareJSONToolCalls(from: remainder, originalText: text, idGenerator: idGenerator)
    }
    if trimmed.hasPrefix(toolCallsPrefix) {
        let remainder = String(trimmed.dropFirst(toolCallsPrefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return parseBareJSONToolCalls(from: remainder, originalText: text, idGenerator: idGenerator)
    }

    let bareJSONResult = parseBareJSONToolCalls(from: trimmed, originalText: text, idGenerator: idGenerator)
    if !bareJSONResult.toolCalls.isEmpty {
        return bareJSONResult
    }

    if let toolsInnerText = toolsWrappedInnerText(from: trimmed) {
        return parseBareJSONToolCalls(from: toolsInnerText, originalText: text, idGenerator: idGenerator)
    }

    return bareJSONResult
}

private func parseTaggedToolCalls(
    from text: String,
    openTag: String,
    closeTag: String,
    idGenerator: () -> String
) -> ToolCallParseResult {
    var cursor = text.startIndex
    var contentParts: [String] = []
    var toolCalls: [ParsedToolCall] = []

    while let openRange = text[cursor...].range(of: openTag) {
        guard let closeRange = text[openRange.upperBound...].range(of: closeTag) else {
            contentParts.append(String(text[cursor...]))
            cursor = text.endIndex
            break
        }

        contentParts.append(String(text[cursor..<openRange.lowerBound]))

        let innerText = String(text[openRange.upperBound..<closeRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let toolCall = parseToolCallObject(from: innerText, idGenerator: idGenerator) {
            toolCalls.append(toolCall)
        } else if let xmlToolCalls = parseXMLFunctionToolCalls(from: innerText, idGenerator: idGenerator) {
            toolCalls.append(contentsOf: xmlToolCalls)
        } else {
            contentParts.append(String(text[openRange.lowerBound..<closeRange.upperBound]))
        }

        cursor = closeRange.upperBound
    }

    if cursor < text.endIndex {
        contentParts.append(String(text[cursor...]))
    }

    guard !toolCalls.isEmpty else {
        return ToolCallParseResult(content: text, toolCalls: [])
    }

    return ToolCallParseResult(
        content: contentParts.joined().trimmingCharacters(in: .whitespacesAndNewlines),
        toolCalls: toolCalls
    )
}

private func parseBareJSONToolCalls(
    from text: String,
    originalText: String,
    idGenerator: () -> String
) -> ToolCallParseResult {
    guard let value = parseJSONValue(from: text) else {
        return ToolCallParseResult(content: originalText, toolCalls: [])
    }

    if let object = value as? [String: Any],
        let payload = parseToolCallPayload(from: object)
    {
        return ToolCallParseResult(content: "", toolCalls: [payload.toolCall(id: idGenerator())])
    }

    if let array = value as? [[String: Any]], !array.isEmpty {
        let payloads = array.compactMap(parseToolCallPayload)
        if payloads.count == array.count {
            let toolCalls = payloads.map { $0.toolCall(id: idGenerator()) }
            return ToolCallParseResult(content: "", toolCalls: toolCalls)
        }
    }

    return ToolCallParseResult(content: originalText, toolCalls: [])
}

private func parseBareXMLFunctionToolCalls(
    from text: String,
    idGenerator: () -> String
) -> ToolCallParseResult {
    var cursor = text.startIndex
    var contentParts: [String] = []
    var toolCalls: [ParsedToolCall] = []

    while let openRange = text[cursor...].range(of: functionOpenTagPrefix) {
        guard let tagEnd = text[openRange.upperBound...].firstIndex(of: ">"),
            let closeRange = text[tagEnd...].range(of: functionCloseTag)
        else {
            break
        }

        contentParts.append(String(text[cursor..<openRange.lowerBound]))
        let functionText = String(text[openRange.lowerBound..<closeRange.upperBound])
        if let payload = parseXMLFunctionPayload(from: functionText) {
            toolCalls.append(payload.toolCall(id: idGenerator()))
        } else {
            contentParts.append(functionText)
        }
        cursor = closeRange.upperBound
    }

    if cursor < text.endIndex {
        contentParts.append(String(text[cursor...]))
    }

    guard !toolCalls.isEmpty else {
        return ToolCallParseResult(content: text, toolCalls: [])
    }

    return ToolCallParseResult(
        content: contentParts.joined().trimmingCharacters(in: .whitespacesAndNewlines),
        toolCalls: toolCalls
    )
}

private func parseXMLFunctionToolCalls(
    from text: String,
    idGenerator: () -> String
) -> [ParsedToolCall]? {
    guard text.contains(functionOpenTagPrefix) else {
        return nil
    }

    var cursor = text.startIndex
    var toolCalls: [ParsedToolCall] = []
    while let openRange = text[cursor...].range(of: functionOpenTagPrefix) {
        guard let tagEnd = text[openRange.upperBound...].firstIndex(of: ">"),
            let closeRange = text[tagEnd...].range(of: functionCloseTag)
        else {
            break
        }

        let functionText = String(text[openRange.lowerBound..<closeRange.upperBound])
        if let payload = parseXMLFunctionPayload(from: functionText) {
            toolCalls.append(payload.toolCall(id: idGenerator()))
        }
        cursor = closeRange.upperBound
    }

    return toolCalls.isEmpty ? nil : toolCalls
}

private func parseToolCallObject(
    from text: String,
    idGenerator: () -> String
) -> ParsedToolCall? {
    guard let object = parseJSONValue(from: text) as? [String: Any] else {
        return nil
    }
    return parseToolCallObject(from: object, idGenerator: idGenerator)
}

private func parseToolCallObject(
    from object: [String: Any],
    idGenerator: () -> String
) -> ParsedToolCall? {
    guard let payload = parseToolCallPayload(from: object) else {
        return nil
    }
    return payload.toolCall(id: idGenerator())
}

private struct ToolCallPayload {
    let name: String
    let arguments: String

    func toolCall(id: String) -> ParsedToolCall {
        ParsedToolCall(id: id, name: name, arguments: arguments)
    }
}

private func parseToolCallPayload(from object: [String: Any]) -> ToolCallPayload? {
    guard let name = object["name"] as? String, !name.isEmpty else {
        return nil
    }

    // Llama 3 emits the tool-input object under `parameters`, while the
    // OpenAI/Hermes shape calls it `arguments`. Preserve either spelling in
    // the broad bare-JSON parser; otherwise it claims the call before the
    // Llama-specific fallback and silently replaces valid arguments with `{}`.
    let arguments = argumentsJSONString(from: object["arguments"] ?? object["parameters"]) ?? "{}"
    return ToolCallPayload(name: name, arguments: arguments)
}

private func parseXMLFunctionPayload(from text: String) -> ToolCallPayload? {
    guard let openRange = text.range(of: functionOpenTagPrefix),
        let tagEnd = text[openRange.upperBound...].firstIndex(of: ">")
    else {
        return nil
    }

    let name = String(text[openRange.upperBound..<tagEnd])
        .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !name.isEmpty else {
        return nil
    }

    let bodyEnd = text.range(of: functionCloseTag, range: tagEnd..<text.endIndex)?.lowerBound ?? text.endIndex
    let body = String(text[text.index(after: tagEnd)..<bodyEnd])
    let arguments = xmlFunctionArgumentsJSONString(from: body) ?? "{}"
    return ToolCallPayload(name: name, arguments: arguments)
}

private func xmlFunctionArgumentsJSONString(from text: String) -> String? {
    var cursor = text.startIndex
    var arguments: [String: Any] = [:]

    while let openRange = text[cursor...].range(of: parameterOpenTagPrefix) {
        guard let tagEnd = text[openRange.upperBound...].firstIndex(of: ">"),
            let closeRange = text[tagEnd...].range(of: parameterCloseTag)
        else {
            break
        }

        let key = String(text[openRange.upperBound..<tagEnd])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !key.isEmpty {
            let rawValue = String(text[text.index(after: tagEnd)..<closeRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            arguments[key] = xmlParameterValue(from: rawValue)
        }
        cursor = closeRange.upperBound
    }

    return argumentsJSONString(from: arguments)
}

private func xmlParameterValue(from text: String) -> Any {
    parseJSONFragment(from: text) ?? text
}

private func toolsWrappedInnerText(from text: String) -> String? {
    guard text.hasPrefix(toolsOpenTag), text.hasSuffix(toolsCloseTag) else {
        return nil
    }

    let innerStart = text.index(text.startIndex, offsetBy: toolsOpenTag.count)
    let innerEnd = text.index(text.endIndex, offsetBy: -toolsCloseTag.count)
    guard innerStart <= innerEnd else {
        return nil
    }

    return String(text[innerStart..<innerEnd])
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

private func argumentsJSONString(from value: Any?) -> String? {
    guard let value else {
        return "{}"
    }
    if let string = value as? String {
        return string
    }
    guard JSONSerialization.isValidJSONObject(value),
        let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
        let string = String(data: data, encoding: .utf8)
    else {
        return nil
    }
    return string
}

private func parseJSONValue(from text: String) -> Any? {
    guard let data = text.data(using: .utf8), !data.isEmpty else {
        return nil
    }
    return try? JSONSerialization.jsonObject(with: data)
}

private func parseJSONFragment(from text: String) -> Any? {
    guard let data = text.data(using: .utf8), !data.isEmpty else {
        return nil
    }
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}
