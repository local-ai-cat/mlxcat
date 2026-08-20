import Foundation

/// Integration layer over the `ToolParser`/`ToolParserManager` family parsers.
///
/// The response paths need one call that returns reasoning + visible content +
/// tool calls for a given model. This selects the family (by model id, then by
/// output markers), splits thinking from content, and runs the family's
/// `ToolParser`. gpt-oss harmony is special-cased here (as vLLM special-cases it
/// via HarmonyParser): its tool call lives in the `commentary` channel that the
/// thinking split routes to reasoning, so it needs the raw text and a channel
/// walk rather than a content-string parser.

public enum ToolCallModelFamily: String, Sendable, Equatable {
    case generic
    case harmony
    case gemma4
    case deepseek
}

/// Reasoning, visible content, and tool calls extracted from one completion.
public struct ModelOutputParseResult: Sendable, Equatable {
    public let reasoning: String
    public let content: String
    public let toolCalls: [ParsedToolCall]

    public init(reasoning: String, content: String, toolCalls: [ParsedToolCall]) {
        self.reasoning = reasoning
        self.content = content
        self.toolCalls = toolCalls
    }
}

public func defaultToolCallID() -> String {
    "call_" + String(UUID().uuidString.prefix(8)).lowercased()
}

// MARK: - Family selection

/// Family by model id first, then by native output markers (so an aliased or
/// locally-renamed checkpoint still routes correctly).
public func toolCallModelFamily(forModel model: String, output text: String = "") -> ToolCallModelFamily {
    let id = model.lowercased()
    if id.contains("gpt-oss") || id.contains("gpt_oss") { return .harmony }
    if id.contains("gemma-4") || id.contains("gemma4") { return .gemma4 }
    if id.contains("deepseek") { return .deepseek }

    if text.contains(harmonyChannelToken), text.contains(harmonyFunctionsRecipientPrefix) {
        return .harmony
    }
    if text.contains(deepseekCallsBeginMarker) { return .deepseek }
    if text.contains(gemma4ToolCallOpen), text.contains(gemma4CallMarker) { return .gemma4 }
    return .generic
}

private let deepseekCallsBeginMarker = "<｜tool▁calls▁begin｜>"
let gemma4ToolCallOpen = "<|tool_call>"
let gemma4CallMarker = "call:"

// MARK: - Buffered entry point

/// Parse a full completion into reasoning/content/tool-calls using the family
/// parser for `model`.
///
/// - Parameter includeToolCalls: when `false` (tools not requested), tool
///   extraction is skipped and the result mirrors a plain `extractThinking`
///   split.
/// - Parameter startInThinking: begin in the reasoning channel when the chat
///   template consumed the opening `<think>` marker.
public func parseModelOutput(
    _ text: String,
    model: String,
    includeToolCalls: Bool = true,
    startInThinking: Bool = false,
    idGenerator: @escaping () -> String = defaultToolCallID
) -> ModelOutputParseResult {
    let family = toolCallModelFamily(forModel: model, output: text)

    if family == .harmony {
        return parseHarmonyOutput(text, includeToolCalls: includeToolCalls, idGenerator: idGenerator)
    }

    let extracted = extractThinking(text, startInThinking: startInThinking)
    guard includeToolCalls else {
        return ModelOutputParseResult(reasoning: extracted.reasoning, content: extracted.content, toolCalls: [])
    }

    let info = extractContentToolCalls(family: family, content: extracted.content, idGenerator: idGenerator)
    let content = info.toolsCalled ? (info.content ?? "") : extracted.content
    return ModelOutputParseResult(
        reasoning: extracted.reasoning,
        content: content.trimmingCharacters(in: .whitespacesAndNewlines),
        toolCalls: info.toolCalls
    )
}

/// Run the family's content-channel `ToolParser`, falling back to the generic
/// sequence so no model regresses (a family parser that finds nothing defers to
/// the broad Hermes/XML/JSON/pythonic handling).
private func extractContentToolCalls(
    family: ToolCallModelFamily,
    content: String,
    idGenerator: @escaping () -> String
) -> ExtractedToolCallInformation {
    switch family {
    case .gemma4:
        if let info = runParser("gemma4", content, idGenerator), info.toolsCalled { return info }
    case .deepseek:
        if let info = runParser("deepseek", content, idGenerator), info.toolsCalled { return info }
    case .harmony, .generic:
        break
    }
    return genericExtract(content, idGenerator: idGenerator)
}

/// Generic (Qwen/Hermes/Llama/Mistral) extraction: the broad legacy parser
/// first (Hermes `<tool_call>`, bare XML `<function=>`, `<|python_tag|>`,
/// `[TOOL_CALLS]`, bare JSON), then the pythonic `[func()]` form (Llama-3.2),
/// then the Llama JSON `parameters`-alias form.
private func genericExtract(
    _ content: String,
    idGenerator: @escaping () -> String
) -> ExtractedToolCallInformation {
    let legacy = parseToolCalls(from: content, idGenerator: idGenerator)
    if !legacy.toolCalls.isEmpty {
        return ExtractedToolCallInformation(
            toolsCalled: true,
            toolCalls: legacy.toolCalls,
            content: legacy.content.isEmpty ? nil : legacy.content
        )
    }
    for name in ["pythonic", "llama3_json"] {
        if let info = runParser(name, content, idGenerator), info.toolsCalled { return info }
    }
    return .noTools(content)
}

private func runParser(
    _ name: String,
    _ content: String,
    _ idGenerator: @escaping () -> String
) -> ExtractedToolCallInformation? {
    ToolParserManager.makeParser(named: name, idGenerator: idGenerator)?.extractToolCalls(content)
}

// MARK: - Streaming finish-time entry point

/// Streaming finish-time tool-call extraction for the Responses / Anthropic
/// dialects. They already streamed reasoning/content deltas through the
/// (family-agnostic) incremental `ThinkingParser`; at finish they need the tool
/// calls. Generic models parse the visible content channel; non-generic
/// families re-parse the retained raw text (their markers may sit outside the
/// content channel, e.g. harmony `commentary`).
public func streamingToolCallParse(
    model: String,
    rawText: String,
    bufferedContent: String,
    idGenerator: @escaping () -> String = defaultToolCallID
) -> ToolCallParseResult {
    switch toolCallModelFamily(forModel: model, output: rawText) {
    case .generic:
        let info = genericExtract(bufferedContent, idGenerator: idGenerator)
        return ToolCallParseResult(
            content: info.toolsCalled ? (info.content ?? "") : bufferedContent,
            toolCalls: info.toolCalls
        )
    case .harmony, .gemma4, .deepseek:
        let result = parseModelOutput(rawText, model: model, includeToolCalls: true, idGenerator: idGenerator)
        return ToolCallParseResult(content: result.content, toolCalls: result.toolCalls)
    }
}

// MARK: - Harmony (gpt-oss) channel walk

let harmonyChannelToken = "<|channel|>"
private let harmonyMessageToken = "<|message|>"
private let harmonyBoundaryTokens = ["<|end|>", "<|return|>", "<|call|>", "<|start|>", harmonyChannelToken]
let harmonyFunctionsRecipientPrefix = "to=functions."

/// Walk harmony channels: `analysis`/`commentary` → reasoning, `final` →
/// content, and any message whose header carries `to=functions.NAME` → a tool
/// call (the `<|message|>` body is the JSON arguments). Mirrors omlx's
/// `parse_tool_calls_from_tokens` + vLLM's HarmonyParser channel routing on
/// decoded text (gpt-oss tool parsing is delegated, not a content-string parse).
private func parseHarmonyOutput(
    _ text: String,
    includeToolCalls: Bool,
    idGenerator: @escaping () -> String
) -> ModelOutputParseResult {
    var reasoning = ""
    var content = ""
    var toolCalls: [ParsedToolCall] = []

    var searchStart = text.startIndex
    while let channelRange = text.range(of: harmonyChannelToken, range: searchStart..<text.endIndex) {
        guard let messageRange = text.range(
            of: harmonyMessageToken,
            range: channelRange.upperBound..<text.endIndex
        ) else {
            break
        }

        let channelHeader = String(text[channelRange.upperBound..<messageRange.lowerBound])
        let channel = channelHeader
            .split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" })
            .first
            .map(String.init) ?? ""

        let headerRegion = String(text[searchStart..<messageRange.lowerBound])
        let recipient = harmonyFunctionsRecipient(in: headerRegion)

        let bodyStart = messageRange.upperBound
        let bodyEnd = harmonyBodyEnd(in: text, from: bodyStart)
        let body = String(text[bodyStart..<bodyEnd])

        if includeToolCalls, let name = recipient {
            toolCalls.append(
                ParsedToolCall(id: idGenerator(), name: name, arguments: normalizedArgumentsJSON(from: body))
            )
        } else if channel == "final" {
            content += body
        } else {
            reasoning += body
        }

        searchStart = bodyEnd
    }

    var result = ModelOutputParseResult(
        reasoning: reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
        content: content.trimmingCharacters(in: .whitespacesAndNewlines),
        toolCalls: toolCalls
    )

    // No harmony markers at all: fall back to the generic split.
    if toolCalls.isEmpty, reasoning.isEmpty, content.isEmpty, !text.isEmpty {
        let extracted = extractThinking(text)
        guard includeToolCalls else {
            return ModelOutputParseResult(reasoning: extracted.reasoning, content: extracted.content, toolCalls: [])
        }
        let info = genericExtract(extracted.content, idGenerator: idGenerator)
        return ModelOutputParseResult(
            reasoning: extracted.reasoning,
            content: info.toolsCalled ? (info.content ?? "") : extracted.content,
            toolCalls: info.toolCalls
        )
    }

    // A gpt-oss fine-tune that emitted a generic `<tool_call>` in the final
    // channel instead of a harmony recipient still gets parsed.
    if includeToolCalls, result.toolCalls.isEmpty {
        let info = genericExtract(result.content, idGenerator: idGenerator)
        if info.toolsCalled {
            result = ModelOutputParseResult(
                reasoning: result.reasoning,
                content: info.content ?? "",
                toolCalls: info.toolCalls
            )
        }
    }
    return result
}

private func harmonyFunctionsRecipient(in header: String) -> String? {
    guard let range = header.range(of: harmonyFunctionsRecipientPrefix) else { return nil }
    let tail = header[range.upperBound...]
    let name = tail.prefix { character in
        !(character == " " || character == "\t" || character == "\n" || character == "<")
    }
    return name.isEmpty ? nil : String(name)
}

private func harmonyBodyEnd(in text: String, from index: String.Index) -> String.Index {
    harmonyBoundaryTokens
        .compactMap { text.range(of: $0, range: index..<text.endIndex)?.lowerBound }
        .min() ?? text.endIndex
}
