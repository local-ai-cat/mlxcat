import Foundation

// Per-family tool parsers translated from vLLM `vllm/tool_parsers/`. Each keys
// off its native marker syntax; selection is by family name in
// `ToolParserManager`. See `ToolParser.swift` for the base + registry.

// MARK: - Hermes / Qwen (`<tool_call>{json}</tool_call>`)

/// Port of vLLM `Hermes2ProToolParser` (`hermes_tool_parser.py`). Buffered:
/// each `<tool_call>…</tool_call>` region is a `{"name","arguments"}` JSON
/// object. Streaming: the canonical incremental algorithm — name emitted as
/// soon as it parses, arguments diffed against what was already sent.
public final class HermesToolParser: ToolParser {
    private let startToken = "<tool_call>"
    private let endToken = "</tool_call>"
    private var sentContentIndex = 0

    public override func extractToolCalls(_ modelOutput: String) -> ExtractedToolCallInformation {
        guard modelOutput.contains(startToken) else { return .noTools(modelOutput) }

        var toolCalls: [ParsedToolCall] = []
        for (body, _) in toolCallJSONRegions(in: modelOutput) {
            guard let object = parseJSONObject(body),
                let name = object["name"] as? String, !name.isEmpty
            else {
                return .noTools(modelOutput)
            }
            let arguments = argumentsJSONString(from: object["arguments"])
            toolCalls.append(ParsedToolCall(id: idGenerator(), name: name, arguments: arguments))
        }
        guard !toolCalls.isEmpty else { return .noTools(modelOutput) }

        let prefix = String(modelOutput[modelOutput.startIndex..<(modelOutput.range(of: startToken)?.lowerBound ?? modelOutput.startIndex)])
        return ExtractedToolCallInformation(
            toolsCalled: true,
            toolCalls: toolCalls,
            content: prefix.isEmpty ? nil : prefix
        )
    }

    // Streaming (vLLM Hermes extract_tool_calls_streaming).
    public override func extractToolCallsStreaming(
        previousText: String,
        currentText: String,
        deltaText: String
    ) -> ToolCallStreamDelta? {
        let content = extractStreamContent(currentText)
        let regions = toolCallJSONRegions(in: currentText)
        var deltas: [StreamedToolCall] = []

        for (index, (tcJSON, isComplete)) in regions.enumerated() {
            while streamedNameForTool.count <= index { streamedNameForTool.append("") }
            while streamedArgsForTool.count <= index { streamedArgsForTool.append("") }

            if streamedNameForTool[index].isEmpty {
                guard let name = extractToolName(tcJSON) else { break }
                streamedNameForTool[index] = name
                deltas.append(StreamedToolCall(index: index, id: idGenerator(), name: name))
            }
            if let args = extractToolArguments(tcJSON, isComplete: isComplete),
                args.count > streamedArgsForTool[index].count,
                args.hasPrefix(streamedArgsForTool[index]) {
                let diff = String(args.dropFirst(streamedArgsForTool[index].count))
                streamedArgsForTool[index] = args
                deltas.append(StreamedToolCall(index: index, argumentsDelta: diff))
            }
        }

        if content == nil, deltas.isEmpty { return nil }
        return ToolCallStreamDelta(content: content, toolCalls: deltas)
    }

    private func extractStreamContent(_ currentText: String) -> String? {
        let sendableIndex: Int
        if let range = currentText.range(of: startToken) {
            sendableIndex = currentText.distance(from: currentText.startIndex, to: range.lowerBound)
        } else {
            let overlap = partialTagOverlap(currentText, startToken)
            sendableIndex = currentText.count - overlap
        }
        guard sendableIndex > sentContentIndex else { return nil }
        let start = currentText.index(currentText.startIndex, offsetBy: sentContentIndex)
        let end = currentText.index(currentText.startIndex, offsetBy: sendableIndex)
        sentContentIndex = sendableIndex
        return String(currentText[start..<end])
    }

    /// (jsonBody, isComplete) for each `<tool_call>` region, including a trailing
    /// unclosed one (vLLM `_extract_tool_call_jsons`).
    private func toolCallJSONRegions(in text: String) -> [(String, Bool)] {
        var results: [(String, Bool)] = []
        var cursor = text.startIndex
        while let start = text.range(of: startToken, range: cursor..<text.endIndex) {
            let jsonStart = start.upperBound
            if let end = text.range(of: endToken, range: jsonStart..<text.endIndex) {
                let body = String(text[jsonStart..<end.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                results.append((body, true))
                cursor = end.upperBound
            } else {
                var raw = String(text[jsonStart...])
                let overlap = partialTagOverlap(raw, endToken)
                if overlap > 0 { raw = String(raw.dropLast(overlap)) }
                let body = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                results.append((body, body.isEmpty ? false : isCompleteJSON(body)))
                break
            }
        }
        return results
    }

    private func extractToolName(_ tcJSON: String) -> String? {
        guard let range = tcJSON.range(of: #""name"\s*:\s*"([^"]+)""#, options: .regularExpression) else {
            return nil
        }
        let matched = String(tcJSON[range])
        guard let valueRange = matched.range(of: #""([^"]+)"$"#, options: .regularExpression) else { return nil }
        return String(matched[valueRange]).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    }

    private func extractToolArguments(_ tcJSON: String, isComplete: Bool) -> String? {
        guard let range = tcJSON.range(of: #""arguments"\s*:\s*"#, options: .regularExpression) else {
            return nil
        }
        var raw = String(tcJSON[range.upperBound...])
        if isComplete {
            raw = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if raw.hasSuffix("}") {
                raw = String(raw.dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return raw
    }
}

// MARK: - Llama 3 JSON (`{"name":…, "arguments"|"parameters":…}`)

/// Port of vLLM `Llama3JsonToolParser` (`llama_tool_parser.py`). Scans for
/// top-level JSON objects (optionally after `<|python_tag|>`), each with a
/// `name` and `arguments`/`parameters`. Accepts the `parameters` alias that the
/// generic Hermes parser misses — a likely cause of a Llama tool-call miss.
public final class Llama3JsonToolParser: ToolParser {
    private let botToken = "<|python_tag|>"

    public override func extractToolCalls(_ modelOutput: String) -> ExtractedToolCallInformation {
        guard modelOutput.contains(botToken) || modelOutput.contains("{") else {
            return .noTools(modelOutput)
        }
        var toolCalls: [ParsedToolCall] = []
        for object in topLevelJSONObjects(in: modelOutput) {
            guard let name = object["name"] as? String, !name.isEmpty else { continue }
            let argsValue = object["arguments"] ?? object["parameters"]
            toolCalls.append(
                ParsedToolCall(id: idGenerator(), name: name, arguments: argumentsJSONString(from: argsValue))
            )
        }
        guard !toolCalls.isEmpty else { return .noTools(modelOutput) }
        return ExtractedToolCallInformation(toolsCalled: true, toolCalls: toolCalls, content: nil)
    }

    /// Brace-matched top-level JSON objects (vLLM uses a JSON raw-decoder loop).
    private func topLevelJSONObjects(in text: String) -> [[String: Any]] {
        var objects: [[String: Any]] = []
        var index = text.startIndex
        while index < text.endIndex {
            guard text[index] == "{" else {
                index = text.index(after: index)
                continue
            }
            guard let end = matchingBraceEnd(in: text, openBrace: index) else { break }
            let candidate = String(text[index...end])
            if let object = parseJSONObject(candidate) {
                objects.append(object)
                index = text.index(after: end)
            } else {
                index = text.index(after: index)
            }
        }
        return objects
    }
}

// MARK: - Pythonic (`[func(arg=value, …)]`) — Llama 3.2 / Llama 4

/// Port of vLLM `PythonicToolParser` (`pythonic_tool_parser.py`). Output is a
/// Python list of calls, e.g. `[get_weather(city="London", unit="celsius")]`.
/// Values are Python/JSON literals. This is the format small Llama-3.2 models
/// use that no JSON/XML parser recognizes.
public final class PythonicToolParser: ToolParser {
    public override func extractToolCalls(_ modelOutput: String) -> ExtractedToolCallInformation {
        var text = modelOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix("<|python_start|>") {
            text = String(text.dropFirst("<|python_start|>".count))
            text = text.replacingOccurrences(of: "<|python_end|>", with: "")
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard text.hasPrefix("["), text.hasSuffix("]") else { return .noTools(modelOutput) }

        let inner = String(text.dropFirst().dropLast())
        guard let calls = parsePythonicCalls(inner), !calls.isEmpty else {
            return .noTools(modelOutput)
        }
        return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: nil)
    }

    public override func extractToolCallsStreaming(
        previousText: String,
        currentText: String,
        deltaText: String
    ) -> ToolCallStreamDelta? {
        let trimmed = currentText.trimmingCharacters(in: .whitespacesAndNewlines)
        // Hold content back while a pythonic list is mid-flight (vLLM suppresses
        // content once current_text starts with "[" / python_start).
        if trimmed.hasPrefix("[") || trimmed.hasPrefix("<|python_start|>") {
            return super.extractToolCallsStreaming(
                previousText: previousText, currentText: currentText, deltaText: ""
            )
        }
        return deltaText.isEmpty ? nil : ToolCallStreamDelta(content: deltaText)
    }

    /// Split `name(args), name2(args)` at top level, parsing each call.
    private func parsePythonicCalls(_ inner: String) -> [ParsedToolCall]? {
        var calls: [ParsedToolCall] = []
        var index = inner.startIndex
        while index < inner.endIndex {
            while index < inner.endIndex, inner[index] == " " || inner[index] == "," || inner[index] == "\n" || inner[index] == "\t" {
                index = inner.index(after: index)
            }
            guard index < inner.endIndex else { break }

            // name (allow dotted a.b.c)
            let nameStart = index
            while index < inner.endIndex, inner[index] != "(" {
                index = inner.index(after: index)
            }
            guard index < inner.endIndex else { return calls.isEmpty ? nil : calls }
            let name = String(inner[nameStart..<index]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard isValidPythonicName(name) else { return nil }

            guard let argsEnd = matchingParenEnd(in: inner, openParen: index) else { return nil }
            let argsBody = String(inner[inner.index(after: index)..<argsEnd])
            let arguments = parsePythonicKwargs(argsBody)
            calls.append(ParsedToolCall(id: idGenerator(), name: name, arguments: serializeJSONValue(arguments)))
            index = inner.index(after: argsEnd)
        }
        return calls.isEmpty ? nil : calls
    }

    private func isValidPythonicName(_ name: String) -> Bool {
        guard let first = name.first, first.isLetter || first == "_" else { return false }
        return name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "." }
    }

    private func parsePythonicKwargs(_ body: String) -> [String: Any] {
        var result: [String: Any] = [:]
        for entry in splitTopLevel(body, separator: ",") {
            guard let eq = topLevelIndex(of: "=", in: entry) else { continue }
            let key = String(entry[entry.startIndex..<eq]).trimmingCharacters(in: .whitespacesAndNewlines)
            let rawValue = String(entry[entry.index(after: eq)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else { continue }
            result[key] = pythonLiteralValue(rawValue)
        }
        return result
    }
}

// MARK: - Gemma 4 (`<|tool_call>call:name{args}<tool_call|>`)

/// Port of vLLM `gemma4_utils.parse_tool_calls` + `parser/gemma4._parse_gemma4_args`.
/// Tier 1: `<|tool_call>call:name{args}(<tool_call|>|<turn|>)`. Tier 2 fallback:
/// bare `call:name{args}` / `<call>name{args}`. Args use Gemma's custom
/// `key:<|"|>value<|"|>` / bare / nested syntax.
public final class Gemma4ToolParser: ToolParser {
    public override class var supportsRequiredAndNamed: Bool { false }

    private let startTag = "<|tool_call>"

    public override func extractToolCalls(_ modelOutput: String) -> ExtractedToolCallInformation {
        var calls = parseGemma4Calls(in: modelOutput, fallback: false)
        if calls.isEmpty {
            calls = parseGemma4Calls(in: modelOutput, fallback: true)
        }
        guard !calls.isEmpty else { return .noTools(modelOutput) }

        let prefix: String?
        if let range = modelOutput.range(of: startTag) {
            let text = String(modelOutput[modelOutput.startIndex..<range.lowerBound])
            prefix = text.isEmpty ? nil : text
        } else {
            prefix = nil
        }
        return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: prefix)
    }

    /// Tier 1 scans `<|tool_call>call:NAME{ARGS}` ended by `<tool_call|>` or
    /// `<turn|>`; tier 2 scans bare `call:NAME{ARGS}` / `<call>NAME{ARGS}`.
    private func parseGemma4Calls(in text: String, fallback: Bool) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            let head: Range<String.Index>?
            if fallback {
                head = firstGemma4FallbackHead(in: text, from: cursor)
            } else {
                head = text.range(of: startTag + "call:", range: cursor..<text.endIndex)
                    ?? text.range(of: startTag + "\ncall:", range: cursor..<text.endIndex)
            }
            guard let headRange = head else { break }

            // Name runs from after "call:"/"<call>" to the "{".
            guard let braceIndex = text.range(of: "{", range: headRange.upperBound..<text.endIndex)?.lowerBound else {
                break
            }
            let name = String(text[headRange.upperBound..<braceIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let braceEnd = matchingBraceEnd(in: text, openBrace: braceIndex) else { break }
            let argsBody = String(text[text.index(after: braceIndex)..<braceEnd])
            if isValidGemmaName(name) {
                let arguments = parseGemma4Args(argsBody)
                calls.append(ParsedToolCall(id: idGenerator(), name: name, arguments: serializeJSONValue(arguments)))
            }
            cursor = text.index(after: braceEnd)
        }
        return calls
    }

    private func firstGemma4FallbackHead(in text: String, from start: String.Index) -> Range<String.Index>? {
        // `<call>NAME{` or (start/space)`call:NAME{`
        var best: Range<String.Index>?
        if let r = text.range(of: "<call>", range: start..<text.endIndex) {
            best = r
        }
        var searchStart = start
        while let r = text.range(of: "call:", range: searchStart..<text.endIndex) {
            let precededOK = r.lowerBound == text.startIndex || {
                let prev = text[text.index(before: r.lowerBound)]
                return prev == " " || prev == "\n" || prev == "\t"
            }()
            if precededOK {
                if best == nil || r.lowerBound < best!.lowerBound { best = r }
                break
            }
            searchStart = r.upperBound
        }
        return best
    }

    private func isValidGemmaName(_ name: String) -> Bool {
        !name.isEmpty && name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "." || $0 == ":" || $0 == "-" }
    }
}

// MARK: - DeepSeek (R1/V3 fenced + V3.1)

/// Port of vLLM `DeepSeekV3ToolParser` + `DeepSeekV31ToolParser`
/// (`deepseekv3_tool_parser.py`, `deepseekv31_tool_parser.py`). Both wrap calls
/// in `<｜tool▁calls▁begin｜>…<｜tool▁calls▁end｜>`; each call is
/// `<｜tool▁call▁begin｜>…<｜tool▁call▁end｜>`.
///   V3/R1: `type<｜tool▁sep｜>name\n```json\n{args}\n``` `
///   V3.1:  `name<｜tool▁sep｜>{args}` (no fence)
public final class DeepSeekToolParser: ToolParser {
    public override class var supportsRequiredAndNamed: Bool { false }

    private let callsBegin = "<｜tool▁calls▁begin｜>"
    private let callBegin = "<｜tool▁call▁begin｜>"
    private let callEnd = "<｜tool▁call▁end｜>"
    private let sep = "<｜tool▁sep｜>"

    public override func extractToolCalls(_ modelOutput: String) -> ExtractedToolCallInformation {
        guard let beginRange = modelOutput.range(of: callsBegin) else { return .noTools(modelOutput) }

        var calls: [ParsedToolCall] = []
        var cursor = beginRange.lowerBound
        while let blockStart = modelOutput.range(of: callBegin, range: cursor..<modelOutput.endIndex) {
            let end = modelOutput.range(of: callEnd, range: blockStart.upperBound..<modelOutput.endIndex)?.lowerBound
                ?? modelOutput.endIndex
            let block = String(modelOutput[blockStart.upperBound..<end])
            if let call = parseBlock(block) {
                calls.append(call)
            }
            cursor = end
        }
        guard !calls.isEmpty else { return .noTools(modelOutput) }

        let prefix = String(modelOutput[modelOutput.startIndex..<beginRange.lowerBound])
        return ExtractedToolCallInformation(
            toolsCalled: true,
            toolCalls: calls,
            content: prefix.isEmpty ? nil : prefix
        )
    }

    private func parseBlock(_ block: String) -> ParsedToolCall? {
        guard let sepRange = block.range(of: sep) else { return nil }
        let beforeSep = String(block[block.startIndex..<sepRange.lowerBound])
        let afterSep = String(block[sepRange.upperBound...])

        if let fence = afterSep.range(of: "```json") {
            // V3/R1: name is after sep, up to the fence; args inside the fence.
            let name = String(afterSep[afterSep.startIndex..<fence.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }
            var args = String(afterSep[fence.upperBound...])
            if let nl = args.firstIndex(of: "\n") { args = String(args[args.index(after: nl)...]) }
            if let closing = args.range(of: "```", options: .backwards) {
                args = String(args[args.startIndex..<closing.lowerBound])
            }
            return ParsedToolCall(
                id: idGenerator(),
                name: name,
                arguments: normalizedArgumentsJSON(from: args)
            )
        } else {
            // V3.1: name before sep, args after sep.
            let name = beforeSep.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }
            return ParsedToolCall(
                id: idGenerator(),
                name: name,
                arguments: normalizedArgumentsJSON(from: afterSep)
            )
        }
    }
}
