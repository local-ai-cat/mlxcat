import Foundation

/// Inkling's typed message frames, following Ollama's model/parsers/inkling2.go.
struct InklingOutputParser: Sendable {
    static let model = "<|message_model|>"
    static let thinking = "<|content_thinking|>"
    static let text = "<|content_text|>"
    static let tool = "<|content_invoke_tool_json|>"
    static let end = "<|end_message|>"
    static let eos = "<|content_model_end_sampling|>"

    private enum Mode: Sendable {
        case outside
        case header
        case thinking
        case visible
        case tool(String)
    }

    private var mode: Mode = .outside
    private var buffer = ""

    struct Output {
        var reasoning = ""
        var content = ""
        var calls: [(name: String, arguments: String)] = []
    }

    mutating func feed(_ text: String, finished: Bool = false) -> Output {
        buffer += text
        var output = Output()
        while !buffer.isEmpty {
            switch mode {
            case .outside:
                if let marker = firstMarker([Self.model, Self.eos]) {
                    output.content += buffer[..<marker.range.lowerBound]
                    buffer.removeSubrange(..<marker.range.upperBound)
                    mode = marker.value == Self.model ? .header : .outside
                } else {
                    output.content += consumeSafeText(markers: [Self.model, Self.eos], finished: finished)
                    return output
                }
            case .header:
                guard let marker = firstMarker([Self.thinking, Self.text, Self.tool, Self.end, Self.eos]) else {
                    if finished { buffer = "" }
                    return output
                }
                let name = buffer[..<marker.range.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                buffer.removeSubrange(..<marker.range.upperBound)
                switch marker.value {
                case Self.thinking: mode = .thinking
                case Self.text: mode = .visible
                case Self.tool: mode = .tool(name)
                default: mode = .outside
                }
            case .thinking, .visible:
                let markers = [Self.end, Self.eos, Self.model]
                let fragment: String
                if let marker = firstMarker(markers) {
                    fragment = String(buffer[..<marker.range.lowerBound])
                    buffer.removeSubrange(..<marker.range.upperBound)
                    if case .thinking = mode { output.reasoning += fragment }
                    else { output.content += fragment }
                    mode = marker.value == Self.model ? .header : .outside
                } else {
                    fragment = consumeSafeText(markers: markers, finished: finished)
                    if case .thinking = mode { output.reasoning += fragment }
                    else { output.content += fragment }
                    return output
                }
            case .tool(let name):
                guard let body = completeToolBody(finished: finished) else { return output }
                if let call = Self.toolCall(body, recipient: name) {
                    output.calls.append(call)
                }
                mode = .outside
            }
        }
        return output
    }

    private mutating func completeToolBody(finished: Bool) -> String? {
        var start = buffer.startIndex
        while let marker = firstMarker([Self.end, Self.eos], from: start) {
            let body = String(buffer[..<marker.range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            if Self.isObject(body) {
                buffer.removeSubrange(..<marker.range.upperBound)
                return body
            }
            start = marker.range.upperBound
        }
        guard finished else { return nil }
        let body = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = ""
        return Self.isObject(body) ? body : nil
    }

    private static func toolCall(_ body: String, recipient: String) -> (String, String)? {
        guard let payload = (try? JSONSerialization.jsonObject(with: Data(body.utf8))) as? [String: Any],
            let name = payload["name"] as? String, !name.isEmpty,
            recipient.isEmpty || name == recipient
        else { return nil }
        let arguments: [String: Any]
        if let value = payload["args"] {
            guard let object = value as? [String: Any] else { return nil }
            arguments = object
        } else {
            arguments = [:]
        }
        guard let data = try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]) else {
            return nil
        }
        return (name, String(decoding: data, as: UTF8.self))
    }

    private static func isObject(_ body: String) -> Bool {
        (try? JSONSerialization.jsonObject(with: Data(body.utf8))) is [String: Any]
    }

    private func firstMarker(
        _ markers: [String], from start: String.Index? = nil
    ) -> (value: String, range: Range<String.Index>)? {
        let range = (start ?? buffer.startIndex)..<buffer.endIndex
        return markers.compactMap { marker in
            buffer.range(of: marker, range: range).map { (value: marker, range: $0) }
        }.min { $0.range.lowerBound < $1.range.lowerBound }
    }

    private mutating func consumeSafeText(markers: [String], finished: Bool) -> String {
        var end = buffer.endIndex
        if !finished, let start = buffer.lastIndex(of: "<"),
            markers.contains(where: { $0.hasPrefix(buffer[start...]) })
        {
            end = start
        }
        let result = String(buffer[..<end])
        buffer.removeSubrange(..<end)
        return result
    }
}
