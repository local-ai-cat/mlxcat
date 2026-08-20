import Foundation

private let openThinkTag = "<think>"
private let closeThinkTag = "</think>"
private let minimaxOpenThinkTag = "<mm:think>"
private let minimaxCloseThinkTag = "</mm:think>"
private let harmonyChannelMarker = "<|channel|>"
private let harmonyMessageMarker = "<|message|>"
private let harmonyTerminators = ["<|end|>", "<|return|>", "<|call|>"]
private let harmonyControlMarkers = harmonyTerminators + [
    harmonyChannelMarker,
    harmonyMessageMarker,
    "<|start|>",
]

public func extractThinking(
    _ text: String,
    startInThinking: Bool = false
) -> (reasoning: String, content: String) {
    guard !text.isEmpty else { return ("", "") }

    let normalized = text
        .replacingOccurrences(of: minimaxOpenThinkTag, with: openThinkTag)
        .replacingOccurrences(of: minimaxCloseThinkTag, with: closeThinkTag)

    if normalized.contains(harmonyChannelMarker) {
        let harmony = parseHarmonyChannels(normalized)
        return (
            harmony.reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            harmony.content.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    if startInThinking {
        var parser = ThinkingParser(startInThinking: true)
        let delta = parser.feed(normalized)
        let final = parser.finish()
        return (
            (delta.reasoning + final.reasoning).trimmingCharacters(in: .whitespacesAndNewlines),
            (delta.content + final.content).trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    var thinkingParts: [String] = []
    var remaining = normalized
    while let openRange = remaining.range(of: openThinkTag),
        let closeRange = remaining[openRange.upperBound...].range(of: closeThinkTag)
    {
        thinkingParts.append(String(remaining[openRange.upperBound..<closeRange.lowerBound]))
        remaining.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
    }

    if !thinkingParts.isEmpty {
        return (
            thinkingParts.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines),
            remaining.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    if normalized.contains(closeThinkTag) && !normalized.contains(openThinkTag),
        let closeRange = normalized.range(of: closeThinkTag)
    {
        return (
            String(normalized[..<closeRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines),
            String(normalized[closeRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    if normalized.contains(openThinkTag) && !normalized.contains(closeThinkTag),
        let openRange = normalized.range(of: openThinkTag)
    {
        let before = normalized[..<openRange.lowerBound]
        let after = normalized[openRange.upperBound...]
        return ("", String(before + after).trimmingCharacters(in: .whitespacesAndNewlines))
    }

    return ("", text)
}

public struct ThinkingParser: Sendable {
    private var inThinking: Bool
    private let startedInThinking: Bool
    private var buffer = ""
    private var closeSeen = false
    private var thinkingAccumulated: [String] = []
    private var contentEmitted = false
    private var harmonyActive = false
    private var harmonyRaw = ""
    private var harmonyReasoningEmittedCount = 0
    private var harmonyContentEmittedCount = 0

    public init(startInThinking: Bool = false) {
        self.inThinking = startInThinking
        self.startedInThinking = startInThinking
    }

    public mutating func feed(_ text: String) -> (reasoning: String, content: String) {
        guard !text.isEmpty else { return ("", "") }

        let text = buffer + text
        buffer = ""
        if harmonyActive || Self.couldBeHarmony(text) {
            return feedHarmony(text)
        }

        var thinkingOut = ""
        var contentOut = ""
        var index = text.startIndex

        while index < text.endIndex {
            if text[index] == "<" {
                let remaining = text[index...]

                if remaining.hasPrefix(openThinkTag) {
                    inThinking = true
                    index = text.index(index, offsetBy: openThinkTag.count)
                    continue
                }

                if remaining.hasPrefix(closeThinkTag) {
                    inThinking = false
                    closeSeen = true
                    index = text.index(index, offsetBy: closeThinkTag.count)
                    continue
                }

                if Self.couldBeTag(String(remaining)) {
                    buffer = String(remaining)
                    break
                }

                append("<", thinkingOut: &thinkingOut, contentOut: &contentOut)
                index = text.index(after: index)
            } else {
                append(String(text[index]), thinkingOut: &thinkingOut, contentOut: &contentOut)
                index = text.index(after: index)
            }
        }

        if !thinkingOut.isEmpty {
            thinkingAccumulated.append(thinkingOut)
        }
        if !contentOut.isEmpty {
            contentEmitted = true
        }
        return (thinkingOut, contentOut)
    }

    public mutating func finish() -> (reasoning: String, content: String) {
        if harmonyActive {
            let finalDelta = feedHarmony(buffer)
            buffer = ""
            return finalDelta
        }

        let partial = buffer
        buffer = ""

        if inThinking && !startedInThinking && !closeSeen && !contentEmitted && !thinkingAccumulated.isEmpty {
            contentEmitted = true
            return ("", thinkingAccumulated.joined() + partial)
        }

        guard !partial.isEmpty else { return ("", "") }

        if inThinking {
            thinkingAccumulated.append(partial)
            return (partial, "")
        }

        contentEmitted = true
        return ("", partial)
    }

    private func append(_ value: String, thinkingOut: inout String, contentOut: inout String) {
        if inThinking {
            thinkingOut += value
        } else {
            contentOut += value
        }
    }

    private static func couldBeTag(_ text: String) -> Bool {
        guard text.count < closeThinkTag.count else { return false }
        return openThinkTag.hasPrefix(text) || closeThinkTag.hasPrefix(text)
    }

    private mutating func feedHarmony(_ text: String) -> (reasoning: String, content: String) {
        harmonyActive = true
        harmonyRaw += text
        let parsed = parseHarmonyChannels(harmonyRaw)
        let reasoningDelta = parsed.reasoning.dropFirstCharacters(harmonyReasoningEmittedCount)
        let contentDelta = parsed.content.dropFirstCharacters(harmonyContentEmittedCount)
        harmonyReasoningEmittedCount += reasoningDelta.count
        harmonyContentEmittedCount += contentDelta.count
        if !reasoningDelta.isEmpty {
            thinkingAccumulated.append(reasoningDelta)
        }
        if !contentDelta.isEmpty {
            contentEmitted = true
        }
        return (reasoningDelta, contentDelta)
    }

    private static func couldBeHarmony(_ text: String) -> Bool {
        text.contains("<|") || "<|".hasPrefix(text)
    }
}

private func parseHarmonyChannels(_ text: String) -> (reasoning: String, content: String) {
    var reasoning = ""
    var content = ""
    var index = text.startIndex

    while let channelRange = text[index...].range(of: harmonyChannelMarker) {
        let headerStart = channelRange.upperBound
        guard let messageRange = text[headerStart...].range(of: harmonyMessageMarker) else {
            break
        }

        let header = String(text[headerStart..<messageRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let channel = header.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" })
            .first
            .map(String.init) ?? ""
        let bodyStart = messageRange.upperBound
        let bodyEnd = firstHarmonyBoundary(in: text, from: bodyStart)?.lowerBound
            ?? harmonySafeBodyEnd(in: text, from: bodyStart)
        let body = String(text[bodyStart..<bodyEnd])

        switch channel {
        case "analysis", "commentary":
            reasoning += body
        case "final":
            content += body
        default:
            reasoning += body
        }

        if let boundary = firstHarmonyBoundary(in: text, from: bodyStart) {
            index = boundary.upperBound
        } else {
            index = text.endIndex
        }
    }

    return (reasoning, content)
}

private func firstHarmonyBoundary(in text: String, from index: String.Index) -> Range<String.Index>? {
    return (harmonyTerminators + [harmonyChannelMarker]).compactMap { text[index...].range(of: $0) }
        .min { $0.lowerBound < $1.lowerBound }
}

private func harmonySafeBodyEnd(in text: String, from index: String.Index) -> String.Index {
    guard index < text.endIndex else { return text.endIndex }
    let body = text[index...]
    for marker in harmonyControlMarkers {
        let maxPrefixLength = min(marker.count - 1, body.count)
        guard maxPrefixLength > 0 else { continue }
        for length in stride(from: maxPrefixLength, through: 1, by: -1) {
            let suffixStart = body.index(body.endIndex, offsetBy: -length)
            if marker.hasPrefix(String(body[suffixStart...])) {
                return suffixStart
            }
        }
    }
    return text.endIndex
}

private extension String {
    func dropFirstCharacters(_ count: Int) -> String {
        guard count > 0 else { return self }
        guard count < self.count else { return "" }
        return String(self[index(startIndex, offsetBy: count)...])
    }
}
