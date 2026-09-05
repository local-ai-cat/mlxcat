import Foundation
import Network

public struct StopSequenceMatch: Sendable, Equatable {
    public let text: String
    public let stopped: Bool

    public init(text: String, stopped: Bool) {
        self.text = text
        self.stopped = stopped
    }
}

public struct StreamingStopSequenceMatcher: Sendable {
    private let stopSequences: [String]
    private let maxTailCount: Int
    private var pending = ""

    public init(stopSequences: [String]) {
        let stopSequences = stopSequences.filter { !$0.isEmpty }
        self.stopSequences = stopSequences
        self.maxTailCount = max((stopSequences.map(\.count).max() ?? 0) - 1, 0)
    }

    public mutating func feed(_ text: String) -> StopSequenceMatch {
        guard !stopSequences.isEmpty else {
            return StopSequenceMatch(text: text, stopped: false)
        }

        let combined = pending + text
        if let range = firstStopRange(in: combined, stopSequences: stopSequences) {
            pending = ""
            return StopSequenceMatch(text: String(combined[..<range.lowerBound]), stopped: true)
        }

        let tailCount = min(maxTailCount, combined.count)
        let emitEnd = combined.index(combined.endIndex, offsetBy: -tailCount)
        let emitted = String(combined[..<emitEnd])
        pending = String(combined[emitEnd...])
        return StopSequenceMatch(text: emitted, stopped: false)
    }

    public mutating func finish() -> StopSequenceMatch {
        let text = pending
        pending = ""
        guard let range = firstStopRange(in: text, stopSequences: stopSequences) else {
            return StopSequenceMatch(text: text, stopped: false)
        }
        return StopSequenceMatch(text: String(text[..<range.lowerBound]), stopped: true)
    }
}

public func truncateAtStop(_ text: String, stopSequences: [String]) -> StopSequenceMatch {
    let stopSequences = stopSequences.filter { !$0.isEmpty }
    guard let range = firstStopRange(in: text, stopSequences: stopSequences) else {
        return StopSequenceMatch(text: text, stopped: false)
    }
    return StopSequenceMatch(text: String(text[..<range.lowerBound]), stopped: true)
}

private func firstStopRange(in text: String, stopSequences: [String]) -> Range<String.Index>? {
    var bestRange: Range<String.Index>?
    for stop in stopSequences {
        guard let range = text.range(of: stop) else { continue }
        if let current = bestRange {
            if range.lowerBound < current.lowerBound {
                bestRange = range
            }
        } else {
            bestRange = range
        }
    }
    return bestRange
}

public protocol OpenAIChatBackend: Sendable {
    var models: [OpenAIModelInfo] { get }
    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream
}
