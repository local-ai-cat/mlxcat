import Foundation

/// Token counters and stop matching are shared by buffered and streaming chat.
struct ChatCompletionProgress {
    var firstToken: UInt64?
    var lastToken: UInt64?
    var completionTokens = 0
    var cachedPromptTokens = 0
    var finishReason = "length"
    // Capture raw IDs before thinking parsing or stop trimming.
    var tokenIDs: [Int] = []
    var stoppedByTextStop = false
    private var stopMatcher: StreamingStopSequenceMatcher

    init(stopSequences: [String]) {
        stopMatcher = StreamingStopSequenceMatcher(stopSequences: stopSequences)
    }

    mutating func feed(_ chunk: OpenAIChatChunk) -> String {
        let now = DispatchTime.now().uptimeNanoseconds
        if firstToken == nil { firstToken = now }
        lastToken = now
        completionTokens += 1
        cachedPromptTokens = max(cachedPromptTokens, chunk.cachedPromptTokens)
        tokenIDs.append(chunk.tokenID)
        if let reason = chunk.finishReason { finishReason = reason }
        return accept(stopMatcher.feed(chunk.text))
    }

    mutating func finish() -> String {
        guard !stoppedByTextStop else { return "" }
        return accept(stopMatcher.finish())
    }

    private mutating func accept(_ match: StopSequenceMatch) -> String {
        if match.stopped {
            finishReason = "stop"
            stoppedByTextStop = true
        }
        return match.text
    }

    func usage(promptTokens: Int, started: UInt64, ended: UInt64) -> [String: Any] {
        let firstToken = firstToken ?? ended
        let lastToken = lastToken ?? ended
        let ttft = seconds(firstToken - started)
        let total = seconds(ended - started)
        let generation = max(seconds((lastToken >= firstToken ? lastToken - firstToken : 0)), 1e-9)
        return [
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "total_tokens": promptTokens + completionTokens,
            "input_tokens": promptTokens,
            "output_tokens": completionTokens,
            "prompt_tokens_details": ["cached_tokens": cachedPromptTokens],
            "time_to_first_token": ttft,
            "total_time": total,
            "prompt_eval_duration": ttft,
            "generation_duration": generation,
            "prompt_tokens_per_second": Double(promptTokens) / max(ttft, 1e-9),
            "generation_tokens_per_second": Double(completionTokens) / generation
        ]
    }

    private func seconds(_ nanoseconds: UInt64) -> Double {
        Double(nanoseconds) / 1_000_000_000
    }
}
