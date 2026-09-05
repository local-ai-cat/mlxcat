import Foundation
import Network

extension OpenAIServer {
    func sendStreamingChat(request: OpenAIChatRequest, stream: OpenAIChatStream,
                           started: UInt64, connection: NWConnection) async throws {
        if toolsRequested(request) {
            try await sendStreamingChatWithTools(request: request, stream: stream,
                                                 started: started, connection: connection)
            return
        }
        let response = ChatStreamResponse(server: self, request: request,
                                          promptTokens: stream.promptTokens, started: started, connection: connection)
        var progress = ChatCompletionProgress(stopSequences: request.stop)
        var thinkingParser = ThinkingParser(startInThinking: request.startInThinking)
        try await response.open()
        do {
            for try await chunk in stream.chunks {
                let text = progress.feed(chunk)
                if !text.isEmpty { try await response.sendDelta(thinkingParser.feed(text)) }
                if progress.stoppedByTextStop { break }
            }
        } catch {
            try await response.fail(error)
            return
        }
        let tail = progress.finish()
        if !tail.isEmpty { try await response.sendDelta(thinkingParser.feed(tail)) }
        try await response.sendDelta(thinkingParser.finish())
        try await response.finish(progress)
    }
}
