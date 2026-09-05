import Foundation
import Network

extension OpenAIServer {
    func sendStreamingChatWithTools(request: OpenAIChatRequest, stream: OpenAIChatStream,
                                    started: UInt64, connection: NWConnection) async throws {
        let response = ChatStreamResponse(server: self, request: request,
                                          promptTokens: stream.promptTokens, started: started, connection: connection)
        var progress = ChatCompletionProgress(stopSequences: request.stop)
        var text = ""
        try await response.open()
        do {
            for try await chunk in stream.chunks {
                text += progress.feed(chunk)
                if progress.stoppedByTextStop { break }
            }
        } catch {
            try await response.fail(error)
            return
        }
        text += progress.finish()
        progress.finishReason = try await sendParsedToolOutput(text, response: response,
                                                               finishReason: progress.finishReason)
        try await response.finish(progress)
    }

    private func sendParsedToolOutput(_ text: String, response: ChatStreamResponse,
                                      finishReason: String) async throws -> String {
        let request = response.request
        let result = parseModelOutput(text, model: request.model, startInThinking: request.startInThinking)
        let parsed = ToolCallParseResult(content: result.content, toolCalls: result.toolCalls)
        guard !parsed.toolCalls.isEmpty else {
            try await response.sendDelta((result.reasoning, result.content))
            return finishReason
        }
        if !result.reasoning.isEmpty { try await response.sendDelta((result.reasoning, "")) }
        if !parsed.content.isEmpty { try await response.sendDelta(("", parsed.content)) }
        try await response.sendToolResult(parsed)
        return finishReasonForToolCalls(defaultFinishReason: finishReason, parsed: parsed)
    }
}
