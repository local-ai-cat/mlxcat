import Foundation
import Network

extension OpenAIServer {
    func handleChatCompletion(_ request: HTTPRequest, connection: NWConnection) async throws {
        let chatRequest: OpenAIChatRequest
        do {
            chatRequest = try OpenAIChatRequest.parse(request.body)
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 422
            try await sendJSON(
                openAIErrorBody(message: String(describing: error), status: status),
                status: status,
                connection: connection
            )
            return
        }
        let headerCacheSession = request.headers["x-cache-session"]
        let effectiveChatRequest = await chatRequestByMergingMCPTools(
            chatRequest.withCacheSession(headerCacheSession ?? chatRequest.cacheSession)
        )
        let started = DispatchTime.now().uptimeNanoseconds
        let stream = try await backend.startChatCompletion(effectiveChatRequest)

        try await stream.withLifetime { stream in
            if effectiveChatRequest.stream {
                try await sendStreamingChat(
                    request: effectiveChatRequest,
                    stream: stream,
                    started: started,
                    connection: connection
                )
            } else {
                try await sendBufferedChat(
                    request: effectiveChatRequest,
                    stream: stream,
                    started: started,
                    connection: connection
                )
            }
        }
    }

    private func chatRequestByMergingMCPTools(_ request: OpenAIChatRequest) async -> OpenAIChatRequest {
        guard let mcpManager else {
            return request
        }
        let tools = await mcpManager.mergedOpenAITools(
            userTools: request.tools,
            toolChoice: request.toolChoice
        )
        return openAIChatRequestByReplacingTools(request, tools: tools)
    }

    private func sendBufferedChat(
        request: OpenAIChatRequest,
        stream: OpenAIChatStream,
        started: UInt64,
        connection: NWConnection
    ) async throws {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        var progress = ChatCompletionProgress(stopSequences: request.stop)
        var text = ""
        for try await chunk in stream.chunks {
            text += progress.feed(chunk)
            if progress.stoppedByTextStop { break }
        }
        text += progress.finish()

        let result = parseModelOutput(
            text,
            model: request.model,
            includeToolCalls: toolsRequested(request),
            startInThinking: request.startInThinking
        )
        let parsed = ToolCallParseResult(content: result.content, toolCalls: result.toolCalls)
        let message = buildAssistantMessageWithToolCalls(
            content: result.content,
            reasoning: result.reasoning,
            parsed: parsed
        )
        if !parsed.toolCalls.isEmpty {
            progress.finishReason = finishReasonForToolCalls(
                defaultFinishReason: progress.finishReason,
                parsed: parsed
            )
        }

        let ended = DispatchTime.now().uptimeNanoseconds
        try await sendJSON(
            [
                "id": id,
                "object": "chat.completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": request.model,
                "choices": [
                    [
                        "index": 0,
                        "message": message,
                        "finish_reason": progress.finishReason,
                        "token_ids": progress.tokenIDs
                    ]
                ],
                "usage": progress.usage(promptTokens: stream.promptTokens, started: started, ended: ended)
            ],
            status: 200,
            connection: connection
        )
    }

}
