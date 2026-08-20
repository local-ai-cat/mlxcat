import Foundation
import Network

struct AnthropicMessagesHandler {
    let backend: any OpenAIChatBackend
    let mcpManager: MCPManager?

    init(backend: any OpenAIChatBackend, mcpManager: MCPManager? = nil) {
        self.backend = backend
        self.mcpManager = mcpManager
    }

    func handleMessages(_ request: HTTPRequest, connection: NWConnection) async throws {
        let messagesRequest: AnthropicMessagesRequest
        do {
            messagesRequest = try AnthropicMessagesRequest.parse(request.body)
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 422
            try await sendJSON(openAIErrorBody(message: String(describing: error), status: status), status: status, connection: connection)
            return
        }

        let effectiveRequest = await anthropicMessagesRequestByMergingMCPTools(messagesRequest, manager: mcpManager)
        let stream = try await backend.startChatCompletion(effectiveRequest.openAIRequest())
        if effectiveRequest.stream {
            try await sendStreaming(request: effectiveRequest, stream: stream, connection: connection)
        } else {
            let completion = try await collectBufferedCompletion(stream: stream, stopSequences: effectiveRequest.stopSequences)
            try await sendJSON(
                buildAnthropicMessageResponse(
                    request: effectiveRequest,
                    completion: completion,
                    promptTokens: stream.promptTokens
                ),
                status: 200,
                connection: connection
            )
        }
    }

    func handleCountTokens(_ request: HTTPRequest, connection: NWConnection) async throws {
        let response = try await countTokensResponse(for: request)
        try await sendJSON(response.body, status: response.status, connection: connection)
    }

    func countTokensResponseForTesting(_ request: HTTPRequest) async -> HTTPJSONResponse {
        do {
            return try await countTokensResponse(for: request)
        } catch {
            let httpError = (error as? OpenAIHTTPError)
                ?? OpenAIHTTPError(status: 500, message: String(describing: error))
            return HTTPJSONResponse(
                status: httpError.status,
                body: openAIErrorBody(message: httpError.message, status: httpError.status)
            )
        }
    }

    private func countTokensResponse(for request: HTTPRequest) async throws -> HTTPJSONResponse {
        let countTokensRequest: AnthropicCountTokensRequest
        do {
            countTokensRequest = try AnthropicCountTokensRequest.parse(request.body)
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 422
            return HTTPJSONResponse(
                status: status,
                body: openAIErrorBody(message: String(describing: error), status: status)
            )
        }

        let result: AnthropicCountTokensResult
        if let countingBackend = backend as? any AnthropicTokenCountingBackend {
            result = try await countingBackend.countTokens(countTokensRequest)
        } else {
            result = AnthropicCountTokensResult(
                inputTokens: countTokensRequest.estimatedInputTokens(),
                estimated: true
            )
        }
        return HTTPJSONResponse(status: 200, body: buildAnthropicCountTokensResponse(result: result))
    }

    private func sendStreaming(
        request: AnthropicMessagesRequest,
        stream: OpenAIChatStream,
        connection: NWConnection
    ) async throws {
        try await connection.send(data: Data(streamingHeader.utf8))

        var formatter = AnthropicStreamFormatter(
            id: "msg_\(UUID().uuidString.prefix(8))",
            model: request.model,
            promptTokens: stream.promptTokens,
            stopSequences: request.stopSequences,
            toolsRequested: request.tools != nil
        )
        for event in formatter.startEvents() {
            try await sendSSE(event, connection: connection)
        }

        do {
            for try await chunk in stream.chunks {
                for event in formatter.feed(chunk) {
                    try await sendSSE(event, connection: connection)
                }
            }
            for event in formatter.finishEvents() {
                try await sendSSE(event, connection: connection)
            }
            try await connection.sendFinal(data: Data())
        } catch {
            try await sendSSE(
                AnthropicSSEEvent(
                    name: "error",
                    payload: ["type": "error", "error": openAIErrorObject(message: String(describing: error), status: 500)]
                ),
                connection: connection
            )
            try await connection.sendFinal(data: Data())
        }
    }

    private func collectBufferedCompletion(
        stream: OpenAIChatStream,
        stopSequences: [String]
    ) async throws -> AnthropicBufferedCompletion {
        var stopMatcher = AnthropicStopSequenceMatcher(stopSequences: stopSequences)
        var text = ""
        var completionTokens = 0
        var finishReason = "length"
        var stoppedByTextStop = false
        var stopSequence: String?

        for try await chunk in stream.chunks {
            completionTokens += 1
            if let chunkFinishReason = chunk.finishReason {
                finishReason = chunkFinishReason
            }
            let stopMatch = stopMatcher.feed(chunk.text)
            text += stopMatch.text
            if stopMatch.stopped {
                finishReason = "stop"
                stoppedByTextStop = true
                stopSequence = stopMatch.stopSequence ?? stopSequences.first
                break
            }
        }

        if !stoppedByTextStop {
            let stopMatch = stopMatcher.finish()
            text += stopMatch.text
            if stopMatch.stopped {
                finishReason = "stop"
                stoppedByTextStop = true
                stopSequence = stopMatch.stopSequence ?? stopSequences.first
            }
        }

        return AnthropicBufferedCompletion(
            text: text,
            completionTokens: completionTokens,
            finishReason: finishReason,
            stoppedByTextStop: stoppedByTextStop,
            stopSequence: stopSequence
        )
    }

    private func sendSSE(_ event: AnthropicSSEEvent, connection: NWConnection) async throws {
        let data = try JSONSerialization.data(withJSONObject: event.payload, options: [])
        let payload = "event: \(event.name)\ndata: \(String(decoding: data, as: UTF8.self))\n\n"
        try await connection.send(data: Data(payload.utf8))
    }

    private func sendJSON(_ object: [String: Any], status: Int, connection: NWConnection) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let header = "HTTP/1.1 \(status) \(anthropicReasonPhrase(status))\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + body)
    }

    private var streamingHeader: String {
        "HTTP/1.1 200 OK\r\n"
            + "Content-Type: text/event-stream\r\n"
            + "Cache-Control: no-cache\r\n"
            + "Connection: close\r\n"
            + "X-Accel-Buffering: no\r\n"
            + "\r\n"
    }
}

private func anthropicReasonPhrase(_ status: Int) -> String {
    switch status {
    case 200:
        return "OK"
    case 400:
        return "Bad Request"
    case 404:
        return "Not Found"
    case 422:
        return "Unprocessable Entity"
    case 500:
        return "Internal Server Error"
    default:
        return "OK"
    }
}
