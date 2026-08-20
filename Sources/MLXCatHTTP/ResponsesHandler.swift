import Foundation
import Network

struct ResponsesHandler {
    let backend: any OpenAIChatBackend
    let store: ResponsesStore
    private let mcpManager: MCPManager?

    init(backend: any OpenAIChatBackend, store: ResponsesStore, mcpManager: MCPManager? = nil) {
        self.backend = backend
        self.store = store
        self.mcpManager = mcpManager
    }

    func handleCreate(_ request: HTTPRequest, connection: NWConnection) async throws {
        let responsesRequest: ResponsesRequest
        do {
            responsesRequest = try ResponsesRequest.parse(request.body)
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 422
            try await sendJSON(openAIErrorBody(message: String(describing: error), status: status), status: status, connection: connection)
            return
        }

        let previousMessages: [OpenAIChatMessage]
        if responsesRequest.store, let previousResponseID = responsesRequest.previousResponseID {
            previousMessages = await store.contextMessages(id: previousResponseID) ?? []
        } else {
            previousMessages = []
        }

        let effectiveRequest = await responsesRequestByMergingMCPTools(responsesRequest, manager: mcpManager)
        let stream = try await backend.startChatCompletion(
            effectiveRequest.openAIRequest(previousMessages: previousMessages)
        )
        if effectiveRequest.stream {
            try await sendStreaming(
                request: effectiveRequest,
                previousMessages: previousMessages,
                stream: stream,
                connection: connection
            )
        } else {
            let completion = try await collectBufferedCompletion(stream: stream)
            let id = "resp_\(UUID().uuidString.prefix(8))"
            let response = buildResponsesObject(
                request: effectiveRequest,
                id: id,
                promptTokens: stream.promptTokens,
                completion: completion
            )
            let data = try responsesJSONData(response)
            if effectiveRequest.store {
                let extracted = extractThinking(completion.text)
                await store.put(
                    id: id,
                    responseData: data,
                    contextMessages: previousMessages + effectiveRequest.inputMessages + [
                        OpenAIChatMessage(
                            role: "assistant",
                            content: extracted.content,
                            reasoningContent: extracted.reasoning.isEmpty ? nil : extracted.reasoning
                        )
                    ]
                )
            }
            try await sendJSONData(data, status: 200, connection: connection)
        }
    }

    func handleGet(id: String, connection: NWConnection) async throws {
        guard let data = await store.responseData(id: id) else {
            try await sendJSON(openAIErrorBody(message: "response not found", status: 404), status: 404, connection: connection)
            return
        }
        try await sendJSONData(data, status: 200, connection: connection)
    }

    func handleDelete(id: String, connection: NWConnection) async throws {
        guard await store.delete(id: id) else {
            try await sendJSON(openAIErrorBody(message: "response not found", status: 404), status: 404, connection: connection)
            return
        }
        try await sendJSON(["id": id, "object": "response.deleted", "deleted": true], status: 200, connection: connection)
    }

    private func sendStreaming(
        request: ResponsesRequest,
        previousMessages: [OpenAIChatMessage],
        stream: OpenAIChatStream,
        connection: NWConnection
    ) async throws {
        try await connection.send(data: Data(streamingHeader.utf8))

        let id = "resp_\(UUID().uuidString.prefix(8))"
        var formatter = ResponsesStreamFormatter(
            id: id,
            model: request.model,
            createdAt: Int(Date().timeIntervalSince1970),
            promptTokens: stream.promptTokens,
            request: request
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
            let finalEvents = formatter.finishEvents()
            for event in finalEvents {
                try await sendSSE(event, connection: connection)
            }
            if request.store {
                await store.put(
                    id: id,
                    responseData: try responsesJSONData(formatter.completedResponseObject()),
                    contextMessages: previousMessages + formatter.completedContextMessages
                )
            }
            try await connection.sendFinal(data: Data())
        } catch {
            try await sendSSE(
                ResponseSSEEvent(
                    name: "error",
                    payload: ["type": "error", "error": openAIErrorObject(message: String(describing: error), status: 500)]
                ),
                connection: connection
            )
            try await connection.sendFinal(data: Data())
        }
    }

    private func collectBufferedCompletion(stream: OpenAIChatStream) async throws -> ResponsesBufferedCompletion {
        var text = ""
        var completionTokens = 0
        for try await chunk in stream.chunks {
            completionTokens += 1
            text += chunk.text
        }
        return ResponsesBufferedCompletion(text: text, completionTokens: completionTokens)
    }

    private func sendSSE(_ event: ResponseSSEEvent, connection: NWConnection) async throws {
        let data = try JSONSerialization.data(withJSONObject: event.payload, options: [])
        let payload = "event: \(event.name)\ndata: \(String(decoding: data, as: UTF8.self))\n\n"
        try await connection.send(data: Data(payload.utf8))
    }

    private func sendJSON(_ object: [String: Any], status: Int, connection: NWConnection) async throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        try await sendJSONData(data, status: status, connection: connection)
    }

    private func sendJSONData(_ data: Data, status: Int, connection: NWConnection) async throws {
        let header = "HTTP/1.1 \(status) \(responsesReasonPhrase(status))\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(data.count)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + data)
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

private func responsesReasonPhrase(_ status: Int) -> String {
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
