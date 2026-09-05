import Foundation
import Network

struct ChatStreamResponse {
    let server: OpenAIServer
    let request: OpenAIChatRequest
    let promptTokens: Int
    let started: UInt64
    let connection: NWConnection
    let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
    let created = Int(Date().timeIntervalSince1970)

    func open() async throws {
        let header = "HTTP/1.1 200 OK\r\n"
            + "Content-Type: text/event-stream\r\n"
            + "Cache-Control: no-cache\r\n"
            + "Connection: close\r\n"
            + "X-Accel-Buffering: no\r\n\r\n"
        try await connection.send(data: Data(header.utf8))
        for chunk in chatStreamOpeningChunks(id: id, created: created, model: request.model) {
            try await server.sendSSE(chunk, connection: connection)
        }
    }

    func sendDelta(_ delta: (reasoning: String, content: String)) async throws {
        try await server.sendParserDelta(delta, id: id, created: created, model: request.model, connection: connection)
    }

    func fail(_ error: Error) async throws {
        try await server.sendSSE([
            "id": id, "object": "error", "created": created,
            "error": openAIErrorObject(message: String(describing: error), status: 500)
        ], connection: connection)
        try await connection.sendFinal(data: Data(chatStreamingDoneFrame.utf8))
        connection.cancel()
    }

    func finish(_ progress: ChatCompletionProgress) async throws {
        let ended = DispatchTime.now().uptimeNanoseconds
        try await server.sendSSE([
            "id": id, "object": "chat.completion.chunk", "created": created, "model": request.model,
            "choices": [["index": 0, "delta": ["token_ids": progress.tokenIDs],
                         "finish_reason": progress.finishReason]]
        ], connection: connection)
        if request.includeUsage {
            try await server.sendSSE([
                "id": id, "object": "chat.completion.chunk", "created": created, "model": request.model,
                "choices": [], "usage": progress.usage(promptTokens: promptTokens, started: started, ended: ended)
            ], connection: connection)
        }
        try await connection.sendFinal(data: Data(chatStreamingDoneFrame.utf8))
        connection.cancel()
    }

    func sendToolResult(_ result: ToolCallParseResult) async throws {
        try await server.sendSSE([
            "id": id, "object": "chat.completion.chunk", "created": created, "model": request.model,
            "choices": [["index": 0, "delta": ["tool_calls": toolCallDeltaDictionaries(from: result)],
                         "finish_reason": NSNull()]]
        ], connection: connection)
    }
}
