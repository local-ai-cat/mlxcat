import Foundation
import Network

struct CompletionsHandler {
    let backend: any OpenAICompletionBackend

    func handleCompletions(_ request: HTTPRequest, connection: NWConnection) async throws {
        let completionRequest: OpenAICompletionRequest
        do {
            completionRequest = try OpenAICompletionRequest.parse(request.body)
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 422
            try await sendJSON(openAIErrorBody(message: String(describing: error), status: status), status: status, connection: connection)
            return
        }

        let prompts = completionRequest.prompt.values
        if completionRequest.stream, prompts.count > 1 {
            try await sendJSON(
                openAIErrorBody(message: "streaming completions support a single prompt", status: 400),
                status: 400,
                connection: connection
            )
            return
        }

        if completionRequest.stream {
            let stream = try await backend.startCompletion(completionRequest.request(forPrompt: prompts[0]))
            try await sendStreaming(request: completionRequest, stream: stream, connection: connection)
        } else {
            try await sendBuffered(request: completionRequest, prompts: prompts, connection: connection)
        }
    }

    private func sendBuffered(
        request: OpenAICompletionRequest,
        prompts: [String],
        connection: NWConnection
    ) async throws {
        let id = "cmpl-\(UUID().uuidString.prefix(8))"
        let created = Int(Date().timeIntervalSince1970)
        var choices: [[String: Any]] = []
        var promptTokens = 0
        var completionTokens = 0
        var cachedPromptTokens = 0

        for (index, prompt) in prompts.enumerated() {
            let stream = try await backend.startCompletion(request.request(forPrompt: prompt))
            let completion = try await collectCompletion(stream: stream, stopSequences: request.stop)
            promptTokens += stream.promptTokens
            completionTokens += completion.completionTokens
            cachedPromptTokens += completion.cachedPromptTokens
            choices.append(
                buildCompletionChoice(
                    index: index,
                    text: completion.text,
                    finishReason: completion.finishReason
                )
            )
        }

        try await sendJSON(
            buildCompletionResponse(
                request: request,
                id: id,
                created: created,
                choices: choices,
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                cachedPromptTokens: cachedPromptTokens
            ),
            status: 200,
            connection: connection
        )
    }

    private func sendStreaming(
        request: OpenAICompletionRequest,
        stream: OpenAIChatStream,
        connection: NWConnection
    ) async throws {
        try await connection.send(data: Data(streamingHeader.utf8))

        let id = "cmpl-\(UUID().uuidString.prefix(8))"
        let created = Int(Date().timeIntervalSince1970)
        var formatter = CompletionStreamFormatter(
            id: id,
            model: request.model,
            created: created,
            stopSequences: request.stop
        )

        do {
            for try await chunk in stream.chunks {
                for payload in formatter.feed(chunk) {
                    try await sendSSE(payload, connection: connection)
                }
                if formatter.isStopped {
                    break
                }
            }
            for payload in formatter.finish() {
                try await sendSSE(payload, connection: connection)
            }
            if request.includeUsage {
                try await sendSSE(
                    [
                        "id": id,
                        "object": "text_completion",
                        "created": created,
                        "model": request.model,
                        "choices": [],
                        "usage": completionUsage(
                            promptTokens: stream.promptTokens,
                            completionTokens: formatter.completionTokens,
                            cachedPromptTokens: formatter.cachedPromptTokens
                        ),
                    ],
                    connection: connection
                )
            }
            try await connection.sendFinal(data: Data("data: [DONE]\n\n".utf8))
        } catch {
            try await sendSSE(
                [
                    "id": id,
                    "object": "error",
                    "created": created,
                    "error": openAIErrorObject(message: String(describing: error), status: 500),
                ],
                connection: connection
            )
            try await connection.sendFinal(data: Data("data: [DONE]\n\n".utf8))
        }
    }

    private func collectCompletion(stream: OpenAIChatStream, stopSequences: [String]) async throws -> CompletionBufferedResult {
        var text = ""
        var completionTokens = 0
        var cachedPromptTokens = 0
        var finishReason = "length"
        var stopMatcher = StreamingStopSequenceMatcher(stopSequences: stopSequences)
        var stoppedByTextStop = false

        for try await chunk in stream.chunks {
            completionTokens += 1
            cachedPromptTokens = max(cachedPromptTokens, chunk.cachedPromptTokens)
            if let chunkFinishReason = chunk.finishReason {
                finishReason = chunkFinishReason
            }
            let stopMatch = stopMatcher.feed(chunk.text)
            text += stopMatch.text
            if stopMatch.stopped {
                finishReason = "stop"
                stoppedByTextStop = true
                break
            }
        }

        if !stoppedByTextStop {
            let stopMatch = stopMatcher.finish()
            text += stopMatch.text
            if stopMatch.stopped {
                finishReason = "stop"
            }
        }
        return CompletionBufferedResult(
            text: text,
            completionTokens: completionTokens,
            finishReason: finishReason,
            cachedPromptTokens: cachedPromptTokens
        )
    }

    private func sendSSE(_ object: [String: Any], connection: NWConnection) async throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        let line = "data: \(String(decoding: data, as: UTF8.self))\n\n"
        try await connection.send(data: Data(line.utf8))
    }

    private func sendJSON(_ object: [String: Any], status: Int, connection: NWConnection) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let header = "HTTP/1.1 \(status) \(completionReasonPhrase(status))\r\n"
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

public struct CompletionBufferedResult: Sendable, Equatable {
    public let text: String
    public let completionTokens: Int
    public let finishReason: String
    public let cachedPromptTokens: Int

    public init(
        text: String,
        completionTokens: Int,
        finishReason: String,
        cachedPromptTokens: Int = 0
    ) {
        self.text = text
        self.completionTokens = completionTokens
        self.finishReason = finishReason
        self.cachedPromptTokens = cachedPromptTokens
    }
}

public struct CompletionStreamFormatter: Sendable {
    private let id: String
    private let model: String
    private let created: Int
    private var stopMatcher: StreamingStopSequenceMatcher
    private var finishReason = "length"
    private var emittedFinal = false
    public private(set) var completionTokens = 0
    public private(set) var cachedPromptTokens = 0
    public private(set) var isStopped = false

    public init(id: String, model: String, created: Int, stopSequences: [String]) {
        self.id = id
        self.model = model
        self.created = created
        self.stopMatcher = StreamingStopSequenceMatcher(stopSequences: stopSequences)
    }

    public mutating func feed(_ chunk: OpenAIChatChunk) -> [[String: Any]] {
        guard !isStopped else { return [] }
        completionTokens += 1
        cachedPromptTokens = max(cachedPromptTokens, chunk.cachedPromptTokens)
        if let chunkFinishReason = chunk.finishReason {
            finishReason = chunkFinishReason
        }
        let stopMatch = stopMatcher.feed(chunk.text)
        if stopMatch.stopped {
            finishReason = "stop"
            isStopped = true
        }
        guard !stopMatch.text.isEmpty else { return [] }
        return [chunkPayload(text: stopMatch.text, finishReason: nil)]
    }

    public mutating func finish() -> [[String: Any]] {
        guard !emittedFinal else { return [] }
        var payloads: [[String: Any]] = []
        if !isStopped {
            let stopMatch = stopMatcher.finish()
            if stopMatch.stopped {
                finishReason = "stop"
                isStopped = true
            }
            if !stopMatch.text.isEmpty {
                payloads.append(chunkPayload(text: stopMatch.text, finishReason: nil))
            }
        }
        emittedFinal = true
        payloads.append(chunkPayload(text: "", finishReason: finishReason))
        return payloads
    }

    private func chunkPayload(text: String, finishReason: String?) -> [String: Any] {
        [
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                buildCompletionChoice(
                    index: 0,
                    text: text,
                    finishReason: finishReason
                )
            ],
        ]
    }
}

public func buildCompletionChoice(index: Int, text: String, finishReason: String?) -> [String: Any] {
    [
        "index": index,
        "text": text,
        "finish_reason": finishReason ?? NSNull(),
        "logprobs": NSNull(),
    ]
}

public func buildCompletionResponse(
    request: OpenAICompletionRequest,
    id: String,
    created: Int,
    choices: [[String: Any]],
    promptTokens: Int,
    completionTokens: Int,
    cachedPromptTokens: Int = 0
) -> [String: Any] {
    [
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": choices,
        "usage": completionUsage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            cachedPromptTokens: cachedPromptTokens
        ),
    ]
}

public func completionUsage(
    promptTokens: Int,
    completionTokens: Int,
    cachedPromptTokens: Int = 0
) -> [String: Any] {
    [
        "prompt_tokens": promptTokens,
        "completion_tokens": completionTokens,
        "total_tokens": promptTokens + completionTokens,
        "input_tokens": promptTokens,
        "output_tokens": completionTokens,
        "prompt_tokens_details": ["cached_tokens": cachedPromptTokens],
    ]
}

private func completionReasonPhrase(_ status: Int) -> String {
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
