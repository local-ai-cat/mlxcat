import Foundation
import Network

public struct OpenAIModelInfo: Sendable {
    public let id: String
    public let maxModelLength: Int?
    public let modelType: String

    public init(id: String, maxModelLength: Int? = nil, modelType: String = "llm") {
        self.id = id
        self.maxModelLength = maxModelLength
        self.modelType = modelType
    }
}

public struct OpenAIChatImageReference: Sendable, Equatable {
    public enum Source: Sendable, Equatable {
        case dataURI(String)
        case url(String)
    }

    public let source: Source

    public init(source: Source) {
        self.source = source
    }
}

/// An audio or video attachment the MODEL consumes, as opposed to
/// `/v1/audio/transcriptions`, which is speech-to-text and unrelated.
///
/// Same two sources as an image reference: a URL the engine can open, or an inline
/// base64 data URI. Kept as a separate type rather than reusing
/// `OpenAIChatImageReference` because "image" in the name of the thing carrying a video
/// is the sort of detail that later reads as a bug.
public struct OpenAIChatMediaReference: Sendable, Equatable {
    public enum Source: Sendable, Equatable {
        case dataURI(String)
        case url(String)
    }

    public let source: Source

    public init(source: Source) {
        self.source = source
    }
}

public struct OpenAIChatMessage: Sendable {
    public let role: String
    public let content: String
    public let reasoningContent: String?
    public let imageReferences: [OpenAIChatImageReference]
    /// Audio the model consumes. Audio/video-capable models (gemma-4) use them; models
    /// without those towers ignore them. There is deliberately NO text-description
    /// fallback at this layer — deciding to describe media into text is a caller's
    /// policy, not the engine's.
    public let audioReferences: [OpenAIChatMediaReference]
    public let videoReferences: [OpenAIChatMediaReference]

    public init(
        role: String,
        content: String,
        reasoningContent: String? = nil,
        imageReferences: [OpenAIChatImageReference] = [],
        audioReferences: [OpenAIChatMediaReference] = [],
        videoReferences: [OpenAIChatMediaReference] = []
    ) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.imageReferences = imageReferences
        self.audioReferences = audioReferences
        self.videoReferences = videoReferences
    }
}

public enum OpenAIJSONValue: Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: OpenAIJSONValue])
    case array([OpenAIJSONValue])
    case null

    init?(_ value: Any) {
        switch value {
        case let string as String:
            self = .string(string)
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                self = .bool(number.boolValue)
            } else {
                self = .number(number.doubleValue)
            }
        case let object as [String: Any]:
            var converted: [String: OpenAIJSONValue] = [:]
            for (key, value) in object {
                guard let jsonValue = OpenAIJSONValue(value) else { return nil }
                converted[key] = jsonValue
            }
            self = .object(converted)
        case let array as [Any]:
            var converted: [OpenAIJSONValue] = []
            for value in array {
                guard let jsonValue = OpenAIJSONValue(value) else { return nil }
                converted.append(jsonValue)
            }
            self = .array(converted)
        case _ as NSNull:
            self = .null
        default:
            return nil
        }
    }
}

public struct OpenAIChatRequest: Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let repetitionPenalty: Float
    public let minP: Float
    public let xtcProbability: Float
    public let xtcThreshold: Float
    public let presencePenalty: Float
    public let frequencyPenalty: Float
    public let minTokens: Int
    public let logitBias: [Int: Float]
    public let stop: [String]
    public let seed: Int?
    public let stream: Bool
    public let includeUsage: Bool
    public let enableThinking: Bool?
    public let thinkingBudget: Int?
    public let chatTemplateKwargs: [String: OpenAIJSONValue]?
    public let structuredOutput: StructuredOutputSpec
    public let tools: [OpenAIJSONValue]?
    public let toolChoice: OpenAIToolChoice?
    public let cacheSession: String?

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        maxTokens: Int,
        temperature: Float = 0,
        topP: Float = 0,
        topK: Int = 0,
        repetitionPenalty: Float = 1,
        minP: Float = 0,
        xtcProbability: Float = 0,
        xtcThreshold: Float = 0.1,
        presencePenalty: Float = 0,
        frequencyPenalty: Float = 0,
        minTokens: Int = 0,
        logitBias: [Int: Float] = [:],
        stop: [String] = [],
        seed: Int? = nil,
        stream: Bool = false,
        includeUsage: Bool = false,
        enableThinking: Bool? = nil,
        thinkingBudget: Int? = nil,
        chatTemplateKwargs: [String: OpenAIJSONValue]? = nil,
        structuredOutput: StructuredOutputSpec = .none,
        tools: [OpenAIJSONValue]? = nil,
        toolChoice: OpenAIToolChoice? = nil,
        cacheSession: String? = nil
    ) {
        self.model = model
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.minP = minP
        self.xtcProbability = xtcProbability
        self.xtcThreshold = xtcThreshold
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.minTokens = max(0, minTokens)
        self.logitBias = logitBias
        self.stop = stop
        self.seed = seed
        self.stream = stream
        self.includeUsage = includeUsage
        self.enableThinking = enableThinking
        self.thinkingBudget = thinkingBudget
        self.chatTemplateKwargs = chatTemplateKwargs
        self.structuredOutput = structuredOutput
        self.tools = tools
        self.toolChoice = toolChoice
        self.cacheSession = cacheSession
    }
}

public struct OpenAIChatChunk: Sendable {
    public let text: String
    public let tokenID: Int
    public let finishReason: String?
    public let cachedPromptTokens: Int

    public init(
        text: String,
        tokenID: Int,
        finishReason: String? = nil,
        cachedPromptTokens: Int = 0
    ) {
        self.text = text
        self.tokenID = tokenID
        self.finishReason = finishReason
        self.cachedPromptTokens = cachedPromptTokens
    }
}

public struct OpenAIChatStream: Sendable {
    public let promptTokens: Int
    public let chunks: AsyncThrowingStream<OpenAIChatChunk, Error>

    public init(promptTokens: Int, chunks: AsyncThrowingStream<OpenAIChatChunk, Error>) {
        self.promptTokens = promptTokens
        self.chunks = chunks
    }
}

public extension OpenAIChatRequest {
    internal var startInThinking: Bool {
        enableThinking ?? (thinkingBudget != nil)
    }

    func withCacheSession(_ cacheSession: String?) -> OpenAIChatRequest {
        OpenAIChatRequest(
            model: model,
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            minP: minP,
            xtcProbability: xtcProbability,
            xtcThreshold: xtcThreshold,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty,
            minTokens: minTokens,
            logitBias: logitBias,
            stop: stop,
            seed: seed,
            stream: stream,
            includeUsage: includeUsage,
            enableThinking: enableThinking,
            thinkingBudget: thinkingBudget,
            chatTemplateKwargs: chatTemplateKwargs,
            structuredOutput: structuredOutput,
            tools: tools,
            toolChoice: toolChoice,
            cacheSession: cacheSession
        )
    }
}

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

private final class ListenerStartState: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?

    init(_ continuation: CheckedContinuation<Void, Error>) {
        self.continuation = continuation
    }

    func resume(_ result: Result<Void, Error>) {
        lock.lock()
        let continuation = continuation
        self.continuation = nil
        lock.unlock()

        guard let continuation else { return }
        switch result {
        case .success:
            continuation.resume(returning: ())
        case .failure(let error):
            continuation.resume(throwing: error)
        }
    }
}

public final class OpenAIServer: @unchecked Sendable {
    private let host: NWEndpoint.Host
    private let port: NWEndpoint.Port
    private let backend: any OpenAIChatBackend
    private let listener: NWListener
    private let responsesStore: ResponsesStore
    private let mcpManager: MCPManager?
    private let connectionsLock = NSLock()
    private var openConnections: [ObjectIdentifier: NWConnection] = [:]
    private var isStopped = false

    public init(
        host: String = "127.0.0.1",
        port: UInt16,
        backend: any OpenAIChatBackend,
        mcpManager: MCPManager? = nil
    ) throws {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw OpenAIServerError.invalidPort(port)
        }
        self.host = NWEndpoint.Host(host)
        self.port = nwPort
        self.backend = backend
        // Loopback hosts must actually bind loopback-only: the `host` parameter
        // used to be stored but never applied, so the listener accepted
        // connections on every interface regardless of it.
        let parameters: NWParameters = .tcp
        if ["127.0.0.1", "localhost", "::1"].contains(host) {
            parameters.requiredInterfaceType = .loopback
        }
        self.listener = try NWListener(using: parameters, on: nwPort)
        self.responsesStore = ResponsesStore()
        self.mcpManager = mcpManager
    }

    /// The port the listener is actually bound to — differs from the requested
    /// port when constructed with port 0 (ephemeral). Nil until `start()` returns.
    public var boundPort: UInt16? {
        listener.port?.rawValue
    }

    public func start() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let startState = ListenerStartState(continuation)
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    startState.resume(.success(()))
                case .failed(let error):
                    startState.resume(.failure(OpenAIServerError.listenerFailed(String(describing: error))))
                case .cancelled:
                    startState.resume(.failure(OpenAIServerError.listenerFailed("listener cancelled")))
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                guard let self, self.track(connection) else {
                    connection.cancel()
                    return
                }
                connection.start(queue: .global(qos: .userInitiated))
                Task {
                    await self.handle(connection)
                    self.untrack(connection)
                }
            }
            listener.start(queue: .global(qos: .userInitiated))
        }
    }

    /// Stops accepting new connections and cancels the in-flight ones, releasing
    /// the port. Idempotent. Embedded hosts (in-app serving) use this; the CLI
    /// never stops. The instance cannot be restarted — create a new server.
    public func stop() {
        connectionsLock.lock()
        let alreadyStopped = isStopped
        isStopped = true
        let connections = Array(openConnections.values)
        openConnections.removeAll()
        connectionsLock.unlock()

        guard !alreadyStopped else { return }
        listener.cancel()
        for connection in connections {
            connection.cancel()
        }
    }

    private func track(_ connection: NWConnection) -> Bool {
        connectionsLock.lock()
        defer { connectionsLock.unlock() }
        guard !isStopped else { return false }
        openConnections[ObjectIdentifier(connection)] = connection
        return true
    }

    private func untrack(_ connection: NWConnection) {
        connectionsLock.lock()
        openConnections.removeValue(forKey: ObjectIdentifier(connection))
        connectionsLock.unlock()
        // NWConnection keeps its file descriptor until cancel() is called, even
        // after the peer closes (lsof shows the socket parked in (CLOSED)). This
        // server is connection-per-request, so every handled connection ends
        // here — without this cancel the process leaks one fd per request and
        // wedges at the 256-fd rlimit (~245 requests served, then EMFILE).
        connection.cancel()
    }

    public func waitForever() {
        DispatchSemaphore(value: 0).wait()
    }

    private func handle(_ connection: NWConnection) async {
        do {
            let request = try await HTTPRequest.read(from: connection)
            switch (request.method, request.path) {
            case ("GET", "/health"):
                try await sendJSON(healthResponse(), status: 200, connection: connection)
            case ("GET", "/v1/models"):
                try await sendJSON(modelsResponse(), status: 200, connection: connection)
            case ("GET", "/v1/models/status"):
                try await sendLifecycleResponse(request, connection: connection)
            case ("GET", "/v1/mcp/tools"):
                try await MCPHandler(manager: mcpManager).handleTools(connection: connection)
            case ("GET", "/v1/mcp/servers"):
                try await MCPHandler(manager: mcpManager).handleServers(connection: connection)
            case ("POST", "/v1/mcp/execute"):
                try await MCPHandler(manager: mcpManager).handleExecute(request, connection: connection)
            case ("POST", "/v1/chat/completions"):
                try await handleChatCompletion(request, connection: connection)
            case ("POST", "/v1/completions"):
                guard let completionBackend = backend as? any OpenAICompletionBackend else {
                    try await sendJSON(openAIErrorBody(message: "completions backend unavailable", status: 404), status: 404, connection: connection)
                    return
                }
                try await CompletionsHandler(backend: completionBackend).handleCompletions(request, connection: connection)
            case ("POST", "/v1/embeddings"):
                guard let embeddingsBackend = backend as? any OpenAIEmbeddingsBackend else {
                    try await sendJSON(openAIErrorBody(message: "embeddings backend unavailable", status: 404), status: 404, connection: connection)
                    return
                }
                try await EmbeddingsHandler(backend: embeddingsBackend).handleEmbeddings(request, connection: connection)
            case ("POST", "/v1/rerank"):
                guard let rerankBackend = backend as? any OpenAIRerankBackend else {
                    try await sendJSON(openAIErrorBody(message: "rerank backend unavailable", status: 404), status: 404, connection: connection)
                    return
                }
                try await RerankHandler(backend: rerankBackend).handleRerank(request, connection: connection)
            case ("POST", "/v1/audio/transcriptions"):
                guard let audioBackend = backend as? any AudioTranscriptionBackend, !audioBackend.transcriptionModels.isEmpty else {
                    try await sendJSON(
                        openAIErrorBody(message: "transcription backend not configured", status: 501),
                        status: 501,
                        connection: connection
                    )
                    return
                }
                try await AudioTranscriptionHandler(backend: audioBackend).handleTranscription(request, connection: connection)
            case ("POST", "/v1/messages"):
                try await AnthropicMessagesHandler(backend: backend, mcpManager: mcpManager).handleMessages(request, connection: connection)
            case ("POST", "/v1/messages/count_tokens"):
                try await AnthropicMessagesHandler(backend: backend).handleCountTokens(request, connection: connection)
            case ("POST", "/v1/responses"):
                try await ResponsesHandler(backend: backend, store: responsesStore, mcpManager: mcpManager).handleCreate(request, connection: connection)
            default:
                if request.method == "GET", let id = responseID(from: request.path) {
                    try await ResponsesHandler(backend: backend, store: responsesStore).handleGet(id: id, connection: connection)
                    return
                }
                if request.method == "DELETE", let id = responseID(from: request.path) {
                    try await ResponsesHandler(backend: backend, store: responsesStore).handleDelete(id: id, connection: connection)
                    return
                }
                if isModelLifecyclePath(request.path) {
                    try await sendLifecycleResponse(request, connection: connection)
                    return
                }
                try await sendJSON(openAIErrorBody(message: "not found", status: 404), status: 404, connection: connection)
            }
        } catch {
            try? await sendError(error, connection: connection)
        }
    }

    func lifecycleRouteResponseForTesting(_ request: HTTPRequest) async -> HTTPJSONResponse {
        do {
            if let response = try await lifecycleResponse(for: request) {
                return response
            }
            return HTTPJSONResponse(
                status: 404,
                body: openAIErrorBody(message: "not found", status: 404)
            )
        } catch {
            return errorResponse(for: error)
        }
    }

    func modelsResponseForTesting() -> [String: Any] {
        modelsResponse()
    }

    private func responseID(from path: String) -> String? {
        let prefix = "/v1/responses/"
        guard path.hasPrefix(prefix) else { return nil }
        let id = String(path.dropFirst(prefix.count))
        return id.isEmpty ? nil : id
    }

    private func sendLifecycleResponse(_ request: HTTPRequest, connection: NWConnection) async throws {
        guard let response = try await lifecycleResponse(for: request) else {
            try await sendJSON(openAIErrorBody(message: "not found", status: 404), status: 404, connection: connection)
            return
        }

        try await sendJSON(
            response.body,
            status: response.status,
            headers: response.headers,
            connection: connection
        )
    }

    private func lifecycleResponse(for request: HTTPRequest) async throws -> HTTPJSONResponse? {
        guard let lifecycleBackend = backend as? any OpenAIModelLifecycleBackend else {
            return nil
        }

        if request.method == "GET", request.path == "/v1/models/status" {
            return HTTPJSONResponse(status: 200, body: modelStatusBody(try await lifecycleBackend.modelPoolStatus()))
        }

        if request.method == "POST", let modelID = lifecycleModelID(from: request.path, action: "load") {
            return HTTPJSONResponse(
                status: 200,
                body: lifecycleResultBody(try await lifecycleBackend.loadModel(modelID))
            )
        }

        if request.method == "POST", let modelID = lifecycleModelID(from: request.path, action: "unload") {
            return HTTPJSONResponse(
                status: 200,
                body: lifecycleResultBody(try await lifecycleBackend.unloadModel(modelID))
            )
        }

        return nil
    }

    private func isModelLifecyclePath(_ path: String) -> Bool {
        lifecycleModelID(from: path, action: "load") != nil
            || lifecycleModelID(from: path, action: "unload") != nil
    }

    private func lifecycleModelID(from path: String, action: String) -> String? {
        let prefix = "/v1/models/"
        let suffix = "/\(action)"
        guard path.hasPrefix(prefix), path.hasSuffix(suffix) else { return nil }
        let idStart = path.index(path.startIndex, offsetBy: prefix.count)
        let idEnd = path.index(path.endIndex, offsetBy: -suffix.count)
        guard idStart < idEnd else { return nil }
        return String(path[idStart..<idEnd])
    }

    private func handleChatCompletion(_ request: HTTPRequest, connection: NWConnection) async throws {
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

    private func sendStreamingChat(
        request: OpenAIChatRequest,
        stream: OpenAIChatStream,
        started: UInt64,
        connection: NWConnection
    ) async throws {
        if toolsRequested(request) {
            try await sendStreamingChatWithTools(
                request: request,
                stream: stream,
                started: started,
                connection: connection
            )
            return
        }

        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let created = Int(Date().timeIntervalSince1970)
        var firstToken: UInt64?
        var lastToken: UInt64?
        var completionTokens = 0
        var cachedPromptTokens = 0
        var finishReason = "length"
        // Raw generated token ids captured before the thinking parser reshapes text,
        // flushed on the terminal chunk delta for the migration-gate parity oracle.
        var streamTokenIDs: [Int] = []
        // Stop matching runs on raw decoded model text before thinking tags are split,
        // matching omlx's output_text-level stop behavior.
        var stopMatcher = StreamingStopSequenceMatcher(stopSequences: request.stop)
        var thinkingParser = ThinkingParser(startInThinking: request.startInThinking)
        var stoppedByTextStop = false
        let header = "HTTP/1.1 200 OK\r\n"
            + "Content-Type: text/event-stream\r\n"
            + "Cache-Control: no-cache\r\n"
            + "Connection: close\r\n"
            + "X-Accel-Buffering: no\r\n"
            + "\r\n"

        try await connection.send(
            data: Data(header.utf8)
        )
        for chunk in chatStreamOpeningChunks(id: id, created: created, model: request.model) {
            try await sendSSE(chunk, connection: connection)
        }

        do {
            for try await chunk in stream.chunks {
                let now = DispatchTime.now().uptimeNanoseconds
                if firstToken == nil {
                    firstToken = now
                }
                lastToken = now
                completionTokens += 1
                cachedPromptTokens = max(cachedPromptTokens, chunk.cachedPromptTokens)
                streamTokenIDs.append(chunk.tokenID)
                if let chunkFinishReason = chunk.finishReason {
                    finishReason = chunkFinishReason
                }
                let stopMatch = stopMatcher.feed(chunk.text)
                if stopMatch.stopped {
                    finishReason = "stop"
                    stoppedByTextStop = true
                }
                if !stopMatch.text.isEmpty {
                    try await sendParserDelta(
                        thinkingParser.feed(stopMatch.text),
                        id: id,
                        created: created,
                        model: request.model,
                        connection: connection
                    )
                }
                if stoppedByTextStop {
                    break
                }
            }
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
            try await connection.sendFinal(data: Data(chatStreamingDoneFrame.utf8))
            connection.cancel()
            return
        }

        if !stoppedByTextStop {
            let stopMatch = stopMatcher.finish()
            if stopMatch.stopped {
                finishReason = "stop"
                stoppedByTextStop = true
            }
            if !stopMatch.text.isEmpty {
                try await sendParserDelta(
                    thinkingParser.feed(stopMatch.text),
                    id: id,
                    created: created,
                    model: request.model,
                    connection: connection
                )
            }
        }
        try await sendParserDelta(
            thinkingParser.finish(),
            id: id,
            created: created,
            model: request.model,
            connection: connection
        )

        let ended = DispatchTime.now().uptimeNanoseconds
        try await sendSSE(
            [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    [
                        "index": 0,
                        "delta": ["token_ids": streamTokenIDs],
                        "finish_reason": finishReason,
                    ]
                ],
            ],
            connection: connection
        )
        if request.includeUsage {
            try await sendSSE(
                [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [],
                    "usage": usage(
                        promptTokens: stream.promptTokens,
                        cachedPromptTokens: cachedPromptTokens,
                        completionTokens: completionTokens,
                        started: started,
                        firstToken: firstToken ?? ended,
                        lastToken: lastToken ?? ended,
                        ended: ended
                    ),
                ],
                connection: connection
            )
        }
        try await connection.sendFinal(data: Data(chatStreamingDoneFrame.utf8))
        connection.cancel()
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

    private func sendStreamingChatWithTools(
        request: OpenAIChatRequest,
        stream: OpenAIChatStream,
        started: UInt64,
        connection: NWConnection
    ) async throws {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let created = Int(Date().timeIntervalSince1970)
        var firstToken: UInt64?
        var lastToken: UInt64?
        var completionTokens = 0
        var cachedPromptTokens = 0
        var finishReason = "length"
        var text = ""
        // Raw generated token ids captured before the thinking parser reshapes text,
        // flushed on the terminal chunk delta for the migration-gate parity oracle.
        var streamTokenIDs: [Int] = []
        // Stop matching runs on raw decoded model text before thinking tags are split,
        // matching omlx's output_text-level stop behavior.
        var stopMatcher = StreamingStopSequenceMatcher(stopSequences: request.stop)
        var stoppedByTextStop = false

        try await connection.send(
            data: Data(
                (
                    "HTTP/1.1 200 OK\r\n"
                        + "Content-Type: text/event-stream\r\n"
                        + "Cache-Control: no-cache\r\n"
                        + "Connection: close\r\n"
                        + "X-Accel-Buffering: no\r\n"
                        + "\r\n"
                ).utf8
            )
        )
        for chunk in chatStreamOpeningChunks(id: id, created: created, model: request.model) {
            try await sendSSE(chunk, connection: connection)
        }

        do {
            for try await chunk in stream.chunks {
                let now = DispatchTime.now().uptimeNanoseconds
                if firstToken == nil {
                    firstToken = now
                }
                lastToken = now
                completionTokens += 1
                cachedPromptTokens = max(cachedPromptTokens, chunk.cachedPromptTokens)
                streamTokenIDs.append(chunk.tokenID)
                if let chunkFinishReason = chunk.finishReason {
                    finishReason = chunkFinishReason
                }
                let stopMatch = stopMatcher.feed(chunk.text)
                text += stopMatch.text
                if stopMatch.stopped {
                    finishReason = "stop"
                    stoppedByTextStop = true
                }
                if stoppedByTextStop {
                    break
                }
            }
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
            try await connection.sendFinal(data: Data(chatStreamingDoneFrame.utf8))
            connection.cancel()
            return
        }

        if !stoppedByTextStop {
            let stopMatch = stopMatcher.finish()
            text += stopMatch.text
            if stopMatch.stopped {
                finishReason = "stop"
            }
        }

        let result = parseModelOutput(
            text,
            model: request.model,
            startInThinking: request.startInThinking
        )
        let parsed = ToolCallParseResult(content: result.content, toolCalls: result.toolCalls)
        if parsed.toolCalls.isEmpty {
            try await sendParserDelta(
                (result.reasoning, result.content),
                id: id,
                created: created,
                model: request.model,
                connection: connection
            )
        } else {
            if !result.reasoning.isEmpty {
                try await sendParserDelta(
                    (result.reasoning, ""),
                    id: id,
                    created: created,
                    model: request.model,
                    connection: connection
                )
            }
            if !parsed.content.isEmpty {
                try await sendParserDelta(
                    ("", parsed.content),
                    id: id,
                    created: created,
                    model: request.model,
                    connection: connection
                )
            }
            try await sendSSE(
                [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        [
                            "index": 0,
                            "delta": ["tool_calls": toolCallDeltaDictionaries(from: parsed)],
                            "finish_reason": NSNull(),
                        ]
                    ],
                ],
                connection: connection
            )
            finishReason = finishReasonForToolCalls(
                defaultFinishReason: finishReason,
                parsed: parsed
            )
        }

        let ended = DispatchTime.now().uptimeNanoseconds
        try await sendSSE(
            [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    [
                        "index": 0,
                        "delta": ["token_ids": streamTokenIDs],
                        "finish_reason": finishReason,
                    ]
                ],
            ],
            connection: connection
        )
        if request.includeUsage {
            try await sendSSE(
                [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [],
                    "usage": usage(
                        promptTokens: stream.promptTokens,
                        cachedPromptTokens: cachedPromptTokens,
                        completionTokens: completionTokens,
                        started: started,
                        firstToken: firstToken ?? ended,
                        lastToken: lastToken ?? ended,
                        ended: ended
                    ),
                ],
                connection: connection
            )
        }
        try await connection.sendFinal(data: Data(chatStreamingDoneFrame.utf8))
        connection.cancel()
    }

    private func sendBufferedChat(
        request: OpenAIChatRequest,
        stream: OpenAIChatStream,
        started: UInt64,
        connection: NWConnection
    ) async throws {
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        var firstToken: UInt64?
        var lastToken: UInt64?
        var completionTokens = 0
        var cachedPromptTokens = 0
        var finishReason = "length"
        var text = ""
        // Raw generated token ids, in order, for the migration-gate parity oracle.
        // Captured before stop-sequence text trimming so the sequence stays aligned
        // with completion_tokens and is identical across platforms running this engine.
        var tokenIDs: [Int] = []
        // Stop matching runs on raw decoded model text before thinking tags are split,
        // matching omlx's output_text-level stop behavior.
        var stopMatcher = StreamingStopSequenceMatcher(stopSequences: request.stop)
        var stoppedByTextStop = false

        for try await chunk in stream.chunks {
            let now = DispatchTime.now().uptimeNanoseconds
            if firstToken == nil {
                firstToken = now
            }
            lastToken = now
            completionTokens += 1
            cachedPromptTokens = max(cachedPromptTokens, chunk.cachedPromptTokens)
            tokenIDs.append(chunk.tokenID)
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
            finishReason = finishReasonForToolCalls(
                defaultFinishReason: finishReason,
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
                        "finish_reason": finishReason,
                        "token_ids": tokenIDs,
                    ]
                ],
                "usage": usage(
                    promptTokens: stream.promptTokens,
                    cachedPromptTokens: cachedPromptTokens,
                    completionTokens: completionTokens,
                    started: started,
                    firstToken: firstToken ?? ended,
                    lastToken: lastToken ?? ended,
                    ended: ended
                ),
            ],
            status: 200,
            connection: connection
        )
    }

    private func modelsResponse() -> [String: Any] {
        var models = backend.models
        if let embeddingsBackend = backend as? any OpenAIEmbeddingsBackend {
            models.append(contentsOf: embeddingsBackend.embeddingModels)
        }
        if let rerankBackend = backend as? any OpenAIRerankBackend {
            models.append(contentsOf: rerankBackend.rerankModels)
        }
        if let audioBackend = backend as? any AudioTranscriptionBackend {
            models.append(contentsOf: audioBackend.transcriptionModels)
        }

        return [
            "object": "list",
            "data": models.map { model in
                var payload: [String: Any] = [
                    "id": model.id,
                    "object": "model",
                    "created": Int(Date().timeIntervalSince1970),
                    "owned_by": "mlxserve-native",
                    "model_type": model.modelType,
                ]
                if let maxModelLength = model.maxModelLength {
                    payload["max_model_len"] = maxModelLength
                }
                return payload
            },
        ]
    }

    private func healthResponse() -> [String: Any] {
        if let healthProvider = backend as? any OpenAIHealthProviding {
            return buildHealthResponse(healthProvider.healthInfo)
        }
        return buildHealthResponse(
            OpenAIHealthInfo(
                defaultModel: backend.models.first?.id,
                enginePool: nil
            )
        )
    }

    private func toolsRequested(_ request: OpenAIChatRequest) -> Bool {
        selectOpenAITools(tools: request.tools, toolChoice: request.toolChoice) != nil
    }

    private func usage(
        promptTokens: Int,
        cachedPromptTokens: Int,
        completionTokens: Int,
        started: UInt64,
        firstToken: UInt64,
        lastToken: UInt64,
        ended: UInt64
    ) -> [String: Any] {
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
            "generation_tokens_per_second": Double(completionTokens) / generation,
        ]
    }

    private func seconds(_ nanoseconds: UInt64) -> Double {
        Double(nanoseconds) / 1_000_000_000
    }

    private func sendSSE(_ object: [String: Any], connection: NWConnection) async throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        let line = "data: \(String(decoding: data, as: UTF8.self))\n\n"
        try await connection.send(data: Data(line.utf8))
    }


    private func sendParserDelta(
        _ delta: (reasoning: String, content: String),
        id: String,
        created: Int,
        model: String,
        connection: NWConnection
    ) async throws {
        guard !delta.reasoning.isEmpty || !delta.content.isEmpty else { return }

        var deltaFields = ""
        if !delta.reasoning.isEmpty {
            deltaFields += "\"reasoning_content\":\"\(jsonEscape(delta.reasoning))\""
        }
        if !delta.content.isEmpty {
            if !deltaFields.isEmpty { deltaFields += "," }
            deltaFields += "\"content\":\"\(jsonEscape(delta.content))\""
        }

        let line = "data: {\"id\":\"\(id)\",\"object\":\"chat.completion.chunk\","
            + "\"created\":\(created),\"model\":\"\(jsonEscape(model))\","
            + "\"choices\":[{\"index\":0,\"delta\":{\(deltaFields)},"
            + "\"finish_reason\":null}]}\n\n"
        try await connection.send(data: Data(line.utf8))
    }

    private func jsonEscape(_ s: String) -> String {
        guard s.contains(where: { $0 == "\\" || $0 == "\"" || $0 == "\n" || $0 == "\r" || $0 == "\t" || $0.asciiValue.map({ $0 < 0x20 }) == true }) else {
            return s
        }
        var result = ""
        result.reserveCapacity(s.count + 8)
        for c in s {
            switch c {
            case "\\": result += "\\\\"
            case "\"": result += "\\\""
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            default:
                if let ascii = c.asciiValue, ascii < 0x20 {
                    result += String(format: "\\u%04x", ascii)
                } else {
                    result.append(c)
                }
            }
        }
        return result
    }

    private func sendJSON(
        _ object: [String: Any],
        status: Int,
        headers: [String: String] = [:],
        connection: NWConnection
    ) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let reason = httpReasonPhrase(status)
        var header = "HTTP/1.1 \(status) \(reason)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Connection: close\r\n"
        for (key, value) in headers.sorted(by: { $0.key < $1.key }) {
            header += "\(key): \(value)\r\n"
        }
        header += "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + body)
    }

    private func sendError(_ error: Error, connection: NWConnection) async throws {
        let response = errorResponse(for: error)
        try await sendJSON(
            response.body,
            status: response.status,
            headers: response.headers,
            connection: connection
        )
    }
}

struct HTTPJSONResponse {
    let status: Int
    let body: [String: Any]
    let headers: [String: String]

    init(status: Int, body: [String: Any], headers: [String: String] = [:]) {
        self.status = status
        self.body = body
        self.headers = headers
    }
}

let chatStreamingDoneFrame = "data: [DONE]\n\n"

func chatStreamOpeningChunks(id: String, created: Int, model: String) -> [[String: Any]] {
    [
        chatKeepalivePrimingChunk(id: id),
        [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                [
                    "index": 0,
                    "delta": ["role": "assistant"],
                    "finish_reason": NSNull(),
                ]
            ],
        ],
    ]
}

func chatKeepalivePrimingChunk(id: String) -> [String: Any] {
    [
        "id": id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "keepalive",
        "choices": [
            [
                "index": 0,
                "delta": ["content": ""],
                "finish_reason": NSNull(),
            ]
        ],
    ]
}

private func modelStatusBody(_ status: OpenAIModelPoolStatus) -> [String: Any] {
    [
        "final_ceiling": status.finalCeiling,
        "current_model_memory": status.currentModelMemory,
        "model_count": status.modelCount,
        "loaded_count": status.loadedCount,
        "models": status.models.map { model in
            [
                "id": model.id,
                "model_type": model.modelType,
                "model_path": model.modelPath,
                "loaded": model.loaded,
                "is_loading": model.isLoading,
                "estimated_size": model.estimatedSize,
                "actual_size": model.actualSize.map { $0 as Any } ?? NSNull(),
                "pinned": model.pinned,
                "last_access": model.lastAccess.map { $0 as Any } ?? NSNull(),
                "in_use": model.inUse,
            ] as [String: Any]
        },
    ]
}

private func lifecycleResultBody(_ result: OpenAIModelLifecycleResult) -> [String: Any] {
    var body: [String: Any] = [
        "status": result.status,
        "model_id": result.modelID,
    ]
    if let message = result.message {
        body["message"] = message
    }
    return body
}

private func errorResponse(for error: Error) -> HTTPJSONResponse {
    let httpError = openAIHTTPError(from: error)
    var headers: [String: String] = [:]
    if let retryAfterSeconds = httpError.retryAfterSeconds {
        headers["Retry-After"] = "\(retryAfterSeconds)"
    }
    return HTTPJSONResponse(
        status: httpError.status,
        body: openAIErrorBody(message: httpError.message, status: httpError.status, code: httpError.code),
        headers: headers
    )
}

private func openAIHTTPError(from error: Error) -> OpenAIHTTPError {
    if let httpError = error as? OpenAIHTTPError {
        return httpError
    }
    if let serverError = error as? OpenAIServerError {
        return OpenAIHTTPError(status: serverError.httpStatus, message: String(describing: serverError))
    }
    return OpenAIHTTPError(status: 500, message: String(describing: error))
}

private func httpReasonPhrase(_ status: Int) -> String {
    switch status {
    case 200:
        return "OK"
    case 400:
        return "Bad Request"
    case 401:
        return "Unauthorized"
    case 404:
        return "Not Found"
    case 409:
        return "Conflict"
    case 413:
        return "Payload Too Large"
    case 422:
        return "Unprocessable Entity"
    case 429:
        return "Too Many Requests"
    case 503:
        return "Service Unavailable"
    case 507:
        return "Insufficient Storage"
    case 500:
        return "Internal Server Error"
    case 501:
        return "Not Implemented"
    default:
        return "OK"
    }
}

public enum OpenAIServerError: Error, Equatable, CustomStringConvertible {
    case invalidPort(UInt16)
    case invalidRequest
    case invalidJSON
    case missingField(String)
    case unsupportedContent
    case invalidContentLength
    case payloadTooLarge
    case listenerFailed(String)
    case invalidStructuredOutput(String)

    var httpStatus: Int {
        switch self {
        case .invalidJSON, .missingField:
            return 422
        case .payloadTooLarge:
            return 413
        case .invalidRequest, .unsupportedContent, .invalidContentLength, .invalidStructuredOutput:
            return 400
        case .invalidPort, .listenerFailed:
            return 500
        }
    }

    public var description: String {
        switch self {
        case .invalidPort(let port):
            return "invalid port: \(port)"
        case .invalidRequest:
            return "invalid HTTP request"
        case .invalidJSON:
            return "invalid JSON request body"
        case .missingField(let field):
            return "missing required field: \(field)"
        case .unsupportedContent:
            return "unsupported request content"
        case .invalidContentLength:
            return "invalid Content-Length"
        case .payloadTooLarge:
            return "HTTP payload too large"
        case .listenerFailed(let message):
            return "listener failed: \(message)"
        case .invalidStructuredOutput(let message):
            return message
        }
    }
}

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    private static let maxHeaderBytes = 64 * 1024
    private static let maxBodyBytes = 16 * 1024 * 1024

    static func read(from connection: NWConnection) async throws -> HTTPRequest {
        var buffer = Data()
        var headerEnd: Range<Data.Index>?
        while headerEnd == nil {
            let chunk = try await connection.receive()
            guard !chunk.isEmpty else { throw OpenAIServerError.invalidRequest }
            buffer.append(chunk)
            guard buffer.count <= maxHeaderBytes else {
                throw OpenAIServerError.payloadTooLarge
            }
            headerEnd = buffer.range(of: Data("\r\n\r\n".utf8))
        }

        let request = try parseComplete(buffer)
        var body = request.body
        let contentLength = try parsedContentLength(request.headers)
        while body.count < contentLength {
            let chunk = try await connection.receive()
            guard !chunk.isEmpty else { throw OpenAIServerError.invalidRequest }
            body.append(chunk)
        }
        if body.count > contentLength {
            body = body.prefix(contentLength)
        }

        return HTTPRequest(
            method: request.method,
            path: request.path,
            headers: request.headers,
            body: body
        )
    }

    static func parseComplete(_ buffer: Data) throws -> HTTPRequest {
        guard let headerRange = buffer.range(of: Data("\r\n\r\n".utf8)) else {
            throw OpenAIServerError.invalidRequest
        }
        let headerData = buffer[..<headerRange.lowerBound]
        guard let headerText = String(data: headerData, encoding: .utf8) else {
            throw OpenAIServerError.invalidRequest
        }
        let lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { throw OpenAIServerError.invalidRequest }
        let requestParts = requestLine.split(separator: " ", maxSplits: 2).map(String.init)
        guard requestParts.count >= 2 else { throw OpenAIServerError.invalidRequest }

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            let parts = line.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            headers[parts[0].lowercased()] = parts[1].trimmingCharacters(in: .whitespaces)
        }

        let contentLength = try parsedContentLength(headers)
        let bodyStart = headerRange.upperBound
        var body = Data(buffer[bodyStart...])
        if body.count > contentLength {
            body = body.prefix(contentLength)
        }

        return HTTPRequest(
            method: requestParts[0],
            path: requestParts[1].split(separator: "?", maxSplits: 1).first.map(String.init) ?? requestParts[1],
            headers: headers,
            body: body
        )
    }

    private static func parsedContentLength(_ headers: [String: String]) throws -> Int {
        guard let contentLength = Int(headers["content-length"] ?? "0"),
            contentLength >= 0
        else {
            throw OpenAIServerError.invalidContentLength
        }
        guard contentLength <= maxBodyBytes else {
            throw OpenAIServerError.payloadTooLarge
        }
        return contentLength
    }
}

public extension OpenAIChatRequest {
    static func parse(_ body: Data) throws -> OpenAIChatRequest {
        let rawObject: Any
        do {
            rawObject = try JSONSerialization.jsonObject(with: body)
        } catch {
            throw OpenAIServerError.invalidJSON
        }

        guard let object = rawObject as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        guard let model = object["model"] as? String, !model.isEmpty else {
            throw OpenAIServerError.missingField("model")
        }
        guard let rawMessages = object["messages"] as? [[String: Any]] else {
            throw OpenAIServerError.missingField("messages")
        }

        let messages = rawMessages.compactMap { raw -> OpenAIChatMessage? in
            guard let role = raw["role"] as? String else { return nil }
            let reasoningContent = raw["reasoning_content"] as? String
            if let content = raw["content"] as? String {
                return OpenAIChatMessage(role: role, content: content, reasoningContent: reasoningContent)
            }
            if let parts = raw["content"] as? [[String: Any]] {
                let text = parts.compactMap { part in
                    part["type"] as? String == "text" ? part["text"] as? String : nil
                }.joined()
                let imageReferences = parts.compactMap(Self.imageReference)
                return OpenAIChatMessage(
                    role: role,
                    content: text,
                    reasoningContent: reasoningContent,
                    imageReferences: imageReferences,
                    audioReferences: parts.compactMap { Self.mediaReference(from: $0, kind: .audio) },
                    videoReferences: parts.compactMap { Self.mediaReference(from: $0, kind: .video) }
                )
            }
            return nil
        }
        guard !messages.isEmpty else { throw OpenAIServerError.missingField("messages") }
        let streamOptions = object["stream_options"] as? [String: Any]
        let chatTemplateKwargs = try parseChatTemplateKwargs(object["chat_template_kwargs"])
        let structuredOutput = try StructuredOutputParser.parse(from: object)
        let tools = try parseTools(object["tools"])
        let toolChoice = try OpenAIToolChoice.parse(object["tool_choice"])

        return OpenAIChatRequest(
            model: model,
            messages: messages,
            maxTokens: intValue(object["max_tokens"] ?? object["max_completion_tokens"]) ?? 16,
            temperature: floatValue(object["temperature"]) ?? 0,
            topP: floatValue(object["top_p"]) ?? 0,
            topK: intValue(object["top_k"]) ?? 0,
            repetitionPenalty: floatValue(object["repetition_penalty"]) ?? 1,
            minP: floatValue(object["min_p"]) ?? 0,
            xtcProbability: floatValue(object["xtc_probability"]) ?? 0,
            xtcThreshold: floatValue(object["xtc_threshold"]) ?? 0.1,
            presencePenalty: floatValue(object["presence_penalty"]) ?? 0,
            frequencyPenalty: floatValue(object["frequency_penalty"]) ?? 0,
            minTokens: intValue(object["min_tokens"]) ?? 0,
            logitBias: try openAILogitBias(object["logit_bias"]),
            stop: try stopValues(object["stop"]),
            seed: intValue(object["seed"]),
            stream: object["stream"] as? Bool ?? false,
            includeUsage: streamOptions?["include_usage"] as? Bool ?? false,
            enableThinking: object["enable_thinking"] as? Bool,
            thinkingBudget: intValue(object["thinking_budget"]),
            chatTemplateKwargs: chatTemplateKwargs,
            structuredOutput: structuredOutput,
            tools: tools,
            toolChoice: toolChoice,
            cacheSession: stringValue(object["cache_session"])
        )
    }

    private static func imageReference(from part: [String: Any]) -> OpenAIChatImageReference? {
        guard part["type"] as? String == "image_url",
            let imageURL = part["image_url"] as? [String: Any],
            let url = imageURL["url"] as? String,
            !url.isEmpty
        else {
            return nil
        }

        if url.lowercased().hasPrefix("data:") {
            return OpenAIChatImageReference(source: .dataURI(url))
        }
        return OpenAIChatImageReference(source: .url(url))
    }

    fileprivate enum MediaKind {
        case audio
        case video

        /// The content-part spellings we accept. `audio_url`/`video_url` mirror
        /// `image_url` and are what Local AI Cat sends; `input_audio` is OpenAI's own
        /// spelling for inline audio and is accepted so ordinary OpenAI clients work.
        var partTypes: [String] {
            switch self {
            case .audio: return ["audio_url", "input_audio"]
            case .video: return ["video_url"]
            }
        }

        var urlKey: String {
            switch self {
            case .audio: return "audio_url"
            case .video: return "video_url"
            }
        }
    }

    private static func mediaReference(
        from part: [String: Any],
        kind: MediaKind
    ) -> OpenAIChatMediaReference? {
        guard let type = part["type"] as? String, kind.partTypes.contains(type) else {
            return nil
        }

        // OpenAI's `input_audio` carries base64 + a format rather than a URL, so it is
        // reassembled into the data URI the rest of this path already understands.
        if type == "input_audio" {
            guard let audio = part["input_audio"] as? [String: Any],
                let data = audio["data"] as? String, !data.isEmpty
            else {
                return nil
            }
            let format = (audio["format"] as? String) ?? "wav"
            return OpenAIChatMediaReference(source: .dataURI("data:audio/\(format);base64,\(data)"))
        }

        guard let container = part[kind.urlKey] as? [String: Any],
            let url = container["url"] as? String,
            !url.isEmpty
        else {
            return nil
        }

        if url.lowercased().hasPrefix("data:") {
            return OpenAIChatMediaReference(source: .dataURI(url))
        }
        return OpenAIChatMediaReference(source: .url(url))
    }

    private static func floatValue(_ value: Any?) -> Float? {
        switch value {
        case let value as Double:
            return Float(value)
        case let value as Float:
            return value
        case let value as Int:
            return Float(value)
        case let value as NSNumber:
            return value.floatValue
        default:
            return nil
        }
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as NSNumber:
            return value.intValue
        default:
            return nil
        }
    }

    private static func stringValue(_ value: Any?) -> String? {
        guard let value = value as? String, !value.isEmpty else { return nil }
        return value
    }

    private static func stopValues(_ value: Any?) throws -> [String] {
        guard let value else { return [] }
        if let string = value as? String {
            return [string]
        }
        if let strings = value as? [String] {
            return strings
        }
        throw OpenAIServerError.invalidJSON
    }

    private static func parseChatTemplateKwargs(_ value: Any?) throws -> [String: OpenAIJSONValue]? {
        guard let value else { return nil }
        guard let object = value as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        var converted: [String: OpenAIJSONValue] = [:]
        for (key, value) in object {
            guard let jsonValue = OpenAIJSONValue(value) else {
                throw OpenAIServerError.invalidJSON
            }
            converted[key] = jsonValue
        }
        return converted
    }

    private static func parseTools(_ value: Any?) throws -> [OpenAIJSONValue]? {
        guard let value else { return nil }
        guard let array = value as? [Any] else {
            throw OpenAIServerError.invalidJSON
        }
        var converted: [OpenAIJSONValue] = []
        for item in array {
            guard let jsonValue = OpenAIJSONValue(item) else {
                throw OpenAIServerError.invalidJSON
            }
            converted.append(jsonValue)
        }
        return converted
    }
}

private extension NWConnection {
    func receive() async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { data, _, isComplete, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                if let data {
                    continuation.resume(returning: data)
                    return
                }
                if isComplete {
                    continuation.resume(returning: Data())
                    return
                }
                continuation.resume(throwing: OpenAIServerError.invalidRequest)
            }
        }
    }
}

extension NWConnection {
    func send(data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            send(content: data, completion: .contentProcessed { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            })
        }
    }

    func sendFinal(data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            send(content: data, isComplete: true, completion: .contentProcessed { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            })
        }
    }
}
