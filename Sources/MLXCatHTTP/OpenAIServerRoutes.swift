import Foundation
import Network

extension OpenAIServer {
    func handle(_ connection: NWConnection) async {
        do {
            let request = try await HTTPRequest.read(from: connection)
            if try await routeCatalog(request, connection: connection) { return }
            if try await routeInference(request, connection: connection) { return }
            try await routeDynamic(request, connection: connection)
        } catch {
            try? await sendError(error, connection: connection)
        }
    }

    private func routeCatalog(_ request: HTTPRequest, connection: NWConnection) async throws -> Bool {
        switch (request.method, request.path) {
        case ("GET", "/health"):
            try await sendJSON(healthResponse(), status: 200, connection: connection)
            return true
        case ("GET", "/v1/models"):
            try await sendJSON(modelsResponse(), status: 200, connection: connection)
            return true
        case ("GET", "/v1/models/status"):
            try await sendLifecycleResponse(request, connection: connection)
            return true
        case ("GET", "/v1/mcp/tools"):
            try await MCPHandler(manager: mcpManager).handleTools(connection: connection)
            return true
        case ("GET", "/v1/mcp/servers"):
            try await MCPHandler(manager: mcpManager).handleServers(connection: connection)
            return true
        case ("POST", "/v1/mcp/execute"):
            try await MCPHandler(manager: mcpManager).handleExecute(request, connection: connection)
            return true
        default: return false
        }
    }

    private func routeInference(_ request: HTTPRequest, connection: NWConnection) async throws -> Bool {
        switch (request.method, request.path) {
        case ("POST", "/v1/chat/completions"):
            try await handleChatCompletion(request, connection: connection)
            return true
        case ("POST", "/v1/messages"):
            try await AnthropicMessagesHandler(backend: backend, mcpManager: mcpManager).handleMessages(
                request,
                connection: connection
            )
            return true
        case ("POST", "/v1/messages/count_tokens"):
            try await AnthropicMessagesHandler(backend: backend).handleCountTokens(request, connection: connection)
            return true
        case ("POST", "/v1/responses"):
            try await ResponsesHandler(backend: backend, store: responsesStore, mcpManager: mcpManager).handleCreate(
                request,
                connection: connection
            )
            return true
        default: return try await routeAuxiliary(request, connection: connection)
        }
    }

    private func routeAuxiliary(_ request: HTTPRequest, connection: NWConnection) async throws -> Bool {
        switch (request.method, request.path) {
        case ("POST", "/v1/completions"):
            guard let completionBackend = backend as? any OpenAICompletionBackend else {
                try await sendJSON(
                    openAIErrorBody(message: "completions backend unavailable", status: 404),
                    status: 404,
                    connection: connection
                )
                return true
            }
            try await CompletionsHandler(backend: completionBackend).handleCompletions(request, connection: connection)
            return true
        case ("POST", "/v1/embeddings"):
            guard let embeddingsBackend = backend as? any OpenAIEmbeddingsBackend else {
                try await sendJSON(
                    openAIErrorBody(message: "embeddings backend unavailable", status: 404),
                    status: 404,
                    connection: connection
                )
                return true
            }
            try await EmbeddingsHandler(backend: embeddingsBackend).handleEmbeddings(request, connection: connection)
            return true
        case ("POST", "/v1/rerank"):
            guard let rerankBackend = backend as? any OpenAIRerankBackend else {
                try await sendJSON(
                    openAIErrorBody(message: "rerank backend unavailable", status: 404),
                    status: 404,
                    connection: connection
                )
                return true
            }
            try await RerankHandler(backend: rerankBackend).handleRerank(request, connection: connection)
            return true
        case ("POST", "/v1/audio/transcriptions"):
            guard let audioBackend = backend as? any AudioTranscriptionBackend,
                  !audioBackend.transcriptionModels.isEmpty else {
                try await sendJSON(openAIErrorBody(message: "transcription backend not configured", status: 501),
                                   status: 501, connection: connection)
                return true
            }
            try await AudioTranscriptionHandler(backend: audioBackend).handleTranscription(
                request,
                connection: connection
            )
            return true
        default: return false
        }
    }

    private func routeDynamic(_ request: HTTPRequest, connection: NWConnection) async throws {
        if request.method == "GET", let id = responseID(from: request.path) {
            try await ResponsesHandler(backend: backend, store: responsesStore).handleGet(
                id: id,
                connection: connection
            )
            return
        }
        if request.method == "DELETE", let id = responseID(from: request.path) {
            try await ResponsesHandler(backend: backend, store: responsesStore).handleDelete(
                id: id,
                connection: connection
            )
            return
        }
        if isModelLifecyclePath(request.path) {
            try await sendLifecycleResponse(request, connection: connection)
            return
        }
        try await sendJSON(openAIErrorBody(message: "not found", status: 404), status: 404, connection: connection)
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

        if request.method == "POST", let modelID = Self.lifecycleModelID(from: request.path, action: "load") {
            return HTTPJSONResponse(
                status: 200,
                body: lifecycleResultBody(try await lifecycleBackend.loadModel(modelID))
            )
        }

        if request.method == "POST", let modelID = Self.lifecycleModelID(from: request.path, action: "unload") {
            return HTTPJSONResponse(
                status: 200,
                body: lifecycleResultBody(try await lifecycleBackend.unloadModel(modelID))
            )
        }

        return nil
    }

    private func isModelLifecyclePath(_ path: String) -> Bool {
        Self.lifecycleModelID(from: path, action: "load") != nil
            || Self.lifecycleModelID(from: path, action: "unload") != nil
    }

    static func lifecycleModelID(from path: String, action: String) -> String? {
        let prefix = "/v1/models/"
        let suffix = "/\(action)"
        guard path.hasPrefix(prefix), path.hasSuffix(suffix) else { return nil }
        let idStart = path.index(path.startIndex, offsetBy: prefix.count)
        let idEnd = path.index(path.endIndex, offsetBy: -suffix.count)
        guard idStart < idEnd else { return nil }
        return String(path[idStart..<idEnd]).removingPercentEncoding
    }

    func modelsResponse() -> [String: Any] {
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
                    "model_type": model.modelType
                ]
                if let maxModelLength = model.maxModelLength {
                    payload["max_model_len"] = maxModelLength
                }
                return payload
            }
        ]
    }

    func healthResponse() -> [String: Any] {
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

}
