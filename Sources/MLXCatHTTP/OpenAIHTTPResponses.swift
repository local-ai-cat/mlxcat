import Foundation
import Network

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
                    "finish_reason": NSNull()
                ]
            ]
        ]
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
                "finish_reason": NSNull()
            ]
        ]
    ]
}

func modelStatusBody(_ status: OpenAIModelPoolStatus) -> [String: Any] {
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
                "footprint_source": model.footprintSource.map { $0 as Any } ?? NSNull(),
                "accounted_bytes": model.accountedBytes.map { $0 as Any } ?? NSNull()
            ] as [String: Any]
        }
    ]
}

func lifecycleResultBody(_ result: OpenAIModelLifecycleResult) -> [String: Any] {
    var body: [String: Any] = [
        "status": result.status,
        "model_id": result.modelID
    ]
    if let message = result.message {
        body["message"] = message
    }
    return body
}

func errorResponse(for error: Error) -> HTTPJSONResponse {
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

func httpReasonPhrase(_ status: Int) -> String {
    let phrases = [
        200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
        409: "Conflict", 413: "Payload Too Large", 422: "Unprocessable Entity",
        429: "Too Many Requests", 503: "Service Unavailable", 507: "Insufficient Storage",
        500: "Internal Server Error", 501: "Not Implemented"
    ]
    return phrases[status] ?? "OK"
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
