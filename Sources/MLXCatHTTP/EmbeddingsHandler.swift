import Foundation
import Network

struct EmbeddingsHandler {
    let backend: any OpenAIEmbeddingsBackend

    func handleEmbeddings(_ request: HTTPRequest, connection: NWConnection) async throws {
        let embeddingsRequest: OpenAIEmbeddingsRequest
        do {
            embeddingsRequest = try OpenAIEmbeddingsRequest.parse(request.body)
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 422
            try await sendJSON(openAIErrorBody(message: String(describing: error), status: status), status: status, connection: connection)
            return
        }

        if let errorResponse = embeddingModelAvailabilityResponse(
            requestedModel: embeddingsRequest.model,
            embeddingModels: backend.embeddingModels
        ) {
            try await sendJSON(errorResponse.body, status: errorResponse.status, connection: connection)
            return
        }

        let result = try await backend.embed(embeddingsRequest)
        try await sendJSON(
            buildEmbeddingsResponse(request: embeddingsRequest, result: result),
            status: 200,
            connection: connection
        )
    }

    private func sendJSON(_ object: [String: Any], status: Int, connection: NWConnection) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let header = "HTTP/1.1 \(status) \(embeddingsReasonPhrase(status))\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + body)
    }
}

func embeddingModelAvailabilityResponse(
    requestedModel: String,
    embeddingModels: [OpenAIModelInfo]
) -> HTTPJSONResponse? {
    guard !embeddingModels.isEmpty else {
        return HTTPJSONResponse(
            status: 404,
            body: openAIErrorBody(message: "embeddings backend unavailable", status: 404)
        )
    }

    guard !embeddingModels.contains(where: { $0.id == requestedModel }) else {
        return nil
    }

    return HTTPJSONResponse(
        status: 400,
        body: openAIErrorBody(
            message: "Model '\(requestedModel)' is not an embedding model. Use /v1/chat/completions for LLM models.",
            status: 400
        )
    )
}

public func buildEmbeddingsResponse(
    request: OpenAIEmbeddingsRequest,
    result: OpenAIEmbeddingsResult
) -> [String: Any] {
    [
        "object": "list",
        "data": result.embeddings.enumerated().map { index, embedding in
            buildEmbeddingData(
                index: index,
                embedding: embedding,
                encodingFormat: request.encodingFormat,
                dimensions: request.dimensions
            )
        },
        "model": request.model,
        "usage": [
            "prompt_tokens": result.promptTokens,
            "total_tokens": result.promptTokens,
        ],
    ]
}

public func buildEmbeddingData(
    index: Int,
    embedding: [Float],
    encodingFormat: OpenAIEmbeddingEncodingFormat,
    dimensions: Int?
) -> [String: Any] {
    let prepared = truncateAndRenormalizeEmbedding(embedding, dimensions: dimensions)
    let encoded: Any
    switch encodingFormat {
    case .float:
        encoded = prepared
    case .base64:
        encoded = encodeEmbeddingBase64(prepared)
    }
    return [
        "object": "embedding",
        "index": index,
        "embedding": encoded,
    ]
}

public func truncateAndRenormalizeEmbedding(_ embedding: [Float], dimensions: Int?) -> [Float] {
    guard let dimensions, dimensions < embedding.count else {
        return embedding
    }

    let truncated = Array(embedding.prefix(dimensions))
    let norm = sqrt(truncated.reduce(Float(0)) { partial, value in
        partial + value * value
    })
    guard norm > 0 else { return truncated }
    return truncated.map { $0 / norm }
}

public func encodeEmbeddingBase64(_ embedding: [Float]) -> String {
    var data = Data()
    data.reserveCapacity(embedding.count * MemoryLayout<Float>.size)
    for value in embedding {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { bytes in
            data.append(contentsOf: bytes)
        }
    }
    return data.base64EncodedString()
}

private func embeddingsReasonPhrase(_ status: Int) -> String {
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
