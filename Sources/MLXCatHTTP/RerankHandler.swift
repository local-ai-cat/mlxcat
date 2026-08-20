import Foundation
import Network

struct RerankHandler {
    let backend: any OpenAIRerankBackend

    func handleRerank(_ request: HTTPRequest, connection: NWConnection) async throws {
        let rerankRequest: OpenAIRerankRequest
        do {
            rerankRequest = try OpenAIRerankRequest.parse(request.body)
        } catch {
            let status = (error as? OpenAIHTTPError)?.status
                ?? (error as? OpenAIServerError)?.httpStatus
                ?? 422
            try await sendJSON(
                openAIErrorBody(message: String(describing: error), status: status),
                status: status,
                connection: connection
            )
            return
        }

        if let errorResponse = rerankModelAvailabilityResponse(
            requestedModel: rerankRequest.model,
            rerankModels: backend.rerankModels
        ) {
            try await sendJSON(errorResponse.body, status: errorResponse.status, connection: connection)
            return
        }

        let result = try await backend.rerank(rerankRequest)
        try await sendJSON(
            buildRerankResponse(request: rerankRequest, result: result),
            status: 200,
            connection: connection
        )
    }

    private func sendJSON(_ object: [String: Any], status: Int, connection: NWConnection) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let header = "HTTP/1.1 \(status) \(rerankReasonPhrase(status))\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + body)
    }
}

func rerankModelAvailabilityResponse(
    requestedModel: String,
    rerankModels: [OpenAIModelInfo]
) -> HTTPJSONResponse? {
    guard !rerankModels.isEmpty else {
        return HTTPJSONResponse(
            status: 404,
            body: openAIErrorBody(message: "rerank backend unavailable", status: 404)
        )
    }

    guard !rerankModels.contains(where: { $0.id == requestedModel }) else {
        return nil
    }

    return HTTPJSONResponse(
        status: 404,
        body: openAIErrorBody(
            message: "Rerank model '\(requestedModel)' not found.",
            status: 404
        )
    )
}

public func buildRerankResponse(
    request: OpenAIRerankRequest,
    result: OpenAIRerankResult
) -> [String: Any] {
    [
        "id": "rerank-\(String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(8)).lowercased())",
        "results": result.indices.map { index in
            buildRerankResult(index: index, request: request, result: result)
        },
        "model": request.model,
        "usage": [
            "total_tokens": result.totalTokens
        ],
    ]
}

private func buildRerankResult(
    index: Int,
    request: OpenAIRerankRequest,
    result: OpenAIRerankResult
) -> [String: Any] {
    var payload: [String: Any] = [
        "index": index,
        "relevance_score": result.scores[index],
    ]
    if request.returnDocuments {
        payload["document"] = request.documents[index].documentPayload
    } else {
        payload["document"] = NSNull()
    }
    return payload
}

private func rerankReasonPhrase(_ status: Int) -> String {
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
