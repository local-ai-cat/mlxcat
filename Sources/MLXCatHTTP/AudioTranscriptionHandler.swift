import Foundation
import Network

struct AudioTranscriptionHandler {
    let backend: any AudioTranscriptionBackend

    func handleTranscription(_ request: HTTPRequest, connection: NWConnection) async throws {
        let transcriptionRequest: AudioTranscriptionRequest
        do {
            transcriptionRequest = try AudioTranscriptionRequest.parse(
                contentType: request.headers["content-type"] ?? "",
                body: request.body
            )
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 422
            try await sendJSON(openAIErrorBody(message: String(describing: error), status: status), status: status, connection: connection)
            return
        }

        let result = try await backend.transcribe(transcriptionRequest)
        try await sendJSON(
            buildAudioTranscriptionResponse(result),
            status: 200,
            connection: connection
        )
    }

    private func sendJSON(_ object: [String: Any], status: Int, connection: NWConnection) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let header = "HTTP/1.1 \(status) \(audioTranscriptionReasonPhrase(status))\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + body)
    }
}

private func audioTranscriptionReasonPhrase(_ status: Int) -> String {
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
    case 501:
        return "Not Implemented"
    default:
        return "OK"
    }
}
