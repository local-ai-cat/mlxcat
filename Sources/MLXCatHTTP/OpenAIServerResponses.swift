import Foundation
import Network

extension OpenAIServer {
    func toolsRequested(_ request: OpenAIChatRequest) -> Bool {
        selectOpenAITools(tools: request.tools, toolChoice: request.toolChoice) != nil
    }

    func sendSSE(_ object: [String: Any], connection: NWConnection) async throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        guard let json = String(data: data, encoding: .utf8) else { throw OpenAIServerError.invalidJSON }
        let line = "data: \(json)\n\n"
        try await connection.send(data: Data(line.utf8))
    }

    func sendParserDelta(
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

    private func jsonEscape(_ text: String) -> String {
        guard text
            .contains(where: {
            $0 == "\\" || $0 == "\"" || $0 == "\n" || $0 == "\r" || $0 == "\t" || $0.asciiValue
            .map({ $0 < 0x20 }) == true }) else {
            return text
        }
        var result = ""
        result.reserveCapacity(text.count + 8)
        for character in text {
            switch character {
            case "\\": result += "\\\\"
            case "\"": result += "\\\""
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            default:
                if let ascii = character.asciiValue, ascii < 0x20 {
                    result += String(format: "\\u%04x", ascii)
                } else {
                    result.append(character)
                }
            }
        }
        return result
    }

    func sendJSON(
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

    func sendError(_ error: Error, connection: NWConnection) async throws {
        let response = errorResponse(for: error)
        try await sendJSON(
            response.body,
            status: response.status,
            headers: response.headers,
            connection: connection
        )
    }

}
