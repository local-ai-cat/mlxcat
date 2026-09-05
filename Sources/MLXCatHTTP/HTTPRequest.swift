import Foundation
import Network

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
