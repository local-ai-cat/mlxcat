import Foundation

public enum MultipartFormDataError: Error, Equatable, CustomStringConvertible {
    case missingBoundary
    case malformedBody
    case malformedHeader

    public var description: String {
        switch self {
        case .missingBoundary:
            return "missing multipart boundary"
        case .malformedBody:
            return "malformed multipart body"
        case .malformedHeader:
            return "malformed multipart header"
        }
    }
}

public struct MultipartFormData: Sendable {
    public struct Part: Sendable, Equatable {
        public let name: String
        public let filename: String?
        public let contentType: String?
        public let data: Data
    }

    public let parts: [Part]

    public init(contentType: String, body: Data) throws {
        let boundary = try Self.boundary(from: contentType)
        self.parts = try Self.parseParts(boundary: boundary, body: body)
    }

    public func stringValue(forField name: String) -> String? {
        guard
            let part = parts.first(where: { $0.name == name && $0.filename == nil }),
            let value = String(data: part.data, encoding: .utf8)
        else {
            return nil
        }
        return value.trimmingTrailingCRLF()
    }

    public func filePart(named name: String) -> (filename: String, contentType: String?, data: Data)? {
        guard let part = parts.first(where: { $0.name == name }), let filename = part.filename else {
            return nil
        }
        return (filename, part.contentType, part.data)
    }

    private static func boundary(from contentType: String) throws -> String {
        let parameters = splitParameters(contentType)
        guard
            let first = parameters.first,
            first.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "multipart/form-data"
        else {
            throw MultipartFormDataError.missingBoundary
        }

        for parameter in parameters.dropFirst() {
            let pair = parameter.split(separator: "=", maxSplits: 1).map(String.init)
            guard pair.count == 2 else { continue }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard key == "boundary" else { continue }
            let value = unquoted(pair[1].trimmingCharacters(in: .whitespacesAndNewlines))
            guard !value.isEmpty else { throw MultipartFormDataError.missingBoundary }
            return value
        }

        throw MultipartFormDataError.missingBoundary
    }

    private static func parseParts(boundary: String, body: Data) throws -> [Part] {
        let bodyBytes = [UInt8](body)
        let delimiter = [UInt8]("--\(boundary)".utf8)
        let prefixedDelimiter = [UInt8]("\r\n--\(boundary)".utf8)
        let crlf = [UInt8]("\r\n".utf8)
        let headerTerminator = [UInt8]("\r\n\r\n".utf8)
        let closeMarker = [UInt8]("--".utf8)

        guard starts(with: delimiter, in: bodyBytes, at: 0) else {
            throw MultipartFormDataError.malformedBody
        }

        var cursor = delimiter.count
        if starts(with: closeMarker, in: bodyBytes, at: cursor) {
            return []
        }
        guard starts(with: crlf, in: bodyBytes, at: cursor) else {
            throw MultipartFormDataError.malformedBody
        }
        cursor += crlf.count

        var parts: [Part] = []
        while cursor < bodyBytes.count {
            guard let headerEnd = firstRange(of: headerTerminator, in: bodyBytes, startingAt: cursor) else {
                throw MultipartFormDataError.malformedHeader
            }
            let headerBytes = bodyBytes[cursor..<headerEnd]
            let headers = try parseHeaders(Data(headerBytes))
            cursor = headerEnd + headerTerminator.count

            guard let nextDelimiter = firstRange(of: prefixedDelimiter, in: bodyBytes, startingAt: cursor) else {
                throw MultipartFormDataError.malformedBody
            }
            let value = Data(bodyBytes[cursor..<nextDelimiter])
            cursor = nextDelimiter + prefixedDelimiter.count

            guard let disposition = headers["content-disposition"] else {
                throw MultipartFormDataError.malformedHeader
            }
            let parsedDisposition = try parseContentDisposition(disposition)
            parts.append(
                Part(
                    name: parsedDisposition.name,
                    filename: parsedDisposition.filename,
                    contentType: headers["content-type"],
                    data: value
                )
            )

            if starts(with: closeMarker, in: bodyBytes, at: cursor) {
                return parts
            }
            guard starts(with: crlf, in: bodyBytes, at: cursor) else {
                throw MultipartFormDataError.malformedBody
            }
            cursor += crlf.count
        }

        throw MultipartFormDataError.malformedBody
    }

    private static func parseHeaders(_ data: Data) throws -> [String: String] {
        guard let headerText = String(data: data, encoding: .utf8) else {
            throw MultipartFormDataError.malformedHeader
        }

        var headers: [String: String] = [:]
        for line in headerText.components(separatedBy: "\r\n") where !line.isEmpty {
            let pair = line.split(separator: ":", maxSplits: 1).map(String.init)
            guard pair.count == 2 else {
                throw MultipartFormDataError.malformedHeader
            }
            let name = pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            headers[name] = value
        }
        return headers
    }

    private static func parseContentDisposition(_ value: String) throws -> (name: String, filename: String?) {
        let parameters = splitParameters(value)
        guard parameters.first?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "form-data" else {
            throw MultipartFormDataError.malformedHeader
        }

        var name: String?
        var filename: String?
        for parameter in parameters.dropFirst() {
            let pair = parameter.split(separator: "=", maxSplits: 1).map(String.init)
            guard pair.count == 2 else { continue }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = unquoted(pair[1].trimmingCharacters(in: .whitespacesAndNewlines))
            if key == "name" {
                name = value
            } else if key == "filename" {
                filename = value
            }
        }

        guard let name, !name.isEmpty else {
            throw MultipartFormDataError.malformedHeader
        }
        return (name, filename)
    }

    private static func splitParameters(_ value: String) -> [String] {
        var parameters: [String] = []
        var current = ""
        var isQuoted = false
        var isEscaped = false

        for character in value {
            if isEscaped {
                current.append(character)
                isEscaped = false
                continue
            }
            if character == "\\" {
                current.append(character)
                isEscaped = true
                continue
            }
            if character == "\"" {
                isQuoted.toggle()
                current.append(character)
                continue
            }
            if character == ";", !isQuoted {
                parameters.append(current)
                current = ""
                continue
            }
            current.append(character)
        }
        parameters.append(current)
        return parameters
    }

    private static func unquoted(_ value: String) -> String {
        guard value.count >= 2, value.first == "\"", value.last == "\"" else {
            return value
        }

        var result = ""
        var isEscaped = false
        for character in value.dropFirst().dropLast() {
            if isEscaped {
                result.append(character)
                isEscaped = false
            } else if character == "\\" {
                isEscaped = true
            } else {
                result.append(character)
            }
        }
        return result
    }

    private static func firstRange(of pattern: [UInt8], in bytes: [UInt8], startingAt start: Int) -> Int? {
        guard !pattern.isEmpty, start <= bytes.count - pattern.count else {
            return nil
        }

        var index = start
        while index <= bytes.count - pattern.count {
            if starts(with: pattern, in: bytes, at: index) {
                return index
            }
            index += 1
        }
        return nil
    }

    private static func starts(with pattern: [UInt8], in bytes: [UInt8], at index: Int) -> Bool {
        guard index >= 0, index + pattern.count <= bytes.count else {
            return false
        }

        for offset in 0..<pattern.count where bytes[index + offset] != pattern[offset] {
            return false
        }
        return true
    }
}

private extension String {
    func trimmingTrailingCRLF() -> String {
        var value = self
        while value.hasSuffix("\r\n") {
            value.removeLast(2)
        }
        return value
    }
}
