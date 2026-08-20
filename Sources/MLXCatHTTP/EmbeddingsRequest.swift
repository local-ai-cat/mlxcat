import Foundation

public enum OpenAIEmbeddingInput: Sendable, Equatable {
    case string(String)
    case strings([String])

    public var values: [String] {
        switch self {
        case .string(let value):
            return [value]
        case .strings(let values):
            return values
        }
    }
}

public enum OpenAIEmbeddingEncodingFormat: String, Sendable {
    case float
    case base64
}

public struct OpenAIEmbeddingsRequest: Sendable, Equatable {
    public let input: OpenAIEmbeddingInput
    public let model: String
    public let encodingFormat: OpenAIEmbeddingEncodingFormat
    public let dimensions: Int?

    public init(
        input: OpenAIEmbeddingInput,
        model: String,
        encodingFormat: OpenAIEmbeddingEncodingFormat = .float,
        dimensions: Int? = nil
    ) {
        self.input = input
        self.model = model
        self.encodingFormat = encodingFormat
        self.dimensions = dimensions
    }

    public static func parse(_ body: Data) throws -> OpenAIEmbeddingsRequest {
        guard
            let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
            let model = object["model"] as? String
        else {
            throw OpenAIServerError.invalidJSON
        }

        let input: OpenAIEmbeddingInput
        if let string = object["input"] as? String {
            input = .string(string)
        } else if let strings = object["input"] as? [String], !strings.isEmpty {
            input = .strings(strings)
        } else {
            throw OpenAIServerError.invalidJSON
        }

        let encodingFormat: OpenAIEmbeddingEncodingFormat
        if let rawEncodingFormat = object["encoding_format"] as? String {
            guard let parsed = OpenAIEmbeddingEncodingFormat(rawValue: rawEncodingFormat) else {
                throw OpenAIServerError.invalidJSON
            }
            encodingFormat = parsed
        } else {
            encodingFormat = .float
        }

        let dimensions = openAIIntValue(object["dimensions"])
        if let dimensions, dimensions <= 0 {
            throw OpenAIServerError.invalidJSON
        }

        return OpenAIEmbeddingsRequest(
            input: input,
            model: model,
            encodingFormat: encodingFormat,
            dimensions: dimensions
        )
    }
}

public struct OpenAIEmbeddingsResult: Sendable, Equatable {
    public let embeddings: [[Float]]
    public let promptTokens: Int

    public init(embeddings: [[Float]], promptTokens: Int) {
        self.embeddings = embeddings
        self.promptTokens = promptTokens
    }
}

public protocol OpenAIEmbeddingsBackend: Sendable {
    var embeddingModels: [OpenAIModelInfo] { get }
    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> OpenAIEmbeddingsResult
}
