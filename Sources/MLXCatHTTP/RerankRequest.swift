import Foundation

public enum OpenAIRerankText: Sendable, Equatable {
    case string(String)
    case object([String: OpenAIJSONValue])

    public var text: String {
        switch self {
        case .string(let value):
            return value
        case .object(let value):
            guard case .string(let text)? = value["text"] else {
                return ""
            }
            return text
        }
    }

    public var documentPayload: [String: Any] {
        switch self {
        case .string(let value):
            return ["text": value]
        case .object(let value):
            return value.mapValues { jsonObject(from: $0) }
        }
    }
}

public struct OpenAIRerankRequest: Sendable, Equatable {
    public let model: String
    public let query: OpenAIRerankText
    public let documents: [OpenAIRerankText]
    public let topN: Int?
    public let returnDocuments: Bool
    public let maxChunksPerDoc: Int?

    public init(
        model: String,
        query: OpenAIRerankText,
        documents: [OpenAIRerankText],
        topN: Int? = nil,
        returnDocuments: Bool = true,
        maxChunksPerDoc: Int? = nil
    ) {
        self.model = model
        self.query = query
        self.documents = documents
        self.topN = topN
        self.returnDocuments = returnDocuments
        self.maxChunksPerDoc = maxChunksPerDoc
    }

    public static func parse(_ body: Data) throws -> OpenAIRerankRequest {
        guard
            let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
            let model = object["model"] as? String,
            !model.isEmpty
        else {
            throw OpenAIServerError.invalidJSON
        }

        let query = try parseText(object["query"])
        let documents = try parseDocuments(object["documents"])
        guard !query.text.isEmpty else {
            throw OpenAIHTTPError(status: 400, message: "Query cannot be empty")
        }
        guard !documents.isEmpty else {
            throw OpenAIHTTPError(status: 400, message: "Documents cannot be empty")
        }

        let topN = openAIIntValue(object["top_n"])
        let returnDocuments = object["return_documents"] as? Bool ?? true
        let maxChunksPerDoc = openAIIntValue(object["max_chunks_per_doc"])
        return OpenAIRerankRequest(
            model: model,
            query: query,
            documents: documents,
            topN: topN,
            returnDocuments: returnDocuments,
            maxChunksPerDoc: maxChunksPerDoc
        )
    }

    private static func parseDocuments(_ value: Any?) throws -> [OpenAIRerankText] {
        if let strings = value as? [String] {
            return strings.map(OpenAIRerankText.string)
        }
        if let values = value as? [Any] {
            return try values.map(parseText)
        }
        throw OpenAIServerError.invalidJSON
    }

    private static func parseText(_ value: Any?) throws -> OpenAIRerankText {
        if let string = value as? String {
            return .string(string)
        }
        if let object = value as? [String: Any] {
            var converted: [String: OpenAIJSONValue] = [:]
            for (key, value) in object {
                guard let jsonValue = OpenAIJSONValue(value) else {
                    throw OpenAIServerError.invalidJSON
                }
                converted[key] = jsonValue
            }
            return .object(converted)
        }
        throw OpenAIServerError.invalidJSON
    }
}

public struct OpenAIRerankResult: Sendable, Equatable {
    public let scores: [Float]
    public let indices: [Int]
    public let totalTokens: Int

    public init(scores: [Float], indices: [Int], totalTokens: Int) {
        self.scores = scores
        self.indices = indices
        self.totalTokens = totalTokens
    }
}

public protocol OpenAIRerankBackend: Sendable {
    var rerankModels: [OpenAIModelInfo] { get }
    func rerank(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult
}

public func applyRerankTopN(_ indices: [Int], topN: Int?) -> [Int] {
    guard let topN, topN < indices.count else {
        return indices
    }
    let count = topN >= 0 ? topN : max(indices.count + topN, 0)
    return Array(indices.prefix(count))
}
