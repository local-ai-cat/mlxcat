import Foundation

public enum OpenAICompletionPrompt: Sendable, Equatable {
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

public struct OpenAICompletionRequest: Sendable, Equatable {
    public let model: String
    public let prompt: OpenAICompletionPrompt
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let minP: Float
    public let repetitionPenalty: Float
    public let presencePenalty: Float
    public let frequencyPenalty: Float
    public let minTokens: Int
    public let logitBias: [Int: Float]
    public let stop: [String]
    public let seed: Int?
    public let stream: Bool
    public let includeUsage: Bool
    public let structuredOutput: StructuredOutputSpec

    public init(
        model: String,
        prompt: OpenAICompletionPrompt,
        maxTokens: Int,
        temperature: Float = 0,
        topP: Float = 0,
        topK: Int = 0,
        minP: Float = 0,
        repetitionPenalty: Float = 1,
        presencePenalty: Float = 0,
        frequencyPenalty: Float = 0,
        minTokens: Int = 0,
        logitBias: [Int: Float] = [:],
        stop: [String] = [],
        seed: Int? = nil,
        stream: Bool = false,
        includeUsage: Bool = false,
        structuredOutput: StructuredOutputSpec = .none
    ) {
        self.model = model
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.minTokens = max(0, minTokens)
        self.logitBias = logitBias
        self.stop = stop
        self.seed = seed
        self.stream = stream
        self.includeUsage = includeUsage
        self.structuredOutput = structuredOutput
    }

    public func request(forPrompt prompt: String) -> OpenAICompletionRequest {
        OpenAICompletionRequest(
            model: model,
            prompt: .string(prompt),
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty,
            minTokens: minTokens,
            logitBias: logitBias,
            stop: stop,
            seed: seed,
            stream: stream,
            includeUsage: includeUsage,
            structuredOutput: structuredOutput
        )
    }

    public static func parse(_ body: Data) throws -> OpenAICompletionRequest {
        guard
            let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
            let model = object["model"] as? String
        else {
            throw OpenAIServerError.invalidJSON
        }

        let prompt: OpenAICompletionPrompt
        if let string = object["prompt"] as? String {
            prompt = .string(string)
        } else if let strings = object["prompt"] as? [String], !strings.isEmpty {
            prompt = .strings(strings)
        } else {
            throw OpenAIServerError.invalidJSON
        }

        let streamOptions = object["stream_options"] as? [String: Any]
        return OpenAICompletionRequest(
            model: model,
            prompt: prompt,
            maxTokens: openAIIntValue(object["max_tokens"]) ?? 16,
            temperature: openAIFloatValue(object["temperature"]) ?? 0,
            topP: openAIFloatValue(object["top_p"]) ?? 0,
            topK: openAIIntValue(object["top_k"]) ?? 0,
            minP: openAIFloatValue(object["min_p"]) ?? 0,
            repetitionPenalty: openAIFloatValue(object["repetition_penalty"]) ?? 1,
            presencePenalty: openAIFloatValue(object["presence_penalty"]) ?? 0,
            frequencyPenalty: openAIFloatValue(object["frequency_penalty"]) ?? 0,
            minTokens: openAIIntValue(object["min_tokens"]) ?? 0,
            logitBias: try openAILogitBias(object["logit_bias"]),
            stop: try openAIStringArray(object["stop"]),
            seed: openAIIntValue(object["seed"]),
            stream: object["stream"] as? Bool ?? false,
            includeUsage: streamOptions?["include_usage"] as? Bool ?? false,
            structuredOutput: try StructuredOutputParser.parse(from: object)
        )
    }
}

public protocol OpenAICompletionBackend: Sendable {
    func startCompletion(_ request: OpenAICompletionRequest) async throws -> OpenAIChatStream
}

func openAIFloatValue(_ value: Any?) -> Float? {
    switch value {
    case let value as Double:
        return Float(value)
    case let value as Float:
        return value
    case let value as Int:
        return Float(value)
    case let value as NSNumber:
        return value.floatValue
    default:
        return nil
    }
}

func openAIIntValue(_ value: Any?) -> Int? {
    switch value {
    case let value as Int:
        return value
    case let value as NSNumber:
        return value.intValue
    default:
        return nil
    }
}

func openAIStringArray(_ value: Any?) throws -> [String] {
    guard let value else { return [] }
    if let string = value as? String {
        return [string]
    }
    if let strings = value as? [String] {
        return strings
    }
    throw OpenAIServerError.invalidJSON
}

func openAILogitBias(_ value: Any?) throws -> [Int: Float] {
    guard let value else { return [:] }
    guard let rawBias = value as? [String: Any] else {
        throw OpenAIServerError.invalidJSON
    }

    var result: [Int: Float] = [:]
    for (tokenID, rawValue) in rawBias {
        guard let tokenID = Int(tokenID), let bias = openAIFloatValue(rawValue) else {
            throw OpenAIServerError.invalidJSON
        }
        result[tokenID] = bias
    }
    return result
}
