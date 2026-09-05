import Foundation
import Network

public struct OpenAIModelInfo: Sendable {
    public let id: String
    public let maxModelLength: Int?
    public let modelType: String

    public init(id: String, maxModelLength: Int? = nil, modelType: String = "llm") {
        self.id = id
        self.maxModelLength = maxModelLength
        self.modelType = modelType
    }
}

public struct OpenAIChatImageReference: Sendable, Equatable {
    public enum Source: Sendable, Equatable {
        case dataURI(String)
        case url(String)
    }

    public let source: Source

    public init(source: Source) {
        self.source = source
    }
}

/// An audio or video attachment the MODEL consumes, as opposed to
/// `/v1/audio/transcriptions`, which is speech-to-text and unrelated.
///
/// Same two sources as an image reference: a URL the engine can open, or an inline
/// base64 data URI. Kept as a separate type rather than reusing
/// `OpenAIChatImageReference` because "image" in the name of the thing carrying a video
/// is the sort of detail that later reads as a bug.
public struct OpenAIChatMediaReference: Sendable, Equatable {
    public enum Source: Sendable, Equatable {
        case dataURI(String)
        case url(String)
    }

    public let source: Source

    public init(source: Source) {
        self.source = source
    }
}

public struct OpenAIChatMessage: Sendable {
    public let role: String
    public let content: String
    public let reasoningContent: String?
    public let imageReferences: [OpenAIChatImageReference]
    /// Audio the model consumes. Audio/video-capable models (gemma-4) use them; models
    /// without those towers ignore them. There is deliberately NO text-description
    /// fallback at this layer — deciding to describe media into text is a caller's
    /// policy, not the engine's.
    public let audioReferences: [OpenAIChatMediaReference]
    public let videoReferences: [OpenAIChatMediaReference]

    public init(
        role: String,
        content: String,
        reasoningContent: String? = nil,
        imageReferences: [OpenAIChatImageReference] = [],
        audioReferences: [OpenAIChatMediaReference] = [],
        videoReferences: [OpenAIChatMediaReference] = []
    ) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.imageReferences = imageReferences
        self.audioReferences = audioReferences
        self.videoReferences = videoReferences
    }
}

public enum OpenAIJSONValue: Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: OpenAIJSONValue])
    case array([OpenAIJSONValue])
    case null

    init?(_ value: Any) {
        switch value {
        case let string as String:
            self = .string(string)
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                self = .bool(number.boolValue)
            } else {
                self = .number(number.doubleValue)
            }
        case let object as [String: Any]:
            guard let converted = Self.convertObject(object) else { return nil }
            self = .object(converted)
        case let array as [Any]:
            var converted: [OpenAIJSONValue] = []
            for value in array {
                guard let jsonValue = OpenAIJSONValue(value) else { return nil }
                converted.append(jsonValue)
            }
            self = .array(converted)
        case _ as NSNull:
            self = .null
        default:
            return nil
        }
    }
    private static func convertObject(_ object: [String: Any]) -> [String: OpenAIJSONValue]? {
        var converted: [String: OpenAIJSONValue] = [:]
        for (key, value) in object {
            guard let jsonValue = OpenAIJSONValue(value) else { return nil }
            converted[key] = jsonValue
        }
        return converted
    }

}

public struct OpenAIChatRequest: Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let repetitionPenalty: Float
    public let minP: Float
    public let xtcProbability: Float
    public let xtcThreshold: Float
    public let presencePenalty: Float
    public let frequencyPenalty: Float
    public let minTokens: Int
    public let logitBias: [Int: Float]
    public let stop: [String]
    public let seed: Int?
    public let stream: Bool
    public let includeUsage: Bool
    public let enableThinking: Bool?
    public let thinkingBudget: Int?
    public let chatTemplateKwargs: [String: OpenAIJSONValue]?
    public let structuredOutput: StructuredOutputSpec
    public let tools: [OpenAIJSONValue]?
    public let toolChoice: OpenAIToolChoice?
    /// Per-request override for the Qwen-XML tool grammar (`tool_grammar` in the
    /// JSON body). nil = server default; see `QwenXMLToolGrammarPolicy`.
    public let toolGrammar: Bool?
    public let cacheSession: String?

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        maxTokens: Int,
        temperature: Float = 0,
        topP: Float = 0,
        topK: Int = 0,
        repetitionPenalty: Float = 1,
        minP: Float = 0,
        xtcProbability: Float = 0,
        xtcThreshold: Float = 0.1,
        presencePenalty: Float = 0,
        frequencyPenalty: Float = 0,
        minTokens: Int = 0,
        logitBias: [Int: Float] = [:],
        stop: [String] = [],
        seed: Int? = nil,
        stream: Bool = false,
        includeUsage: Bool = false,
        enableThinking: Bool? = nil,
        thinkingBudget: Int? = nil,
        chatTemplateKwargs: [String: OpenAIJSONValue]? = nil,
        structuredOutput: StructuredOutputSpec = .none,
        tools: [OpenAIJSONValue]? = nil,
        toolChoice: OpenAIToolChoice? = nil,
        toolGrammar: Bool? = nil,
        cacheSession: String? = nil
    ) {
        self.model = model
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.minP = minP
        self.xtcProbability = xtcProbability
        self.xtcThreshold = xtcThreshold
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.minTokens = max(0, minTokens)
        self.logitBias = logitBias
        self.stop = stop
        self.seed = seed
        self.stream = stream
        self.includeUsage = includeUsage
        self.enableThinking = enableThinking
        self.thinkingBudget = thinkingBudget
        self.chatTemplateKwargs = chatTemplateKwargs
        self.structuredOutput = structuredOutput
        self.tools = tools
        self.toolChoice = toolChoice
        self.toolGrammar = toolGrammar
        self.cacheSession = cacheSession
    }
}

public struct OpenAIChatChunk: Sendable {
    public let text: String
    public let tokenID: Int
    public let finishReason: String?
    public let cachedPromptTokens: Int

    public init(
        text: String,
        tokenID: Int,
        finishReason: String? = nil,
        cachedPromptTokens: Int = 0
    ) {
        self.text = text
        self.tokenID = tokenID
        self.finishReason = finishReason
        self.cachedPromptTokens = cachedPromptTokens
    }
}

public struct OpenAIChatStream: Sendable {
    public let promptTokens: Int
    public let chunks: AsyncThrowingStream<OpenAIChatChunk, Error>
    /// Completes only after the producer acknowledges terminal resource removal.
    public let cancelAndWait: @Sendable () async -> Void

    /// Every HTTP consumer closes its producer on success, early stop, or write failure.
    /// A loop break alone does not terminate an AsyncThrowingStream producer.
    public func withLifetime<Result>(
        _ body: (OpenAIChatStream) async throws -> Result
    ) async throws -> Result {
        do {
            let result = try await body(self)
            await cancelAndWait()
            return result
        } catch {
            await cancelAndWait()
            throw error
        }
    }

    public init(
        promptTokens: Int,
        chunks: AsyncThrowingStream<OpenAIChatChunk, Error>,
        cancelAndWait: @escaping @Sendable () async -> Void = {}
    ) {
        self.promptTokens = promptTokens
        self.chunks = chunks
        self.cancelAndWait = cancelAndWait
    }
}

public extension OpenAIChatRequest {
    internal var startInThinking: Bool {
        enableThinking ?? (thinkingBudget != nil)
    }

    func withCacheSession(_ cacheSession: String?) -> OpenAIChatRequest {
        OpenAIChatRequest(
            model: model,
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            minP: minP,
            xtcProbability: xtcProbability,
            xtcThreshold: xtcThreshold,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty,
            minTokens: minTokens,
            logitBias: logitBias,
            stop: stop,
            seed: seed,
            stream: stream,
            includeUsage: includeUsage,
            enableThinking: enableThinking,
            thinkingBudget: thinkingBudget,
            chatTemplateKwargs: chatTemplateKwargs,
            structuredOutput: structuredOutput,
            tools: tools,
            toolChoice: toolChoice,
            cacheSession: cacheSession
        )
    }
}
