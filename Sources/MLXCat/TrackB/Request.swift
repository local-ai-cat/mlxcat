import MLXLMCommon

public struct Request: @unchecked Sendable {
    public let uid: String
    public let input: LMInput
    public let maxTokens: Int
    public let sampling: SamplingParameters
    public let eosTokenIds: Set<Int>
    public let cacheSession: String?

    public init(
        uid: String,
        input: LMInput,
        maxTokens: Int,
        sampling: SamplingParameters = SamplingParameters(),
        eosTokenIds: Set<Int> = [],
        cacheSession: String? = nil
    ) {
        self.uid = uid
        self.input = input
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.eosTokenIds = eosTokenIds
        self.cacheSession = cacheSession
    }
}

final class RunningRequest {
    let request: Request
    let promptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
    var generatedTokens: [Int]
    let generatedTokensIncludedInPrompt: Int
    var cachedTokenCount: Int

    init(
        request: Request,
        promptTokens: [Int],
        prefixHit: PrefixKVStoreHit?,
        generatedTokens: [Int],
        generatedTokensIncludedInPrompt: Int,
        cachedTokenCount: Int
    ) {
        self.request = request
        self.promptTokens = promptTokens
        self.prefixHit = prefixHit
        self.generatedTokens = generatedTokens
        self.generatedTokensIncludedInPrompt = generatedTokensIncludedInPrompt
        self.cachedTokenCount = cachedTokenCount
    }

    var generatedTokenCount: Int {
        generatedTokens.count
    }
}

public enum SchedulerError: Error, Equatable {
    case queueFull(retryAfterSteps: Int)
    case duplicateRequest(String)
    case invalidPrefixHit
}
