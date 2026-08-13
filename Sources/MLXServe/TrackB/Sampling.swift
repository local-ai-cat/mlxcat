import MLX

/// Unambiguous alias for cross-module use: upstream MLXLMCommon gained its own
/// `ThinkingBudgetConfiguration` (mlx-swift-lm #521), and the `MLXServe`
/// namespace enum shadows the module name, so `MLXServe.` qualification
/// cannot disambiguate from importing modules.
public typealias ServeThinkingBudgetConfiguration = ThinkingBudgetConfiguration

public struct ThinkingBudgetConfiguration: Sendable, Equatable {
    public var budget: Int
    public var closeTokenIDs: [Int]
    public var startTokenIDs: [Int]
    public var leadingTokenIDs: [Int]
    public var trailingTokenIDs: [Int]
    public var startsInThinking: Bool

    public init(
        budget: Int,
        closeTokenIDs: [Int],
        startTokenIDs: [Int] = [],
        leadingTokenIDs: [Int] = [],
        trailingTokenIDs: [Int] = [],
        startsInThinking: Bool = true
    ) {
        self.budget = max(0, budget)
        self.closeTokenIDs = closeTokenIDs
        self.startTokenIDs = startTokenIDs
        self.leadingTokenIDs = leadingTokenIDs
        self.trailingTokenIDs = trailingTokenIDs
        self.startsInThinking = startsInThinking
    }

    var forceSequence: [Int] {
        leadingTokenIDs + closeTokenIDs + trailingTokenIDs
    }
}

public struct ThinkingBudgetState: Sendable, Equatable {
    private let configuration: ThinkingBudgetConfiguration
    private var thinkingTokens = 0
    private var inThinking: Bool
    private var forcing = false
    private var forceIndex = 0
    private var done = false
    private var skippedForcedToken = false
    private var recentCloseTokens: [Int] = []
    private var recentStartTokens: [Int] = []

    public init(configuration: ThinkingBudgetConfiguration) {
        self.configuration = configuration
        self.inThinking = configuration.startsInThinking
    }

    public var countedThinkingTokens: Int {
        thinkingTokens
    }

    public var isInThinking: Bool {
        inThinking
    }

    public mutating func nextForcedTokenID() -> Int? {
        guard !done else { return nil }
        let sequence = configuration.forceSequence
        guard !sequence.isEmpty else { return nil }
        if forcing {
            guard forceIndex < sequence.count else { return nil }
            return sequence[forceIndex]
        }
        guard inThinking, thinkingTokens >= configuration.budget else { return nil }
        forcing = true
        forceIndex = 0
        recentCloseTokens.removeAll()
        return sequence[forceIndex]
    }

    public mutating func deferForcedToken() {
        guard forcing else { return }
        skippedForcedToken = true
    }

    public mutating func advance(tokenID: Int) {
        if forcing && !skippedForcedToken {
            forceIndex += 1
            if forceIndex >= configuration.forceSequence.count {
                forcing = false
                inThinking = false
                done = true
                recentCloseTokens.removeAll()
            }
            return
        }
        skippedForcedToken = false

        if done {
            detectThinkingStart(tokenID: tokenID)
            return
        }

        if detectNaturalClose(tokenID: tokenID) {
            inThinking = false
            done = true
            return
        }

        if inThinking {
            thinkingTokens += 1
        } else {
            detectThinkingStart(tokenID: tokenID)
        }
    }

    private mutating func detectNaturalClose(tokenID: Int) -> Bool {
        let closeIDs = configuration.closeTokenIDs
        guard !closeIDs.isEmpty else { return false }
        recentCloseTokens.append(tokenID)
        if recentCloseTokens.count > closeIDs.count {
            recentCloseTokens.removeFirst()
        }
        return recentCloseTokens == closeIDs
    }

    private mutating func detectThinkingStart(tokenID: Int) {
        let startIDs = configuration.startTokenIDs
        guard !startIDs.isEmpty else { return }
        recentStartTokens.append(tokenID)
        if recentStartTokens.count > startIDs.count {
            recentStartTokens.removeFirst()
        }
        if recentStartTokens == startIDs {
            inThinking = true
            done = false
            thinkingTokens = 0
            recentCloseTokens.removeAll()
        }
    }
}

public struct SamplingParameters: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float
    public var presencePenalty: Float
    public var frequencyPenalty: Float
    public var minTokens: Int
    public var logitBias: [Int: Float]
    public var eosTokenIds: Set<Int>
    public var xtcProbability: Float
    public var xtcThreshold: Float
    public var xtcSpecialTokens: [Int]
    public var seed: Int?
    public var logprobCount: Int?
    public var allowedSequences: [[Int]]?
    public var jsonGrammar: JSONGrammarConfiguration?
    public var regexGrammar: RegexGrammarConfiguration?
    public var gbnfGrammar: GBNFGrammarConfiguration?
    public var thinkingBudget: ThinkingBudgetConfiguration?

    public init(
        temperature: Float = 0,
        topP: Float = 0,
        topK: Int = 0,
        minP: Float = 0,
        repetitionPenalty: Float = 1,
        presencePenalty: Float = 0,
        frequencyPenalty: Float = 0,
        minTokens: Int = 0,
        logitBias: [Int: Float] = [:],
        eosTokenIds: Set<Int> = [],
        xtcProbability: Float = 0,
        xtcThreshold: Float = 0.1,
        xtcSpecialTokens: [Int] = [],
        seed: Int? = nil,
        logprobCount: Int? = nil,
        allowedSequences: [[Int]]? = nil,
        jsonGrammar: JSONGrammarConfiguration? = nil,
        regexGrammar: RegexGrammarConfiguration? = nil,
        gbnfGrammar: GBNFGrammarConfiguration? = nil,
        thinkingBudget: ThinkingBudgetConfiguration? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.minTokens = max(0, minTokens)
        self.logitBias = logitBias
        self.eosTokenIds = eosTokenIds
        self.xtcProbability = xtcProbability
        self.xtcThreshold = xtcThreshold
        self.xtcSpecialTokens = xtcSpecialTokens
        self.seed = seed
        self.logprobCount = logprobCount
        self.allowedSequences = allowedSequences
        self.jsonGrammar = jsonGrammar
        self.regexGrammar = regexGrammar
        self.gbnfGrammar = gbnfGrammar
        self.thinkingBudget = thinkingBudget
    }

    var hasSamplingFiltersOrPenalties: Bool {
        (topP > 0 && topP < 1)
            || topK > 0
            || minP > 0
            || repetitionPenalty != 1
            || presencePenalty != 0
            || frequencyPenalty != 0
            || xtcProbability > 0
    }
}

public enum TokenSampler {
    static func sampleBatch(
        logits: MLXArray,
        parameters: [SamplingParameters],
        generatedTokenHistories: [[Int]],
        randomStates: [MLXRandom.RandomState?]
    ) -> MLXArray? {
        guard logits.shape.count == 2,
            !parameters.isEmpty,
            parameters.count == generatedTokenHistories.count,
            parameters.count == randomStates.count,
            let sharedParameters = sharedBatchParameters(
                parameters,
                randomStates: randomStates
            )
        else {
            return nil
        }

        var logits = applyBatchedLogitsProcessors(
            logits,
            parameters: sharedParameters,
            generatedTokenHistories: generatedTokenHistories
        )
        if sharedParameters.temperature == 0 && !sharedParameters.hasSamplingFiltersOrPenalties {
            return argMax(logits, axis: -1).asType(.int32).reshaped([parameters.count])
        }

        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }
        logits = applyBatchedPenalties(
            logits,
            parameters: sharedParameters,
            generatedTokenHistories: generatedTokenHistories
        )

        if sharedParameters.temperature == 0 {
            return argMax(logits, axis: -1).asType(.int32).reshaped([parameters.count])
        }

        var logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        if sharedParameters.topP > 0 && sharedParameters.topP < 1 {
            logprobs = applyTopP(logprobs, topP: sharedParameters.topP)
        }
        if sharedParameters.minP > 0 {
            logprobs = applyMinP(logprobs, minP: sharedParameters.minP)
        }
        if sharedParameters.topK > 0 {
            logprobs = applyTopK(logprobs, topK: sharedParameters.topK)
        }

        let scaled = logprobs / MLXArray(sharedParameters.temperature)
        return MLXRandom.categorical(scaled, axis: -1).asType(.int32).reshaped([parameters.count])
    }

    /// Samples from raw pre-softmax logits; penalties must be applied in logit space.
    public static func sample(
        logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokens: [Int] = [],
        jsonGrammarMatcher: JSONGrammarMatcher? = nil,
        regexGrammarMatcher: RegexGrammarMatcher? = nil,
        gbnfGrammarMatcher: GBNFGrammarMatcher? = nil,
        randomState: MLXRandom.RandomState? = nil
    ) -> MLXArray {
        var noThinkingBudgetState: ThinkingBudgetState?
        return sample(
            logits: logits,
            parameters: parameters,
            generatedTokens: generatedTokens,
            jsonGrammarMatcher: jsonGrammarMatcher,
            regexGrammarMatcher: regexGrammarMatcher,
            gbnfGrammarMatcher: gbnfGrammarMatcher,
            randomState: randomState,
            thinkingBudgetState: &noThinkingBudgetState
        )
    }

    public static func sample(
        logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokens: [Int] = [],
        jsonGrammarMatcher: JSONGrammarMatcher? = nil,
        regexGrammarMatcher: RegexGrammarMatcher? = nil,
        gbnfGrammarMatcher: GBNFGrammarMatcher? = nil,
        randomState: MLXRandom.RandomState? = nil,
        thinkingBudgetState: inout ThinkingBudgetState?
    ) -> MLXArray {
        sample(
            logits: logits,
            parameters: parameters,
            generatedTokens: generatedTokens,
            jsonGrammarMatcher: jsonGrammarMatcher,
            regexGrammarMatcher: regexGrammarMatcher,
            gbnfGrammarMatcher: gbnfGrammarMatcher,
            randomState: randomState,
            thinkingBudgetState: &thinkingBudgetState,
            precomputedGrammarMasks: nil
        )
    }

    static func sample(
        logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokens: [Int] = [],
        jsonGrammarMatcher: JSONGrammarMatcher? = nil,
        regexGrammarMatcher: RegexGrammarMatcher? = nil,
        gbnfGrammarMatcher: GBNFGrammarMatcher? = nil,
        randomState: MLXRandom.RandomState? = nil,
        thinkingBudgetState: inout ThinkingBudgetState?,
        precomputedGrammarMasks: PrecomputedGrammarMasks?
    ) -> MLXArray {
        var logits = logits
        if logits.shape.count > 1 {
            logits = logits.reshaped([logits.dim(-1)])
        }
        if let forcedTokenID = thinkingBudgetState?.nextForcedTokenID() {
            if acceptsForcedToken(
                forcedTokenID,
                parameters: parameters,
                generatedTokens: generatedTokens,
                jsonGrammarMatcher: jsonGrammarMatcher,
                regexGrammarMatcher: regexGrammarMatcher,
                gbnfGrammarMatcher: gbnfGrammarMatcher
            ) {
                return MLXArray([Int32(forcedTokenID)])
            }
            thinkingBudgetState?.deferForcedToken()
        }
        logits = applyLogitsProcessors(logits, parameters: parameters, generatedTokens: generatedTokens)
        if let allowedSequences = parameters.allowedSequences,
            let allowedTokenIDs = allowedNextTokenIDs(
                allowedSequences: allowedSequences,
                generatedTokens: generatedTokens
            )
        {
            // TRUE constrained decode (prefix trie); choice mode.
            logits = applyAllowedTokenMask(logits, allowedTokenIDs: allowedTokenIDs)
        }
        if let jsonGrammarMatcher {
            // Rejection fast path, greedy only: validating the argmax candidate costs a
            // single prefix parse; the full vocabulary mask costs one per token. Restricted
            // to temp-0 with no filters because there it is trivially exact (a valid argmax
            // equals the masked argmax). With truncation filters (top-p/k/min-p/xtc) or
            // temperature the retry would sample from a differently-truncated candidate
            // set, skewing the constrained distribution — those always mask first.
            if parameters.temperature == 0 && !parameters.hasSamplingFiltersOrPenalties {
                let candidate = argMax(logits, axis: -1).reshaped([1])
                if jsonGrammarMatcher.accepts(tokenID: candidate.item(Int.self)) {
                    return candidate
                }
            }
            logits = applyAllowedTokenMask(
                logits,
                allowedTokenIDs: precomputedGrammarMasks?.jsonAllowedTokenIDs
                    ?? jsonGrammarMatcher.allowedTokenIDs()
            )
        }
        if let regexGrammarMatcher {
            if parameters.temperature == 0 && !parameters.hasSamplingFiltersOrPenalties {
                let candidate = argMax(logits, axis: -1).reshaped([1])
                if regexGrammarMatcher.accepts(tokenID: candidate.item(Int.self)) {
                    return candidate
                }
            }
            logits = applyAllowedTokenMask(
                logits,
                allowedTokenIDs: precomputedGrammarMasks?.regexAllowedTokenIDs
                    ?? regexGrammarMatcher.allowedTokenIDs()
            )
        }
        if let gbnfGrammarMatcher {
            if parameters.temperature == 0 && !parameters.hasSamplingFiltersOrPenalties {
                let candidate = argMax(logits, axis: -1).reshaped([1])
                if gbnfGrammarMatcher.accepts(tokenID: candidate.item(Int.self)) {
                    return candidate
                }
            }
            logits = applyAllowedTokenMask(
                logits,
                allowedTokenIDs: precomputedGrammarMasks?.gbnfAllowedTokenIDs
                    ?? gbnfGrammarMatcher.allowedTokenIDs()
            )
        }
        return sampleUnconstrained(
            logits: logits,
            parameters: parameters,
            generatedTokens: generatedTokens,
            randomState: randomState
        )
    }

    private static func acceptsForcedToken(
        _ tokenID: Int,
        parameters: SamplingParameters,
        generatedTokens: [Int],
        jsonGrammarMatcher: JSONGrammarMatcher?,
        regexGrammarMatcher: RegexGrammarMatcher?,
        gbnfGrammarMatcher: GBNFGrammarMatcher?
    ) -> Bool {
        if let allowedSequences = parameters.allowedSequences,
            let allowedTokenIDs = allowedNextTokenIDs(
                allowedSequences: allowedSequences,
                generatedTokens: generatedTokens
            ),
            !allowedTokenIDs.contains(tokenID)
        {
            return false
        }
        if let jsonGrammarMatcher, !jsonGrammarMatcher.accepts(tokenID: tokenID) {
            return false
        }
        if let regexGrammarMatcher, !regexGrammarMatcher.accepts(tokenID: tokenID) {
            return false
        }
        if let gbnfGrammarMatcher, !gbnfGrammarMatcher.accepts(tokenID: tokenID) {
            return false
        }
        if generatedTokens.count < parameters.minTokens && parameters.eosTokenIds.contains(tokenID) {
            return false
        }
        return true
    }

    private static func sampleUnconstrained(
        logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokens: [Int],
        randomState: MLXRandom.RandomState?
    ) -> MLXArray {
        var logits = logits
        if parameters.temperature == 0 && !parameters.hasSamplingFiltersOrPenalties {
            return argMax(logits, axis: -1).reshaped([1])
        }

        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }
        logits = applyPenalties(logits, parameters: parameters, generatedTokens: generatedTokens)

        if parameters.temperature == 0 {
            return argMax(logits, axis: -1).reshaped([1])
        }

        var logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
        if parameters.topP > 0 && parameters.topP < 1 {
            logprobs = applyTopP(logprobs, topP: parameters.topP)
        }
        if parameters.minP > 0 {
            logprobs = applyMinP(logprobs, minP: parameters.minP)
        }
        if parameters.xtcProbability > 0 {
            logprobs = applyXTC(
                logprobs,
                probability: parameters.xtcProbability,
                threshold: parameters.xtcThreshold,
                specialTokens: parameters.xtcSpecialTokens,
                randomState: randomState
            )
        }
        if parameters.topK > 0 {
            logprobs = applyTopK(logprobs, topK: parameters.topK)
        }

        let scaled = logprobs / MLXArray(parameters.temperature)
        if let randomState {
            return MLXRandom.categorical(scaled, axis: -1, key: randomState).asType(.int32).reshaped([1])
        }
        return MLXRandom.categorical(scaled, axis: -1).asType(.int32).reshaped([1])
    }

    public static func allowedNextTokenIDs(
        allowedSequences: [[Int]],
        generatedTokens: [Int]
    ) -> [Int]? {
        guard !allowedSequences.isEmpty else { return nil }

        var allowed = Set<Int>()
        for sequence in allowedSequences {
            guard generatedTokens.count <= sequence.count else { continue }
            guard sequenceMatchesPrefix(sequence, prefix: generatedTokens) else { continue }
            guard generatedTokens.count < sequence.count else { continue }
            allowed.insert(sequence[generatedTokens.count])
        }

        guard !allowed.isEmpty else { return nil }
        return allowed.sorted()
    }

    private static func sequenceMatchesPrefix(_ sequence: [Int], prefix: [Int]) -> Bool {
        for index in prefix.indices {
            if sequence[index] != prefix[index] {
                return false
            }
        }
        return true
    }

    private static func applyAllowedTokenMask(_ logits: MLXArray, allowedTokenIDs: [Int]) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        let vocabularySize = logits.dim(-1)
        let validTokenIDs = allowedTokenIDs.filter { $0 >= 0 && $0 < vocabularySize }
        guard !validTokenIDs.isEmpty else { return logits }

        let indices = MLXArray(validTokenIDs.map(Int32.init)).asType(.uint32)
        let masked = MLXArray(Array(repeating: -Float.infinity, count: vocabularySize))
        return putAlong(masked, indices, values: logits[indices], axis: -1)
    }

    private static func applyPenalties(
        _ logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokens: [Int]
    ) -> MLXArray {
        guard !generatedTokens.isEmpty else { return logits }

        var result = logits
        let counts = tokenCounts(generatedTokens)
        let uniqueTokens = Array(counts.keys).sorted()
        let indices = MLXArray(uniqueTokens.map(Int32.init)).asType(.uint32)

        if parameters.repetitionPenalty != 1 {
            var selectedLogits = result[indices]
            selectedLogits = MLX.where(
                selectedLogits .< 0,
                selectedLogits * parameters.repetitionPenalty,
                selectedLogits / parameters.repetitionPenalty
            )
            result = putAlong(result, indices, values: selectedLogits, axis: -1)
        }

        if parameters.presencePenalty != 0 {
            result = putAlong(
                result,
                indices,
                values: result[indices] - parameters.presencePenalty,
                axis: -1
            )
        }

        if parameters.frequencyPenalty != 0 {
            let frequencyPenalties = uniqueTokens.map { token in
                Float(counts[token] ?? 0) * parameters.frequencyPenalty
            }
            result = putAlong(
                result,
                indices,
                values: result[indices] - MLXArray(frequencyPenalties),
                axis: -1
            )
        }

        return result
    }

    private static func applyLogitsProcessors(
        _ logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokens: [Int]
    ) -> MLXArray {
        var result = logits
        if result.dtype == .bfloat16 && (!parameters.logitBias.isEmpty || shouldApplyMinTokens(parameters, generatedTokens)) {
            result = result.asType(.float32)
        }
        result = applyLogitBias(result, logitBias: parameters.logitBias)
        if shouldApplyMinTokens(parameters, generatedTokens) {
            result = maskTokens(result, tokenIDs: Array(parameters.eosTokenIds), value: -Float.infinity)
        }
        return result
    }

    private static func applyBatchedLogitsProcessors(
        _ logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokenHistories: [[Int]]
    ) -> MLXArray {
        var result = logits
        if result.dtype == .bfloat16 && (!parameters.logitBias.isEmpty || shouldApplyMinTokens(
            parameters,
            generatedTokenHistories
        )) {
            result = result.asType(.float32)
        }
        result = applyBatchedLogitBias(result, logitBias: parameters.logitBias)
        if shouldApplyMinTokens(parameters, generatedTokenHistories) {
            result = applyBatchedMinTokensMask(
                result,
                parameters: parameters,
                generatedTokenHistories: generatedTokenHistories
            )
        }
        return result
    }

    private static func shouldApplyMinTokens(
        _ parameters: SamplingParameters,
        _ generatedTokens: [Int]
    ) -> Bool {
        parameters.minTokens > 0
            && generatedTokens.count < parameters.minTokens
            && !parameters.eosTokenIds.isEmpty
    }

    private static func shouldApplyMinTokens(
        _ parameters: SamplingParameters,
        _ generatedTokenHistories: [[Int]]
    ) -> Bool {
        parameters.minTokens > 0
            && !parameters.eosTokenIds.isEmpty
            && generatedTokenHistories.contains { $0.count < parameters.minTokens }
    }

    private static func applyLogitBias(_ logits: MLXArray, logitBias: [Int: Float]) -> MLXArray {
        guard !logitBias.isEmpty else { return logits }
        let vocabularySize = logits.dim(-1)
        let validTokenIDs = logitBias.keys.filter { $0 >= 0 && $0 < vocabularySize }.sorted()
        guard !validTokenIDs.isEmpty else { return logits }

        let indices = MLXArray(validTokenIDs.map(Int32.init)).asType(.uint32)
        let biases = MLXArray(validTokenIDs.map { logitBias[$0] ?? 0 })
        return putAlong(logits, indices, values: logits[indices] + biases, axis: -1)
    }

    private static func applyBatchedLogitBias(_ logits: MLXArray, logitBias: [Int: Float]) -> MLXArray {
        guard !logitBias.isEmpty else { return logits }
        let batchSize = logits.dim(0)
        let vocabularySize = logits.dim(-1)
        var biases = [Float](repeating: 0, count: batchSize * vocabularySize)
        let validTokenIDs = logitBias.keys.filter { $0 >= 0 && $0 < vocabularySize }
        guard !validTokenIDs.isEmpty else { return logits }

        for row in 0 ..< batchSize {
            let base = row * vocabularySize
            for tokenID in validTokenIDs {
                biases[base + tokenID] = logitBias[tokenID] ?? 0
            }
        }
        return logits + MLXArray(biases).reshaped([batchSize, vocabularySize])
    }

    private static func maskTokens(_ logits: MLXArray, tokenIDs: [Int], value: Float) -> MLXArray {
        let vocabularySize = logits.dim(-1)
        let validTokenIDs = tokenIDs.filter { $0 >= 0 && $0 < vocabularySize }.sorted()
        guard !validTokenIDs.isEmpty else { return logits }

        let indices = MLXArray(validTokenIDs.map(Int32.init)).asType(.uint32)
        let values = MLXArray(Array(repeating: value, count: validTokenIDs.count))
        return putAlong(logits, indices, values: values, axis: -1)
    }

    private static func applyBatchedMinTokensMask(
        _ logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokenHistories: [[Int]]
    ) -> MLXArray {
        let batchSize = generatedTokenHistories.count
        let vocabularySize = logits.dim(-1)
        var mask = [Float](repeating: 0, count: batchSize * vocabularySize)
        for row in generatedTokenHistories.indices where generatedTokenHistories[row].count < parameters.minTokens {
            for tokenID in parameters.eosTokenIds where tokenID >= 0 && tokenID < vocabularySize {
                mask[row * vocabularySize + tokenID] = 1
            }
        }
        let maskArray = MLXArray(mask).reshaped([batchSize, vocabularySize])
        return MLX.where(maskArray .> MLXArray(Float(0)), MLXArray(-Float.infinity), logits)
    }

    private static func tokenCounts(_ tokens: [Int]) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for token in tokens {
            counts[token, default: 0] += 1
        }
        return counts
    }

    private static func applyBatchedPenalties(
        _ logits: MLXArray,
        parameters: SamplingParameters,
        generatedTokenHistories: [[Int]]
    ) -> MLXArray {
        guard !generatedTokenHistories.allSatisfy(\.isEmpty) else {
            return logits
        }

        let counts = batchedTokenCounts(
            generatedTokenHistories,
            vocabularySize: logits.dim(-1)
        )
        let seen = counts .> MLXArray(Float(0))
        var result = logits

        if parameters.repetitionPenalty != 1 {
            result = MLX.where(
                seen,
                MLX.where(
                    result .< 0,
                    result * parameters.repetitionPenalty,
                    result / parameters.repetitionPenalty
                ),
                result
            )
        }
        if parameters.presencePenalty != 0 {
            result = result - MLX.where(seen, MLXArray(parameters.presencePenalty), MLXArray(Float(0)))
        }
        if parameters.frequencyPenalty != 0 {
            result = result - counts * parameters.frequencyPenalty
        }
        return result
    }

    private static func batchedTokenCounts(
        _ generatedTokenHistories: [[Int]],
        vocabularySize: Int
    ) -> MLXArray {
        var counts = [Float](repeating: 0, count: generatedTokenHistories.count * vocabularySize)
        for row in generatedTokenHistories.indices {
            let base = row * vocabularySize
            for tokenID in generatedTokenHistories[row] where tokenID >= 0 && tokenID < vocabularySize {
                counts[base + tokenID] += 1
            }
        }
        return MLXArray(counts).reshaped([generatedTokenHistories.count, vocabularySize])
    }

    private static func sharedBatchParameters(
        _ parameters: [SamplingParameters],
        randomStates: [MLXRandom.RandomState?]
    ) -> SamplingParameters? {
        var reference = parameters[0]
        reference.seed = nil
        guard canBatch(reference) else { return nil }
        if reference.temperature > 0 && randomStates.contains(where: { $0 != nil }) {
            return nil
        }

        for parameter in parameters {
            var candidate = parameter
            candidate.seed = nil
            guard candidate == reference, canBatch(candidate) else {
                return nil
            }
        }
        return reference
    }

    private static func canBatch(_ parameters: SamplingParameters) -> Bool {
        parameters.allowedSequences == nil
            && parameters.jsonGrammar == nil
            && parameters.regexGrammar == nil
            && parameters.gbnfGrammar == nil
            && parameters.thinkingBudget == nil
            && parameters.xtcProbability == 0
    }

    private static func applyTopP(_ logprobs: MLXArray, topP: Float) -> MLXArray {
        let topP = MLXArray(topP)
        let negInf = MLXArray(-Float.infinity)
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let cumulativeProbs = cumsum(exp(sortedLogprobs), axis: -1)
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    private static func applyMinP(_ logprobs: MLXArray, minP: Float) -> MLXArray {
        let threshold = logprobs.max(axis: -1, keepDims: true) + log(MLXArray(minP))
        return MLX.where(logprobs .>= threshold, logprobs, MLXArray(-Float.infinity))
    }

    private static func applyXTC(
        _ logprobs: MLXArray,
        probability: Float,
        threshold: Float,
        specialTokens: [Int],
        randomState: MLXRandom.RandomState?
    ) -> MLXArray {
        let threshold = max(0, min(threshold, 0.5))
        let probability = max(0, min(probability, 1))
        let probs = softmax(logprobs, axis: -1)
        let aboveThreshold = MLX.where(probs .> threshold, probs, MLXArray(Float.infinity))
        let cutoff = aboveThreshold.min(axis: -1, keepDims: true)
        var mask = probs .> cutoff
        let validSpecialTokens = specialTokens.filter { $0 >= 0 && $0 < logprobs.dim(-1) }
        if !validSpecialTokens.isEmpty {
            let indices = MLXArray(validSpecialTokens.map(Int32.init)).asType(.uint32)
            let values = MLXArray(Array(repeating: false, count: validSpecialTokens.count))
            mask = putAlong(mask, indices, values: values, axis: -1)
        }
        let shouldApply =
            if let randomState {
                MLXRandom.uniform(0 ..< 1, key: randomState) .<= probability
            } else {
                MLXRandom.uniform(0 ..< 1) .<= probability
            }
        return MLX.where(shouldApply, MLX.where(mask, MLXArray(-Float.infinity), logprobs), logprobs)
    }

    private static func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK > 0 && topK < vocabularySize else {
            return logprobs
        }

        let partitioned = argPartition(-logprobs, kth: topK - 1, axis: -1)
        let maskIndices = if logprobs.shape.count == 1 {
            partitioned[topK...]
        } else {
            partitioned[0..., topK...]
        }
        return putAlong(logprobs, maskIndices, values: MLXArray(-Float.infinity), axis: -1)
    }
}
