import MLX
import MLXLMCommon

public struct BatchDecodeStep {
    public let tokenIds: [Int]
    public let logits: MLXArray
}

public struct SpeculativeDecodingConfiguration: Sendable, Equatable {
    public var enabled: Bool
    public var maxProposalTokens: Int
    public var maxSuffixTokens: Int
    public var minContextTokens: Int

    public init(
        enabled: Bool = true,
        maxProposalTokens: Int = 4,
        maxSuffixTokens: Int = 16,
        minContextTokens: Int = 8
    ) {
        self.enabled = enabled
        self.maxProposalTokens = max(0, maxProposalTokens)
        self.maxSuffixTokens = max(1, maxSuffixTokens)
        self.minContextTokens = max(2, minContextTokens)
    }

    public static let disabled = SpeculativeDecodingConfiguration(enabled: false)
}

public struct SpeculativeDecodingStats: Sendable, Equatable {
    public var proposedTokenCount: Int = 0
    public var acceptedTokenCount: Int = 0
    public var proposalBatchCount: Int = 0
    public var rejectedBatchCount: Int = 0

    public init() {}
}

public enum FinishReason: Sendable, Equatable {
    case stop
    case length
    case cancelled
    case failed(String)
}

public struct Response: Sendable, Equatable {
    public let uid: String
    public let token: Int
    public let finishReason: FinishReason?
    public let logprobs: [Int: Float]?

    public init(
        uid: String,
        token: Int,
        finishReason: FinishReason? = nil,
        logprobs: [Int: Float]? = nil
    ) {
        self.uid = uid
        self.token = token
        self.finishReason = finishReason
        self.logprobs = logprobs
    }
}

public final class StaticBatchGenerator {
    private let model: any LanguageModel
    private var cache: [BatchLayerCache]
    private var currentTokens: MLXArray
    private var currentLogits: MLXArray
    private var state: LMOutput.State?

    public init(
        model: any LanguageModel,
        inputs: [LMInput],
        parameters: GenerateParameters
    ) throws {
        precondition(!inputs.isEmpty, "StaticBatchGenerator requires at least one input")

        self.model = model

        var rowCaches: [[any KVCache]] = []
        var firstTokens: [MLXArray] = []
        var firstLogits: [MLXArray] = []

        for input in inputs {
            let cache = model.newCache(parameters: parameters)
            let remaining: LMInput.Text
            // state: nil — `cache` is fresh per input, so no prior model state.
            switch try model.prepare(
                input, cache: cache, state: nil,
                windowSize: parameters.prefillStepSize) {
            case .tokens(let tokens):
                remaining = tokens
            case .logits(let output):
                let logits = output.logits[0..., -1, 0...]
                firstLogits.append(logits)
                firstTokens.append(argMax(logits, axis: -1))
                rowCaches.append(cache)
                continue
            }

            let output = model(remaining[text: .newAxis], cache: cache, state: nil)
            let logits = output.logits[0..., -1, 0...]
            firstLogits.append(logits)
            firstTokens.append(argMax(logits, axis: -1))
            rowCaches.append(cache)
        }

        cache = try (0 ..< rowCaches[0].count).map { layer in
            try BatchLayerCache.merge(rowCaches.map { $0[layer] })
        }
        currentTokens = concatenated(firstTokens, axis: 0)
        currentLogits = concatenated(firstLogits, axis: 0)
        eval(currentTokens, currentLogits, cache.map(\.kvCache))
    }

    public func next() -> BatchDecodeStep {
        let returnedTokens = currentTokens
        let returnedLogits = currentLogits

        let output = model(
            LMInput.Text(tokens: currentTokens[0..., .newAxis]),
            cache: cache.map(\.kvCache),
            state: state
        )
        state = output.state
        currentLogits = output.logits[0..., -1, 0...]
        currentTokens = argMax(currentLogits, axis: -1)

        asyncEval(currentTokens)
        eval(returnedTokens, returnedLogits)

        return BatchDecodeStep(
            tokenIds: returnedTokens.asArray(Int.self),
            logits: returnedLogits
        )
    }
}

public final class ContinuousBatchGenerator {
    private let model: any LanguageModel
    private let parameters: GenerateParameters
    private let speculativeDecoding: SpeculativeDecodingConfiguration
    private var cache: [BatchLayerCache] = []
    private var currentTokens = MLXArray([Int32]())
    private var rowUIDs: [String] = []
    private var samplers: [SamplingParameters] = []
    private var randomStates: [MLXRandom.RandomState?] = []
    private var jsonGrammarMatchers: [JSONGrammarMatcher?] = []
    private var regexGrammarMatchers: [RegexGrammarMatcher?] = []
    private var gbnfGrammarMatchers: [GBNFGrammarMatcher?] = []
    private var jsonGrammarMasks: [AsyncGrammarMask<JSONGrammarMaskSnapshot>?] = []
    private var regexGrammarMasks: [AsyncGrammarMask<RegexGrammarMaskSnapshot>?] = []
    private var gbnfGrammarMasks: [AsyncGrammarMask<GBNFGrammarMaskSnapshot>?] = []
    private var thinkingBudgetStates: [ThinkingBudgetState?] = []
    private var generatedTokenHistory: [[Int]] = []
    private var speculativeTokenHistory: [[Int]] = []
    private var maxGeneratedTokens: [Int?] = []
    private var state: LMOutput.State?
    private var speculativeStats = SpeculativeDecodingStats()

    public init(
        model: any LanguageModel,
        parameters: GenerateParameters,
        speculativeDecoding: SpeculativeDecodingConfiguration = SpeculativeDecodingConfiguration()
    ) {
        self.model = model
        self.parameters = parameters
        self.speculativeDecoding = speculativeDecoding
    }

    public var isEmpty: Bool {
        rowUIDs.isEmpty
    }

    public var count: Int {
        rowUIDs.count
    }

    public var uids: [String] {
        rowUIDs
    }

    public var speculationStats: SpeculativeDecodingStats {
        speculativeStats
    }

    @discardableResult
    public func insert(
        uid: String,
        input: LMInput,
        sampling: SamplingParameters,
        maxGeneratedTokens: Int? = nil,
        speculativeContextTokens: [Int] = []
    ) throws -> Response? {
        let rowCache = model.newCache(parameters: parameters)
        // state: nil — `rowCache` was just created above, so no prior state.
        switch try model.prepare(
            input, cache: rowCache, state: nil,
            windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            if tokens.tokens.dim(0) == 1 {
                let output = model(tokens[text: .newAxis], cache: rowCache, state: nil)
                eval(rowCache)
                let randomState = Self.randomState(for: sampling.seed)
                let firstToken = sampledToken(
                    from: output.logits,
                    sampling: sampling,
                    randomState: randomState
                )
                let tokenID = firstToken.token.item(Int.self)
                try insert(
                    uid: uid,
                    cache: rowCache,
                    lastToken: firstToken.token,
                    sampling: sampling,
                    generatedTokens: [tokenID],
                    maxGeneratedTokens: maxGeneratedTokens,
                    speculativeContextTokens: speculativeContextTokens + [tokenID],
                    thinkingBudgetState: firstToken.thinkingBudgetState,
                    randomState: randomState
                )
                return Response(uid: uid, token: tokenID)
            }
            let lastToken = try prefillWithLastTokenWithheld(tokens, cache: rowCache)
            try insert(
                uid: uid,
                cache: rowCache,
                lastToken: lastToken,
                sampling: sampling,
                maxGeneratedTokens: maxGeneratedTokens,
                speculativeContextTokens: speculativeContextTokens
            )
            return nil
        case .logits(let output):
            let randomState = Self.randomState(for: sampling.seed)
            let firstToken = sampledToken(
                from: output.logits,
                sampling: sampling,
                randomState: randomState
            )
            let tokenID = firstToken.token.item(Int.self)
            try insert(
                uid: uid,
                cache: rowCache,
                lastToken: firstToken.token,
                sampling: sampling,
                generatedTokens: [tokenID],
                maxGeneratedTokens: maxGeneratedTokens,
                speculativeContextTokens: speculativeContextTokens + [tokenID],
                thinkingBudgetState: firstToken.thinkingBudgetState,
                randomState: randomState
            )
            return Response(uid: uid, token: tokenID)
        }
    }

    public func insert(
        uid: String,
        cache rowCache: [any KVCache],
        lastToken: MLXArray,
        sampling: SamplingParameters,
        generatedTokens: [Int] = [],
        maxGeneratedTokens maxGeneratedTokenCount: Int? = nil,
        speculativeContextTokens: [Int] = [],
        thinkingBudgetState initialThinkingBudgetState: ThinkingBudgetState? = nil,
        randomState initialRandomState: MLXRandom.RandomState? = nil
    ) throws {
        precondition(!rowUIDs.contains(uid), "duplicate batch uid '\(uid)'")
        precondition(!rowCache.isEmpty, "continuous batching requires a non-empty KV cache")

        if cache.isEmpty {
            cache = try rowCache.map { try BatchLayerCache.adoptSingle($0) }
            currentTokens = lastToken.reshaped([1])
        } else {
            guard cache.count == rowCache.count else {
                throw BatchGeneratorError.insertedCacheLayerCountChanged(
                    expected: cache.count,
                    actual: rowCache.count
                )
            }
            let rowLayers = try rowCache.map { try BatchLayerCache.merge([$0]) }
            cache = try zip(cache, rowLayers).map { currentLayer, rowLayer in
                let mergedLayer = currentLayer.copyLayer()
                try mergedLayer.extend(rowLayer)
                return mergedLayer
            }
            currentTokens = concatenated([currentTokens, lastToken.reshaped([1])], axis: 0)
        }

        rowUIDs.append(uid)
        samplers.append(sampling)
        randomStates.append(initialRandomState ?? Self.randomState(for: sampling.seed))
        let matcher = sampling.jsonGrammar?.makeMatcher()
        let regexMatcher = sampling.regexGrammar?.makeMatcher()
        let gbnfMatcher = sampling.gbnfGrammar?.makeMatcher()
        var thinkingBudgetState = initialThinkingBudgetState
            ?? sampling.thinkingBudget.map(ThinkingBudgetState.init(configuration:))
        for token in generatedTokens {
            matcher?.advance(tokenID: token)
            regexMatcher?.advance(tokenID: token)
            gbnfMatcher?.advance(tokenID: token)
            if initialThinkingBudgetState == nil {
                thinkingBudgetState?.advance(tokenID: token)
            }
        }
        jsonGrammarMatchers.append(matcher)
        regexGrammarMatchers.append(regexMatcher)
        gbnfGrammarMatchers.append(gbnfMatcher)
        let jsonMask = matcher.map { matcher in
            let mask = AsyncGrammarMask<JSONGrammarMaskSnapshot>()
            mask.prepareCurrentState(from: matcher.makeMaskSnapshot())
            return mask
        }
        let regexMask = regexMatcher.map { matcher in
            let mask = AsyncGrammarMask<RegexGrammarMaskSnapshot>()
            mask.prepareCurrentState(from: matcher.makeMaskSnapshot())
            return mask
        }
        let gbnfMask = gbnfMatcher.map { matcher in
            let mask = AsyncGrammarMask<GBNFGrammarMaskSnapshot>()
            mask.prepareCurrentState(from: matcher.makeMaskSnapshot())
            return mask
        }
        jsonGrammarMasks.append(jsonMask)
        regexGrammarMasks.append(regexMask)
        gbnfGrammarMasks.append(gbnfMask)
        thinkingBudgetStates.append(thinkingBudgetState)
        generatedTokenHistory.append(generatedTokens)
        speculativeTokenHistory.append(
            speculativeContextTokens.isEmpty
                ? generatedTokens
                : speculativeContextTokens
        )
        maxGeneratedTokens.append(maxGeneratedTokenCount)
        state = nil
        eval(currentTokens, cache.map(\.kvCache))
    }

    public func remove(uid: String) {
        guard let row = rowUIDs.firstIndex(of: uid) else { return }
        let keptRows = rowUIDs.indices.filter { $0 != row }
        filter(keeping: keptRows)
    }

    public func extractCache(uid: String) -> [KVCacheSimple]? {
        guard let row = rowUIDs.firstIndex(of: uid) else { return nil }
        return cache.map { $0.extract(row) }
    }

    public func filter(keeping rows: [Int]) {
        guard !rows.isEmpty else {
            cache.removeAll()
            rowUIDs.removeAll()
            samplers.removeAll()
            randomStates.removeAll()
            jsonGrammarMatchers.removeAll()
            regexGrammarMatchers.removeAll()
            gbnfGrammarMatchers.removeAll()
            jsonGrammarMasks.forEach { $0?.invalidate() }
            regexGrammarMasks.forEach { $0?.invalidate() }
            gbnfGrammarMasks.forEach { $0?.invalidate() }
            jsonGrammarMasks.removeAll()
            regexGrammarMasks.removeAll()
            gbnfGrammarMasks.removeAll()
            thinkingBudgetStates.removeAll()
            generatedTokenHistory.removeAll()
            speculativeTokenHistory.removeAll()
            maxGeneratedTokens.removeAll()
            currentTokens = MLXArray([Int32]())
            state = nil
            return
        }

        for layer in cache {
            layer.filter(keeping: rows)
        }
        let rowIndices = MLXArray(rows.map(Int32.init))
        currentTokens = currentTokens.take(rowIndices, axis: 0)
        rowUIDs = rows.map { rowUIDs[$0] }
        samplers = rows.map { samplers[$0] }
        randomStates = rows.map { randomStates[$0] }
        jsonGrammarMatchers = rows.map { jsonGrammarMatchers[$0] }
        regexGrammarMatchers = rows.map { regexGrammarMatchers[$0] }
        gbnfGrammarMatchers = rows.map { gbnfGrammarMatchers[$0] }
        let keptRows = Set(rows)
        for row in jsonGrammarMasks.indices where !keptRows.contains(row) {
            jsonGrammarMasks[row]?.invalidate()
            regexGrammarMasks[row]?.invalidate()
            gbnfGrammarMasks[row]?.invalidate()
        }
        jsonGrammarMasks = rows.map { jsonGrammarMasks[$0] }
        regexGrammarMasks = rows.map { regexGrammarMasks[$0] }
        gbnfGrammarMasks = rows.map { gbnfGrammarMasks[$0] }
        thinkingBudgetStates = rows.map { thinkingBudgetStates[$0] }
        generatedTokenHistory = rows.map { generatedTokenHistory[$0] }
        speculativeTokenHistory = rows.map { speculativeTokenHistory[$0] }
        maxGeneratedTokens = rows.map { maxGeneratedTokens[$0] }
        state = nil
        eval(currentTokens, cache.map(\.kvCache))
    }

    public func next() -> [Response] {
        guard !rowUIDs.isEmpty else { return [] }
        if let speculativeResponses = speculativeNext() {
            return speculativeResponses
        }

        return decodeOneToken()
    }

    private func decodeOneToken() -> [Response] {
        guard !rowUIDs.isEmpty else { return [] }

        let output = model(
            LMInput.Text(tokens: currentTokens[0..., .newAxis]),
            cache: cache.map(\.kvCache),
            state: state
        )
        state = output.state

        let logits = output.logits[0..., -1, 0...]
        let nextTokens = batchedSample(from: logits) ?? sampleRows(from: logits)

        asyncEval(nextTokens)
        eval(currentTokens, logits)
        currentTokens = nextTokens

        let tokenIds = nextTokens.asArray(Int.self)
        for row in generatedTokenHistory.indices {
            recordGeneratedToken(tokenIds[row], row: row)
        }
        return rowUIDs.enumerated().map { row, uid in
            Response(uid: uid, token: tokenIds[row])
        }
    }

    private func batchedSample(from logits: MLXArray) -> MLXArray? {
        guard jsonGrammarMatchers.allSatisfy({ $0 == nil }),
            regexGrammarMatchers.allSatisfy({ $0 == nil }),
            gbnfGrammarMatchers.allSatisfy({ $0 == nil }),
            thinkingBudgetStates.allSatisfy({ $0 == nil })
        else {
            return nil
        }
        return TokenSampler.sampleBatch(
            logits: logits,
            parameters: samplers,
            generatedTokenHistories: generatedTokenHistory,
            randomStates: randomStates
        )
    }

    private func sampleRows(from logits: MLXArray) -> MLXArray {
        var sampledRows: [MLXArray] = []
        sampledRows.reserveCapacity(rowUIDs.count)
        for row in 0 ..< rowUIDs.count {
            let token = TokenSampler.sample(
                logits: logits[row, 0...],
                parameters: samplers[row],
                generatedTokens: generatedTokenHistory[row],
                jsonGrammarMatcher: jsonGrammarMatchers[row],
                regexGrammarMatcher: regexGrammarMatchers[row],
                gbnfGrammarMatcher: gbnfGrammarMatchers[row],
                randomState: randomStates[row],
                thinkingBudgetState: &thinkingBudgetStates[row],
                precomputedGrammarMasks: PrecomputedGrammarMasks(
                    jsonAllowedTokenIDs: jsonGrammarMasks[row]?.readyTokenIDs,
                    regexAllowedTokenIDs: regexGrammarMasks[row]?.readyTokenIDs,
                    gbnfAllowedTokenIDs: gbnfGrammarMasks[row]?.readyTokenIDs
                )
            )
            sampledRows.append(token)
        }
        return concatenated(sampledRows, axis: 0)
    }

    private func speculativeNext() -> [Response]? {
        guard canSpeculate(row: 0),
            let remainingTokenCount = remainingGeneratedTokenCount(row: 0),
            remainingTokenCount > 1
        else {
            return nil
        }

        let proposalLimit = min(speculativeDecoding.maxProposalTokens, remainingTokenCount - 1)
        guard proposalLimit > 0,
            let proposedTokens = proposeSuffixTokens(
                from: speculativeTokenHistory[0],
                limit: proposalLimit
            ),
            !proposedTokens.isEmpty
        else {
            return nil
        }

        speculativeStats.proposalBatchCount += 1
        speculativeStats.proposedTokenCount += proposedTokens.count

        let verificationTokens = [currentTokens.item(Int.self)] + proposedTokens
        let verificationInput = MLXArray(verificationTokens.map(Int32.init))[.newAxis, 0...]
        let workingCache = cache.map { $0.copyLayer() }
        let output = model(
            LMInput.Text(tokens: verificationInput),
            cache: workingCache.map(\.kvCache),
            state: state
        )

        var acceptedTokens: [Int] = []
        var generatedHistory = generatedTokenHistory[0]
        var acceptedAllProposals = true
        for position in 0 ..< verificationTokens.count {
            let sampled = TokenSampler.sample(
                logits: output.logits[0, position, 0...],
                parameters: samplers[0],
                generatedTokens: generatedHistory
            ).item(Int.self)

            if position < proposedTokens.count, sampled != proposedTokens[position] {
                acceptedAllProposals = false
                break
            }

            acceptedTokens.append(sampled)
            generatedHistory.append(sampled)
            if samplers[0].eosTokenIds.contains(sampled) {
                break
            }
        }

        guard acceptedAllProposals else {
            speculativeStats.rejectedBatchCount += 1
            return decodeOneToken()
        }

        cache = workingCache
        state = output.state
        let nextToken = acceptedTokens.last ?? proposedTokens.last ?? currentTokens.item(Int.self)
        currentTokens = MLXArray([Int32(nextToken)])
        asyncEval(currentTokens)
        eval(output.logits, cache.map(\.kvCache))

        for token in acceptedTokens {
            recordGeneratedToken(token, row: 0)
        }
        speculativeStats.acceptedTokenCount += max(0, acceptedTokens.count - 1)

        return acceptedTokens.map { Response(uid: rowUIDs[0], token: $0) }
    }

    private func canSpeculate(row: Int) -> Bool {
        guard speculativeDecoding.enabled,
            rowUIDs.count == 1,
            row == 0,
            speculativeDecoding.maxProposalTokens > 0,
            speculativeTokenHistory[row].count >= speculativeDecoding.minContextTokens,
            samplers[row].temperature == 0,
            samplers[row].allowedSequences == nil,
            samplers[row].jsonGrammar == nil,
            samplers[row].regexGrammar == nil,
            samplers[row].gbnfGrammar == nil,
            samplers[row].thinkingBudget == nil,
            randomStates[row] == nil
        else {
            return false
        }
        return true
    }

    private func remainingGeneratedTokenCount(row: Int) -> Int? {
        guard let maxGenerated = maxGeneratedTokens[row] else { return Int.max }
        return max(0, maxGenerated - generatedTokenHistory[row].count)
    }

    private func proposeSuffixTokens(from context: [Int], limit: Int) -> [Int]? {
        guard context.count >= speculativeDecoding.minContextTokens, limit > 0 else {
            return nil
        }

        let maxSuffixLength = min(
            speculativeDecoding.maxSuffixTokens,
            context.count - 1
        )
        guard maxSuffixLength > 0 else { return nil }

        for suffixLength in stride(from: maxSuffixLength, through: 1, by: -1) {
            let suffixStart = context.count - suffixLength
            let suffix = Array(context[suffixStart...])
            guard suffixStart > 0 else { continue }

            for candidateStart in stride(from: suffixStart - 1, through: 0, by: -1) {
                let candidateEnd = candidateStart + suffixLength
                guard candidateEnd <= suffixStart else { continue }
                if Array(context[candidateStart ..< candidateEnd]) == suffix {
                    let proposalStart = candidateEnd
                    let proposalEnd = min(context.count, proposalStart + limit)
                    guard proposalStart < proposalEnd else { continue }
                    return Array(context[proposalStart ..< proposalEnd])
                }
            }
        }
        return nil
    }

    private func recordGeneratedToken(_ tokenID: Int, row: Int) {
        generatedTokenHistory[row].append(tokenID)
        speculativeTokenHistory[row].append(tokenID)
        if jsonGrammarMatchers[row]?.accepts(tokenID: tokenID) == true {
            jsonGrammarMatchers[row]?.advance(tokenID: tokenID)
            if let matcher = jsonGrammarMatchers[row] {
                jsonGrammarMasks[row]?.prepareAdvancedState(from: matcher.makeMaskSnapshot())
            }
        }
        if regexGrammarMatchers[row]?.accepts(tokenID: tokenID) == true {
            regexGrammarMatchers[row]?.advance(tokenID: tokenID)
            if let matcher = regexGrammarMatchers[row] {
                regexGrammarMasks[row]?.prepareAdvancedState(from: matcher.makeMaskSnapshot())
            }
        }
        if gbnfGrammarMatchers[row]?.accepts(tokenID: tokenID) == true {
            gbnfGrammarMatchers[row]?.advance(tokenID: tokenID)
            if let matcher = gbnfGrammarMatchers[row] {
                gbnfGrammarMasks[row]?.prepareAdvancedState(from: matcher.makeMaskSnapshot())
            }
        }
        thinkingBudgetStates[row]?.advance(tokenID: tokenID)
    }

    private func prefillWithLastTokenWithheld(
        _ text: LMInput.Text,
        cache rowCache: [any KVCache]
    ) throws -> MLXArray {
        let tokens = text.tokens
        let tokenCount = tokens.dim(0)
        guard tokenCount > 1 else {
            throw BatchGeneratorError.promptTooShortForExternalPrefill
        }

        let prefixTokens = tokens[..<(tokenCount - 1)]
        let prefixInput = LMInput.Text(tokens: prefixTokens)
        _ = model(prefixInput[text: .newAxis], cache: rowCache, state: nil)
        eval(rowCache)

        return tokens[tokenCount - 1]
    }

    private func sampledToken(
        from logits: MLXArray,
        sampling: SamplingParameters,
        randomState: MLXRandom.RandomState?
    ) -> PreparedSampledToken {
        let nextTokenLogits = logits[0..., -1, 0...]
        let matcher = sampling.jsonGrammar?.makeMatcher()
        let regexMatcher = sampling.regexGrammar?.makeMatcher()
        let gbnfMatcher = sampling.gbnfGrammar?.makeMatcher()
        var thinkingBudgetState = sampling.thinkingBudget.map(ThinkingBudgetState.init(configuration:))
        let token = TokenSampler.sample(
            logits: nextTokenLogits[0, 0...],
            parameters: sampling,
            generatedTokens: [],
            jsonGrammarMatcher: matcher,
            regexGrammarMatcher: regexMatcher,
            gbnfGrammarMatcher: gbnfMatcher,
            randomState: randomState,
            thinkingBudgetState: &thinkingBudgetState
        )
        let tokenID = token.item(Int.self)
        if matcher?.accepts(tokenID: tokenID) == true {
            matcher?.advance(tokenID: tokenID)
        }
        if regexMatcher?.accepts(tokenID: tokenID) == true {
            regexMatcher?.advance(tokenID: tokenID)
        }
        if gbnfMatcher?.accepts(tokenID: tokenID) == true {
            gbnfMatcher?.advance(tokenID: tokenID)
        }
        thinkingBudgetState?.advance(tokenID: tokenID)
        return PreparedSampledToken(token: token, thinkingBudgetState: thinkingBudgetState)
    }

    private static func randomState(for seed: Int?) -> MLXRandom.RandomState? {
        seed.map { MLXRandom.RandomState(seed: UInt64(bitPattern: Int64($0))) }
    }
}

private struct PreparedSampledToken {
    let token: MLXArray
    let thinkingBudgetState: ThinkingBudgetState?
}

public enum BatchGeneratorError: Error, Equatable {
    case unsupportedPreparedLogits
    case promptTooShortForExternalPrefill
    case insertedCacheLayerCountChanged(expected: Int, actual: Int)
}
