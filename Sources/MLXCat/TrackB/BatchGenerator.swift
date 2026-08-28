import Foundation
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
        enabled: Bool = false,
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
    public let cachedPromptTokens: Int

    public init(
        uid: String,
        token: Int,
        finishReason: FinishReason? = nil,
        logprobs: [Int: Float]? = nil,
        cachedPromptTokens: Int = 0
    ) {
        self.uid = uid
        self.token = token
        self.finishReason = finishReason
        self.logprobs = logprobs
        self.cachedPromptTokens = cachedPromptTokens
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
            let cache = try model.newCache(parameters: parameters)
            let remaining: LMInput.Text
            // state: nil — `cache` is fresh per input, so no prior model state.
            switch try model.prepare(
                input, cache: cache, state: nil,
                prefill: parameters.prefill) {
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

    private enum CacheStorage {
        case empty
        case singleton([any KVCache])
        case batched([BatchLayerCache])

        var cacheForModel: [any KVCache] {
            switch self {
            case .empty: return []
            case .singleton(let cache): return cache
            case .batched(let cache): return cache.map(\.kvCache)
            }
        }

        var batchedCache: [BatchLayerCache]? {
            if case .batched(let cache) = self { return cache }
            return nil
        }

        var isSingleton: Bool {
            if case .singleton = self { return true }
            return false
        }
    }

    private var cacheStorage: CacheStorage = .empty
    private var currentTokens = MLXArray([Int32]())
    private var rowUIDs: [String] = []
    private var samplers: [SamplingParameters] = []
    private var randomStates: [MLXRandom.RandomState?] = []
    private var jsonGrammarMatchers: [JSONGrammarMatcher?] = []
    private var regexGrammarMatchers: [RegexGrammarMatcher?] = []
    private var gbnfGrammarMatchers: [GBNFGrammarMatcher?] = []
    private var toolGrammarStates: [QwenXMLToolGrammarState?] = []
    private var jsonGrammarMasks: [AsyncGrammarMask<JSONGrammarMaskSnapshot>?] = []
    private var regexGrammarMasks: [AsyncGrammarMask<RegexGrammarMaskSnapshot>?] = []
    private var gbnfGrammarMasks: [AsyncGrammarMask<GBNFGrammarMaskSnapshot>?] = []
    private var thinkingBudgetStates: [ThinkingBudgetState?] = []
    private var generatedTokenHistory: [[Int]] = []
    private var speculativeTokenHistory: [[Int]] = []
    private var maxGeneratedTokens: [Int?] = []
    private var state: LMOutput.State?
    /// Set the first time a row's prefill hands back a recognized positional
    /// anchor. While it is non-nil every decode step feeds a per-row vector
    /// built from `rowPrefillDeltas`, and `state` is ignored — see
    /// ``BatchPositionalState``.
    private var positionalKey: LMOutput.Key<MLXArray>?
    private var rowPrefillDeltas: [Int32] = []
    private var speculativeStats = SpeculativeDecodingStats()

    // Explicit-demand decode-phase timing (MLXSERVE_DECODE_PHASE_TIMING=1):
    // separates graph build / token wait / launch enqueue / bookkeeping per
    // pipelined step so a width-1 deficit can be localized without Instruments.
    private static let phaseTimingEnabled =
        ProcessInfo.processInfo.environment["MLXSERVE_DECODE_PHASE_TIMING"] == "1"
    private var phaseBuildNs: UInt64 = 0
    private var phaseWaitNs: UInt64 = 0
    private var phaseLaunchNs: UInt64 = 0
    private var phaseEmitNs: UInt64 = 0
    private var phaseSteps: UInt64 = 0

    public var phaseTimingSummary: String? {
        guard Self.phaseTimingEnabled, phaseSteps > 0 else { return nil }
        let n = Double(phaseSteps)
        func ms(_ v: UInt64) -> String { String(format: "%.3fms", Double(v) / n / 1_000_000) }
        return "decode phases over \(phaseSteps) steps: build \(ms(phaseBuildNs)), "
            + "wait \(ms(phaseWaitNs)), launch \(ms(phaseLaunchNs)), emit \(ms(phaseEmitNs))"
    }

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

    /// The state fed to every decode step. For families with no positional
    /// anchor this is the state the model handed back last step, unchanged.
    private var stepState: LMOutput.State? {
        guard let positionalKey else { return state }
        return BatchPositionalState.stepState(
            key: positionalKey,
            prefillDeltas: rowPrefillDeltas,
            leftPadding: currentLeftPadding
        )
    }

    /// Left padding of the batched full-attention layers, in row order. A
    /// singleton (width 1) cache has none, and neither does a layer whose state
    /// is not sequence-shaped — the Mamba half of a hybrid model — so the first
    /// `BatchKVCache` in the stack is the one that answers.
    private var currentLeftPadding: [Int32] {
        guard let batched = cacheStorage.batchedCache else {
            return Array(repeating: 0, count: rowUIDs.count)
        }
        for layer in batched {
            if let batchCache = layer.kvCache as? BatchKVCache {
                return batchCache.leftPadding.asType(.int32).asArray(Int32.self)
            }
        }
        return Array(repeating: 0, count: rowUIDs.count)
    }

    @discardableResult
    public func insert(
        uid: String,
        input: LMInput,
        sampling: SamplingParameters,
        maxGeneratedTokens: Int? = nil,
        speculativeContextTokens: [Int] = []
    ) throws -> Response? {
        let rowCache = try model.newCache(parameters: parameters)
        // state: nil — `rowCache` was just created above, so no prior state.
        switch try model.prepare(
            input, cache: rowCache, state: nil,
            prefill: parameters.prefill) {
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
                    randomState: randomState,
                    modelState: output.state
                )
                return Response(uid: uid, token: tokenID)
            }
            let (lastToken, prefillState) = try prefillWithLastTokenWithheld(
                tokens, cache: rowCache)
            try insert(
                uid: uid,
                cache: rowCache,
                lastToken: lastToken,
                sampling: sampling,
                maxGeneratedTokens: maxGeneratedTokens,
                speculativeContextTokens: speculativeContextTokens,
                modelState: prefillState
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
                randomState: randomState,
                modelState: output.state
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
        randomState initialRandomState: MLXRandom.RandomState? = nil,
        modelState: LMOutput.State? = nil
    ) throws {
        precondition(!rowUIDs.contains(uid), "duplicate batch uid '\(uid)'")
        precondition(!rowCache.isEmpty, "continuous batching requires a non-empty KV cache")

        // Checked before anything is mutated: a family that anchors its RoPE in
        // model state cannot decode a row whose anchor never arrived, and
        // failing here beats the model's own precondition trapping the process
        // mid-batch (`docs/KNOWN-FAILURES.md` §1d).
        let incomingDelta = BatchPositionalState.delta(in: modelState)
        if let positionalKey, incomingDelta == nil {
            throw BatchGeneratorError.missingPositionalState(key: positionalKey.id, uid: uid)
        }

        switch cacheStorage {
        case .empty:
            cacheStorage = .singleton(rowCache)
            currentTokens = lastToken.reshaped([1])
        case .singleton(let existingCache):
            guard existingCache.count == rowCache.count else {
                throw BatchGeneratorError.insertedCacheLayerCountChanged(
                    expected: existingCache.count,
                    actual: rowCache.count
                )
            }
            let batchCache = try existingCache.map { try BatchLayerCache.adoptSingle($0) }
            let rowLayers = try rowCache.map { try BatchLayerCache.merge([$0]) }
            cacheStorage = .batched(try zip(batchCache, rowLayers).map { currentLayer, rowLayer in
                let mergedLayer = currentLayer.copyLayer()
                try mergedLayer.extend(rowLayer)
                return mergedLayer
            })
            currentTokens = concatenated([currentTokens, lastToken.reshaped([1])], axis: 0)
        case .batched(let cache):
            guard cache.count == rowCache.count else {
                throw BatchGeneratorError.insertedCacheLayerCountChanged(
                    expected: cache.count,
                    actual: rowCache.count
                )
            }
            let rowLayers = try rowCache.map { try BatchLayerCache.merge([$0]) }
            cacheStorage = .batched(try zip(cache, rowLayers).map { currentLayer, rowLayer in
                let mergedLayer = currentLayer.copyLayer()
                try mergedLayer.extend(rowLayer)
                return mergedLayer
            })
            currentTokens = concatenated([currentTokens, lastToken.reshaped([1])], axis: 0)
        }

        rowUIDs.append(uid)
        // Admission already returned this row's `lastToken` to the caller, so
        // it joins the batch with nothing pending.
        pendingEmission.append(false)
        samplers.append(sampling)
        randomStates.append(initialRandomState ?? Self.randomState(for: sampling.seed))
        let matcher = sampling.jsonGrammar?.makeMatcher()
        let regexMatcher = sampling.regexGrammar?.makeMatcher()
        let gbnfMatcher = sampling.gbnfGrammar?.makeMatcher()
        var toolGrammarState = sampling.toolGrammar.map { QwenXMLToolGrammarState(configuration: $0) }
        var thinkingBudgetState = initialThinkingBudgetState
            ?? sampling.thinkingBudget.map(ThinkingBudgetState.init(configuration:))
        for token in generatedTokens {
            matcher?.advance(tokenID: token)
            regexMatcher?.advance(tokenID: token)
            gbnfMatcher?.advance(tokenID: token)
            toolGrammarState?.observeGeneratedToken(token)
            if initialThinkingBudgetState == nil {
                thinkingBudgetState?.advance(tokenID: token)
            }
        }
        jsonGrammarMatchers.append(matcher)
        regexGrammarMatchers.append(regexMatcher)
        gbnfGrammarMatchers.append(gbnfMatcher)
        toolGrammarStates.append(toolGrammarState)
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
        // Models with per-call positional state (Qwen3.5 M-RoPE ropeDeltas)
        // refuse to continue a warm cache without it — the anchor captured
        // during THIS row's prefill must survive into decode. A single scalar
        // cannot represent two rows, so it is kept per row and reassembled into
        // a vector each step by ``BatchPositionalState``; `state` is then only
        // the carrier for families that have no anchor of their own.
        if let incomingDelta {
            positionalKey = incomingDelta.key
            rowPrefillDeltas.append(incomingDelta.delta)
            state = nil
        } else if rowUIDs.count == 1 {
            state = modelState
        } else {
            state = nil
        }
        eval(currentTokens, cacheStorage.cacheForModel)
    }

    public func remove(uid: String) {
        guard let row = rowUIDs.firstIndex(of: uid) else { return }
        let keptRows = rowUIDs.indices.filter { $0 != row }
        filter(keeping: keptRows)
    }

    public func extractCache(uid: String) -> [any KVCache]? {
        guard let row = rowUIDs.firstIndex(of: uid) else { return nil }

        let extracted: [any KVCache]
        switch cacheStorage {
        case .empty:
            return nil
        case .singleton(let cache):
            guard row == 0 else { return nil }
            extracted = cache.map { $0.copy() }
        case .batched(let cache):
            extracted = cache.map { $0.extract(row) }
        }

        if row < pendingEmission.count, pendingEmission[row] {
            for layer in extracted {
                _ = layer.trim(1)
            }
        }
        return extracted
    }

    public func filter(keeping rows: [Int]) {
        guard !rows.isEmpty else {
            cacheStorage = .empty
            rowUIDs.removeAll()
            pendingEmission.removeAll()
            samplers.removeAll()
            randomStates.removeAll()
            jsonGrammarMatchers.removeAll()
            regexGrammarMatchers.removeAll()
            gbnfGrammarMatchers.removeAll()
            toolGrammarStates.removeAll()
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
            rowPrefillDeltas.removeAll()
            currentTokens = MLXArray([Int32]())
            state = nil
            return
        }

        switch cacheStorage {
        case .empty:
            break
        case .singleton:
            guard rows == [0] else {
                cacheStorage = .empty
                break
            }
        case .batched(let cache):
            for layer in cache {
                layer.filter(keeping: rows)
            }
        }
        let rowIndices = MLXArray(rows.map(Int32.init))
        currentTokens = currentTokens.take(rowIndices, axis: 0)
        rowUIDs = rows.map { rowUIDs[$0] }
        pendingEmission = rows.map { pendingEmission[$0] }
        samplers = rows.map { samplers[$0] }
        randomStates = rows.map { randomStates[$0] }
        jsonGrammarMatchers = rows.map { jsonGrammarMatchers[$0] }
        regexGrammarMatchers = rows.map { regexGrammarMatchers[$0] }
        gbnfGrammarMatchers = rows.map { gbnfGrammarMatchers[$0] }
        toolGrammarStates = rows.map { toolGrammarStates[$0] }
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
        if !rowPrefillDeltas.isEmpty {
            rowPrefillDeltas = rows.map { rowPrefillDeltas[$0] }
        }
        state = nil
        eval(currentTokens, cacheStorage.cacheForModel)
    }

    /// Returns the responses for every row that materialized a new token in
    /// THIS call — keyed by uid, and NOT necessarily one per active row.
    /// Coverage is step-dependent by design: a row inserted while a
    /// launched-ahead step is outstanding has nothing pending (its
    /// `pendingEmission` starts false; admission already delivered any first
    /// token via `insert`'s return value) and joins the step this call
    /// launches, so its next token arrives one call later. A speculative step
    /// can conversely return several tokens for a single row. The only caller
    /// that matters, `Scheduler.step()`, consumes per-uid and assumes nothing
    /// about per-call coverage; tests must not assert
    /// `next().count == count` after mutating the batch mid-flight.
    ///
    /// Discarding the launched-ahead step on insert/remove instead would NOT
    /// restore this cadence soundly: the launch-ahead gate
    /// (`canPipelineDecode`) permits seeded RNG rows (whose streams the
    /// launched sample already advanced) and cache layers outside
    /// `discardOneStepIsExact` (a rotated `RotatingKVCache` ring and any
    /// width-≥2 `BatchKVCache`), where a one-step `trim` cannot reproduce the
    /// pre-step state.
    public func next() -> [Response] {
        guard !rowUIDs.isEmpty else { return [] }
        if let speculativeResponses = speculativeNext() {
            return speculativeResponses
        }

        return decodeOneToken()
    }

    /// Rows whose entry in `currentTokens` was computed by a launched-ahead
    /// step and has not been returned to the caller yet. While any entry is
    /// true, the row caches have already consumed that row's previous token,
    /// so `extractCache` trims the extra position to keep scheduler-visible
    /// snapshots identical to the synchronous path.
    private var pendingEmission: [Bool] = []

    /// Launch-ahead is only exact when sampling needs no CPU-side state that
    /// could lag the launched step: grammar masks and thinking budgets prepare
    /// between steps, and speculation manages `currentTokens` itself. Token
    /// histories are safe — they are updated before the next step is sampled.
    /// Kill switch for launch-ahead, so its effect can be MEASURED rather than
    /// argued. `MLXCAT_DECODE_PIPELINING=0` runs decode fully synchronously.
    ///
    /// It exists because of a control-matrix hole: batched gemma at width >= 2
    /// is the only configuration this repo has ever run that combines a
    /// rotating-merged `BatchKVCache` with this path, and it is also the only
    /// one that diverges from serial (`docs/KNOWN-FAILURES.md` §1f). Every clean
    /// control misses one half of that pair — `qwen3_5` at width 8 is pipelined
    /// but has no rotating layers, gpt-oss has rotating layers but runs through
    /// the synchronous `StaticBatchGenerator`, and gemma at width 1 never builds
    /// a `BatchKVCache` at all. Turning this off is how that pair gets split.
    static let pipeliningEnabled =
        ProcessInfo.processInfo.environment["MLXCAT_DECODE_PIPELINING"] != "0"

    private var canPipelineDecode: Bool {
        Self.pipeliningEnabled
            && !speculativeDecoding.enabled
            && jsonGrammarMatchers.allSatisfy { $0 == nil }
            && regexGrammarMatchers.allSatisfy { $0 == nil }
            && gbnfGrammarMatchers.allSatisfy { $0 == nil }
            && toolGrammarStates.allSatisfy { $0?.requiresScalarSampling != true }
            && thinkingBudgetStates.allSatisfy { $0 == nil }
    }

    private func decodeOneToken() -> [Response] {
        if pendingEmission.contains(true) {
            return emitPendingStep()
        }

        let (tokenIds, responses) = advanceStepSynchronously()
        launchStepAheadIfSafe(latestTokenIds: tokenIds)
        return responses
    }

    /// Build one decode step's graph: feed `currentTokens`, thread state, and
    /// sample the next tokens. Purely lazy — callers decide how to evaluate.
    /// The logits are returned so callers can keep marking them as an
    /// evaluation output: that materialization boundary blocks kernel fusion
    /// through the sampling ops, and removing it changes bf16 rounding enough
    /// to flip greedy argmax at near-ties — breaking token-exactness against
    /// the pre-pipelining engine.
    ///
    /// The cache and model state are parameters (not `self` fields) so a
    /// caller can build against shallow cache copies and adopt or discard the
    /// step afterwards — see `prebuildNextStepIfSafe`.
    private func computeNextTokens(
        cacheForModel: [any KVCache],
        state stepState: LMOutput.State?
    ) -> (tokens: MLXArray, logits: MLXArray, state: LMOutput.State?) {
        let output = model(
            LMInput.Text(tokens: currentTokens[0..., .newAxis]),
            cache: cacheForModel,
            state: stepState
        )
        let logits = output.logits[0..., -1, 0...]
        return (batchedSample(from: logits) ?? sampleRows(from: logits), logits, output.state)
    }

    private func advanceStepSynchronously() -> ([Int], [Response]) {
        // The per-step autoreleasepool mirrors the upstream TokenIterator: a
        // forward pass produces hundreds of autoreleased MLX wrappers and this
        // loop never reaches a pool boundary on its own.
        let (nextTokens, logits, nextState) = autoreleasepool {
            computeNextTokens(cacheForModel: cacheStorage.cacheForModel, state: stepState)
        }
        state = nextState
        asyncEval(nextTokens, logits)
        currentTokens = nextTokens
        pendingEmission = Array(repeating: false, count: rowUIDs.count)

        let tokenIds = nextTokens.asArray(Int.self)
        for row in generatedTokenHistory.indices {
            recordGeneratedToken(tokenIds[row], row: row)
        }
        let responses = rowUIDs.enumerated().map { row, uid in
            Response(uid: uid, token: tokenIds[row])
        }
        return (tokenIds, responses)
    }

    /// Steady-state pipelined step: the tokens were computed by the previous
    /// call's launch-ahead, so this usually returns without waiting on the GPU.
    ///
    /// The next step's graph is BUILT before waiting on the in-flight step —
    /// graph construction is per-layer CPU work that would otherwise happen
    /// while the GPU sits idle after draining the in-flight step (the ordering
    /// the upstream `TokenIterator` gets for free by never blocking before its
    /// `step()` call). The build targets shallow cache copies, so nothing is
    /// adopted until the EOS/limit check passes; a discarded build costs one
    /// graph construction per finished request.
    private func emitPendingStep() -> [Response] {
        let t0 = Self.phaseTimingEnabled ? DispatchTime.now().uptimeNanoseconds : 0
        let prebuilt = prebuildNextStepIfSafe()
        let t1 = Self.phaseTimingEnabled ? DispatchTime.now().uptimeNanoseconds : 0
        let tokenIds = currentTokens.asArray(Int.self)
        let t2 = Self.phaseTimingEnabled ? DispatchTime.now().uptimeNanoseconds : 0
        var responses: [Response] = []
        responses.reserveCapacity(rowUIDs.count)
        for row in rowUIDs.indices where pendingEmission[row] {
            recordGeneratedToken(tokenIds[row], row: row)
            responses.append(Response(uid: rowUIDs[row], token: tokenIds[row]))
        }
        pendingEmission = Array(repeating: false, count: rowUIDs.count)
        if let prebuilt {
            if launchIsSafe(latestTokenIds: tokenIds) {
                let t3 = Self.phaseTimingEnabled ? DispatchTime.now().uptimeNanoseconds : 0
                asyncEval(prebuilt.tokens, prebuilt.logits)
                if Self.phaseTimingEnabled {
                    let t4 = DispatchTime.now().uptimeNanoseconds
                    phaseBuildNs += t1 - t0
                    phaseWaitNs += t2 - t1
                    phaseEmitNs += t3 - t2
                    phaseLaunchNs += t4 - t3
                    phaseSteps += 1
                }
                state = prebuilt.state
                currentTokens = prebuilt.tokens
                pendingEmission = Array(repeating: true, count: rowUIDs.count)
            } else {
                for kvCache in cacheStorage.cacheForModel {
                    _ = kvCache.trim(1)
                }
            }
            return responses
        }
        launchStepAheadIfSafe(latestTokenIds: tokenIds)
        return responses
    }

    private struct PrebuiltStep {
        let tokens: MLXArray
        let logits: MLXArray
        let state: LMOutput.State?
    }

    /// Sampling that reads per-row history or RNG state cannot be built one
    /// token early: penalties/minTokens would see a history missing the
    /// pending token, and a discarded build would advance seeded RNG streams.
    /// Temperature-0 and unseeded filter-free sampling are history-free.
    /// Restricted to cache layers whose one-step discard (`trim(1)`) is exact:
    /// `KVCacheSimple` always, `RotatingKVCache` only while the built step
    /// stays in the pre-rotation growth phase — a rotated ring overwrites its
    /// oldest slot, which a trim cannot restore. Width ≥2 batches merge into
    /// `BatchKVCache`/batch-state layouts and fall out of the gate naturally.
    private var canPrebuildNextStep: Bool {
        canPipelineDecode
            && randomStates.allSatisfy { $0 == nil }
            && samplers.allSatisfy { sampler in
                !sampler.hasSamplingFiltersOrPenalties
                    && sampler.minTokens == 0
                    && sampler.allowedSequences == nil
            }
            && cacheStorage.cacheForModel.allSatisfy { Self.discardOneStepIsExact($0) }
    }

    private static func discardOneStepIsExact(_ kvCache: any KVCache) -> Bool {
        if let rotating = kvCache as? RotatingKVCache {
            guard let maxSize = rotating.maxSize else { return false }
            return rotating.offset + 1 < maxSize
        }
        return kvCache is KVCacheSimple
    }

    /// Build the next decode step against the LIVE cache handles while the
    /// launched-ahead step is still on the GPU, so the graph-construction CPU
    /// work overlaps the in-flight compute instead of serializing after it —
    /// the ordering the upstream `TokenIterator` gets by never blocking before
    /// its `step()` call. Building on the live handles keeps MLX buffer
    /// donation intact (a shallow cache copy would pin a second reference to
    /// every KV buffer and force copy-on-write in the slice assignment).
    /// The pending `currentTokens` feed the graph as lazy handles — no
    /// materialization is needed to build.
    private func prebuildNextStepIfSafe() -> PrebuiltStep? {
        guard canPrebuildNextStep, !rowUIDs.isEmpty else { return nil }
        for row in rowUIDs.indices {
            if let limit = maxGeneratedTokens[row],
                generatedTokenHistory[row].count + 1 >= limit
            {
                return nil
            }
        }
        let (tokens, logits, nextState) = autoreleasepool {
            computeNextTokens(cacheForModel: cacheStorage.cacheForModel, state: stepState)
        }
        return PrebuiltStep(tokens: tokens, logits: logits, state: nextState)
    }

    private func launchIsSafe(latestTokenIds: [Int]) -> Bool {
        for row in rowUIDs.indices {
            if samplers[row].eosTokenIds.contains(latestTokenIds[row]) {
                return false
            }
            if let limit = maxGeneratedTokens[row], generatedTokenHistory[row].count >= limit {
                return false
            }
        }
        return true
    }

    /// Start the next decode step before returning the current one, so the GPU
    /// computes it while the caller streams tokens. Skipped when any row just
    /// finished: feeding a finished row's stop token would advance the caches
    /// past what the scheduler is about to snapshot and remove.
    private func launchStepAheadIfSafe(latestTokenIds: [Int]) {
        guard canPipelineDecode, !rowUIDs.isEmpty, launchIsSafe(latestTokenIds: latestTokenIds) else {
            return
        }

        let (nextTokens, logits, nextState) = autoreleasepool {
            computeNextTokens(cacheForModel: cacheStorage.cacheForModel, state: stepState)
        }
        state = nextState
        asyncEval(nextTokens, logits)
        currentTokens = nextTokens
        pendingEmission = Array(repeating: true, count: rowUIDs.count)
    }

    private func batchedSample(from logits: MLXArray) -> MLXArray? {
        guard jsonGrammarMatchers.allSatisfy({ $0 == nil }),
            regexGrammarMatchers.allSatisfy({ $0 == nil }),
            gbnfGrammarMatchers.allSatisfy({ $0 == nil }),
            toolGrammarStates.allSatisfy({ $0?.requiresScalarSampling != true }),
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
                toolGrammarMatcher: toolGrammarStates[row]?.activeMatcher,
                postToolAllowedTokenIDs: toolGrammarStates[row]?.activeMatcher == nil
                    ? toolGrammarStates[row]?.activeAllowedTokenIDs
                    : nil,
                randomState: randomStates[row],
                thinkingBudgetState: &thinkingBudgetStates[row],
                precomputedGrammarMasks: PrecomputedGrammarMasks(
                    jsonAllowedTokenIDs: jsonGrammarMasks[row]?.readyTokenIDs,
                    regexAllowedTokenIDs: regexGrammarMasks[row]?.readyTokenIDs,
                    gbnfAllowedTokenIDs: gbnfGrammarMasks[row]?.readyTokenIDs,
                    toolAllowedTokenIDs: toolGrammarStates[row]?.activeMatcher == nil
                        ? nil
                        : toolGrammarStates[row]?.activeAllowedTokenIDs
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
        let workingCacheForModel = cacheStorage.cacheForModel.map { $0.copy() }
        let output = model(
            LMInput.Text(tokens: verificationInput),
            cache: workingCacheForModel,
            state: stepState
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

        cacheStorage = .singleton(workingCacheForModel)
        state = output.state
        let nextToken = acceptedTokens.last ?? proposedTokens.last ?? currentTokens.item(Int.self)
        currentTokens = MLXArray([Int32(nextToken)])
        asyncEval(currentTokens)
        eval(output.logits, workingCacheForModel)

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
            samplers[row].toolGrammar == nil,
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
        toolGrammarStates[row]?.observeGeneratedToken(tokenID)
        thinkingBudgetStates[row]?.advance(tokenID: tokenID)
    }

    private func prefillWithLastTokenWithheld(
        _ text: LMInput.Text,
        cache rowCache: [any KVCache]
    ) throws -> (lastToken: MLXArray, modelState: LMOutput.State?) {
        let tokens = text.tokens
        let tokenCount = tokens.dim(0)
        guard tokenCount > 1 else {
            throw BatchGeneratorError.promptTooShortForExternalPrefill
        }

        let prefixTokens = tokens[..<(tokenCount - 1)]
        let prefixInput = LMInput.Text(tokens: prefixTokens)
        let output = model(prefixInput[text: .newAxis], cache: rowCache, state: nil)
        eval(rowCache)

        return (tokens[tokenCount - 1], output.state)
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
    case missingPositionalState(key: String, uid: String)
}
