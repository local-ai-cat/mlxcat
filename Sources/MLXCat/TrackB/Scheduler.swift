import Foundation
import MLX
import MLXLMCommon

public final class LanguageModelBox: @unchecked Sendable {
    let model: any LanguageModel

    public init(_ model: any LanguageModel) {
        self.model = model
    }
}

public actor Scheduler {
    public struct PressureSnapshot: Sendable, Equatable {
        public let runningUIDs: [String]
        public let waitingCount: Int
        public let admissionInProgressUID: String?

        public init(runningUIDs: [String], waitingCount: Int, admissionInProgressUID: String?) {
            self.runningUIDs = runningUIDs
            self.waitingCount = waitingCount
            self.admissionInProgressUID = admissionInProgressUID
        }
    }

    public struct PressurePolicy: Sendable {
        public let shouldPreempt: @Sendable (PressureSnapshot) -> Bool

        public init(shouldPreempt: @escaping @Sendable (PressureSnapshot) -> Bool) {
            self.shouldPreempt = shouldPreempt
        }

        public static let disabled = PressurePolicy { _ in false }
    }

    private let model: any LanguageModel
    private let parameters: GenerateParameters
    private let generator: ContinuousBatchGenerator
    private let maxConcurrentRequests: Int
    private let queueLimit: Int
    private let prefixStore: (any PrefixKVStore)?
    private let prefixCacheEnabled: Bool
    private let serializationPolicy: SerializationPolicy
    private let schedulerManagedTextPrefill: Bool
    private let chunkIdlePrefill: Bool
    private let pressurePolicy: PressurePolicy
    private var waiting: [Request] = []
    private var running: [String: RunningRequest] = [:]
    private var admissionInProgress: AdmissionInProgress?
    private var resumeGeneratedTokens: [String: [Int]] = [:]
    private var droppedStaleResponseCount = 0
    private var pendingCancellation: Set<String> = []
    private let collector = OutputCollector()

    public init(
        modelBox: LanguageModelBox,
        parameters: GenerateParameters,
        maxConcurrentRequests: Int,
        prefixStore: (any PrefixKVStore)? = nil,
        cacheCapabilities: ModelCacheCapabilities = .default,
        serializedDecode: Bool = false,
        serializationPolicy: SerializationPolicy? = nil,
        schedulerManagedTextPrefill: Bool = true,
        chunkIdlePrefill: Bool = true,
        pressurePolicy: PressurePolicy = .disabled,
        speculativeDecoding: SpeculativeDecodingConfiguration = SpeculativeDecodingConfiguration()
    ) {
        self.model = modelBox.model
        self.parameters = parameters
        self.generator = ContinuousBatchGenerator(
            model: modelBox.model,
            parameters: parameters,
            speculativeDecoding: speculativeDecoding
        )
        self.maxConcurrentRequests = maxConcurrentRequests
        self.queueLimit = max(maxConcurrentRequests * 4, 32)
        self.prefixStore = prefixStore
        self.prefixCacheEnabled = prefixStore != nil && !cacheCapabilities.usesWindowedKVCache
        self.serializationPolicy = serializationPolicy ?? (serializedDecode ? .always : .never)
        self.schedulerManagedTextPrefill = schedulerManagedTextPrefill
        self.chunkIdlePrefill = chunkIdlePrefill
        self.pressurePolicy = pressurePolicy
    }

    public var isIdle: Bool {
        waiting.isEmpty && running.isEmpty && generator.isEmpty && admissionInProgress == nil
    }

    public var queueDepth: Int {
        waiting.count + running.count + (admissionInProgress == nil ? 0 : 1)
    }

    public func submit(_ request: Request) throws {
        if queueDepth >= queueLimit {
            throw SchedulerError.queueFull(retryAfterSteps: max(1, queueDepth / max(1, maxConcurrentRequests)))
        }
        if waiting.contains(where: { $0.uid == request.uid })
            || running[request.uid] != nil
            || admissionInProgress?.request.uid == request.uid
        {
            throw SchedulerError.duplicateRequest(request.uid)
        }
        waiting.append(request)
    }

    public func cancel(uid: String) {
        if let waitingIndex = waiting.firstIndex(where: { $0.uid == uid }) {
            waiting.remove(at: waitingIndex)
            resumeGeneratedTokens.removeValue(forKey: uid)
            collector.record(Response(uid: uid, token: -1, finishReason: .cancelled))
            return
        }
        if let admission = admissionInProgress, admission.request.uid == uid {
            if let hit = admission.prefixHit {
                prefixStore?.release(hit)
            }
            admissionInProgress = nil
            resumeGeneratedTokens.removeValue(forKey: uid)
            collector.record(Response(uid: uid, token: -1, finishReason: .cancelled))
            return
        }
        if running[uid] != nil {
            pendingCancellation.insert(uid)
        }
    }

    public func step() throws -> [Response] {
        var processed = applyPendingCancellation()
        let generatorWasEmpty = generator.isEmpty
        let admitted = admitWaiting(allowPartialPrefill: !generatorWasEmpty)
        processed.append(contentsOf: admitted)

        guard !generator.isEmpty else { return processed }
        if generatorWasEmpty, admitted.contains(where: { $0.token >= 0 }) {
            return processed
        }

        if pressurePolicy.shouldPreempt(pressureSnapshot()),
            preemptYoungestResumableRequest()
        {
            return processed
        }

        let rawResponses = generator.next()
        var finishedUIDs: [String] = []
        var touchedUIDs: Set<String> = []

        for response in rawResponses {
            guard let runningRequest = running[response.uid] else {
                droppedStaleResponseCount += 1
                continue
            }
            runningRequest.generatedTokens.append(response.token)
            touchedUIDs.insert(response.uid)

            let finishReason: FinishReason?
            if runningRequest.request.eosTokenIds.contains(response.token) {
                finishReason = .stop
            } else if runningRequest.generatedTokenCount >= runningRequest.request.maxTokens {
                finishReason = .length
            } else {
                finishReason = nil
            }

            let processedResponse = Response(
                uid: response.uid,
                token: response.token,
                finishReason: finishReason,
                logprobs: response.logprobs
            )
            processed.append(processedResponse)
            collector.record(processedResponse)

            if finishReason != nil {
                finishedUIDs.append(response.uid)
            }
        }

        for uid in touchedUIDs where !finishedUIDs.contains(uid) {
            if let runningRequest = running[uid] {
                publishAvailablePrefixBlocks(
                    uid: uid,
                    runningRequest: runningRequest,
                    minimumNewTokens: Self.midGenerationPublishInterval
                )
            }
        }

        if !finishedUIDs.isEmpty {
            Stream.gpu.synchronize()
            for uid in finishedUIDs {
                if let finishedRequest = running[uid] {
                    publishAvailablePrefixBlocks(uid: uid, runningRequest: finishedRequest)
                }
                let finishedRequest = running[uid]
                generator.remove(uid: uid)
                if let hit = finishedRequest?.prefixHit {
                    prefixStore?.release(hit)
                }
                running.removeValue(forKey: uid)
                resumeGeneratedTokens.removeValue(forKey: uid)
            }
        }

        if generator.isEmpty {
            processed.append(contentsOf: admitWaiting(allowPartialPrefill: false))
        }
        return processed
    }

    public func collectedTokens() -> [String: [Int]] {
        collector.allTokens
    }

    public func responses(for uid: String) -> [Response] {
        collector.responses(for: uid)
    }

    public func consumeTokens(for uid: String) -> [Int] {
        collector.consumeTokens(for: uid)
    }

    public func tokens(for uid: String) -> [Int] {
        collector.tokens(for: uid)
    }

    public func discardResponses(for uid: String) {
        collector.remove(uid: uid)
    }

    public var droppedStaleResponses: Int {
        droppedStaleResponseCount
    }

    public var speculativeDecodingStats: SpeculativeDecodingStats {
        generator.speculationStats
    }

    private func admitWaiting(allowPartialPrefill: Bool) -> [Response] {
        var admittedResponses: [Response] = []
        while running.count < maxConcurrentRequests {
            // Some model architectures still derive RoPE position ids or
            // shared-KV offsets from scalar cache.offset, so mixed-offset rows
            // must not share a decode batch.
            //
            // `.always` is the blunt version of that and it is expensive: on
            // gemma-4-E2B it costs 50x TTFT and 2.17x aggregate throughput at c4
            // (`docs/COMPETITIVE.md`). `.multimodalOnly` is the same protection
            // aimed at the rows that actually need it — a request carrying
            // images, video or audio has a token count that varies per row, and
            // that is what the scalar offset gets wrong. Text rows of these same
            // families batch correctly: they pass logit-level invariance more
            // cleanly than the model the gate was pinned to, and pass
            // batched-vs-serial token equality. Ragged IMAGE rows do not — all
            // three rows stop at the same early token (measured 2026-08-23).
            let busy = !running.isEmpty || !generator.isEmpty || !admittedResponses.isEmpty
            // Computed from `running`, not tracked alongside it: RunningRequest
            // retains its Request, so there is no second copy of this state to
            // drift out of sync on a cancel or an error path.
            if busy, serializationPolicy.refusesToJoin(running: running.values.map(\.request)) {
                return admittedResponses
            }
            if busy, let next = waiting.first, serializationPolicy.requiresSolitude(next) {
                // A row that needs solitude waits for the batch to drain rather
                // than joining it. Head-of-line blocking is deliberate: admitting
                // it out of order would starve it behind an unbounded text stream.
                return admittedResponses
            }

            if admissionInProgress == nil {
                guard !waiting.isEmpty else { return admittedResponses }
                let request = waiting.removeFirst()
                do {
                    var sampling = request.sampling
                    sampling.eosTokenIds.formUnion(request.eosTokenIds)
                    if !request.eosTokenIds.isEmpty {
                        sampling.xtcSpecialTokens = Array(Set(sampling.xtcSpecialTokens).union(request.eosTokenIds))
                    }

                    switch try prepareForInsert(request, sampling: sampling) {
                    case .ready(let row):
                        if let response = try completeAdmission(
                            row,
                            request: request,
                            sampling: sampling
                        ) {
                            admittedResponses.append(response)
                        }
                        continue
                    case .pending(let admission):
                        admissionInProgress = admission
                    }
                } catch {
                    let response = Response(
                        uid: request.uid,
                        token: -1,
                        finishReason: .failed(String(describing: error))
                    )
                    collector.record(response)
                    admittedResponses.append(response)
                    continue
                }
            }

            do {
                guard let row = try advanceAdmission(allowPartialPrefill: allowPartialPrefill) else {
                    return admittedResponses
                }
                guard let admission = admissionInProgress else { return admittedResponses }
                if let response = try completeAdmission(
                    row,
                    request: admission.request,
                    sampling: admission.sampling
                ) {
                    admittedResponses.append(response)
                }
                admissionInProgress = nil
            } catch {
                let request = admissionInProgress?.request
                if let hit = admissionInProgress?.prefixHit {
                    prefixStore?.release(hit)
                }
                admissionInProgress = nil
                if let request {
                    let response = Response(
                        uid: request.uid,
                        token: -1,
                        finishReason: .failed(String(describing: error))
                    )
                    collector.record(response)
                    admittedResponses.append(response)
                }
                continue
            }
        }
        return admittedResponses
    }

    private func applyPendingCancellation() -> [Response] {
        guard !pendingCancellation.isEmpty else { return [] }

        Stream.gpu.synchronize()
        var responses: [Response] = []
        for uid in pendingCancellation {
            guard let runningRequest = running[uid] else { continue }
            if let hit = runningRequest.prefixHit {
                prefixStore?.release(hit)
            }
            generator.remove(uid: uid)
            running.removeValue(forKey: uid)
            resumeGeneratedTokens.removeValue(forKey: uid)
            let response = Response(uid: uid, token: -1, finishReason: .cancelled)
            collector.record(response)
            responses.append(response)
        }
        pendingCancellation.removeAll()
        return responses
    }

    private func completeAdmission(
        _ row: PreparedBatchRow,
        request: Request,
        sampling: SamplingParameters
    ) throws -> Response? {
        let seededGeneratedTokens = resumeGeneratedTokens.removeValue(forKey: request.uid) ?? []
        let initialTokenID = row.initialGeneratedToken?.tokenID
        let newlyGeneratedTokens = initialTokenID.map { [$0] } ?? []
        let generatedTokenCount = seededGeneratedTokens.count + newlyGeneratedTokens.count
        let finishReason = initialTokenID.map { tokenID in
            self.finishReason(
                token: tokenID,
                generatedTokenCount: generatedTokenCount,
                request: request
            )
        } ?? nil

        if finishReason == nil {
            let generatorSeededTokens = seededGeneratedTokens + newlyGeneratedTokens
            try generator.insert(
                uid: request.uid,
                cache: row.cache,
                lastToken: row.lastToken,
                sampling: sampling,
                generatedTokens: generatorSeededTokens,
                maxGeneratedTokens: request.maxTokens,
                speculativeContextTokens: row.promptTokens + generatorSeededTokens,
                thinkingBudgetState: seededGeneratedTokens.isEmpty
                    ? row.initialGeneratedToken?.thinkingBudgetState
                    : nil,
                modelState: row.modelState
            )
            running[request.uid] = RunningRequest(
                request: request,
                promptTokens: row.promptTokens,
                prefixHit: row.prefixHit,
                generatedTokens: generatorSeededTokens,
                generatedTokensIncludedInPrompt: seededGeneratedTokens.count,
                cachedTokenCount: 0
            )
        } else {
            storeCompletedAdmissionPrefix(row, request: request)
        }

        guard let initialTokenID else { return nil }
        let response = Response(
            uid: request.uid,
            token: initialTokenID,
            finishReason: finishReason,
            cachedPromptTokens: row.prefixHit?.matchedTokenCount ?? 0
        )
        collector.record(response)
        return response
    }

    private func advanceAdmission(allowPartialPrefill: Bool) throws -> PreparedBatchRow? {
        guard var admission = admissionInProgress else { return nil }

        // Two different questions wear one number here, and they want opposite
        // answers, so they get two.
        //
        // IDLE prefill (nothing decoding) is a pure memory question: no stream
        // is waiting on a tick, so a narrow chunk costs only per-chunk overhead
        // and saves attention scratch. Measured at 16k on an M5 Max, widening it
        // from 512 to 2048 cost gemma-4-12B 13.42 -> 15.56 GiB and Qwen3.8-27B
        // 18.90 -> 22.91 GiB. So idle stays narrow.
        //
        // BUSY prefill (something is decoding) is a scheduling question: every
        // chunk boundary costs a decode tick and an actor round trip, so at 512
        // a 16k prompt pays that 32 times. mlx-lm and omlx use 2048
        // (`guest/mlx-lm/mlx_lm/generate.py:1509`,
        // `guest/omlx/omlx/scheduler.py:1304`); mlx-serve uses 8192 with
        // per-model caps (`guest/mlx-serve/src/generate.zig:34`, `:111`). 2048 is
        // the conservative end — what the two engines closest to our stack use.
        //
        // An explicit `prefill.stepSize` overrides both: a caller that names a
        // width means it.
        let idlePrefillStep = 512
        let busyPrefillStep = 2048
        let prefillStep: Int = max(
            1,
            parameters.prefill.stepSize ?? (allowPartialPrefill ? busyPrefillStep : idlePrefillStep)
        )
        let stepBudget: Int = allowPartialPrefill ? prefillStep : Int.max
        var remainingBudget = stepBudget
        // `allowPartialPrefill` decides whether admission YIELDS between chunks
        // so other rows can decode. Whether the forward pass itself spans the
        // whole prompt in one `model()` call is a separate question, and
        // `chunkIdlePrefill` answers it.
        //
        // They used to be the same flag, so an idle admission — every
        // single-stream request, so every c1 cell — ran one call over L
        // positions and materialized a [1, L, vocab] logits tensor plus
        // full-length attention scratch. Chunking that away is worth GiBs on
        // most models and costs GiBs on at least one, which is why the caller
        // decides. Measured at 16k on an M5 Max, after-load -> lifetime max,
        // both arms back to back:
        //
        //     model                  vocab    single-pass   chunked
        //     gemma-4-12B           262,144    34.51         20.74   -13.8
        //     Qwen3.8-27B           248,320    53.02         39.02   -14.0
        //     gpt-oss-20b           201,088    35.17         25.37    -9.8
        //     Qwen3-Coder-30B-A3B   151,936    24.75         40.34   +15.6
        //
        // See `NativeModelLoader.singlePassPrefillModelTypes` for why the last
        // row is an exemption rather than a reason to abandon the change.
        let chunked = allowPartialPrefill || chunkIdlePrefill
        while admission.nextPrefillIndex < admission.prefillRange.upperBound, remainingBudget > 0 {
            let end: Int
            if chunked {
                end = min(
                    admission.nextPrefillIndex + min(prefillStep, remainingBudget),
                    admission.prefillRange.upperBound
                )
            } else {
                end = admission.prefillRange.upperBound
            }
            let input = LMInput.Text(
                tokens: admission.promptTokensArray[admission.nextPrefillIndex ..< end]
            )
            let output = model(input[text: .newAxis], cache: admission.cache, state: admission.state)
            admission.state = output.state
            if end == admission.prefillRange.upperBound {
                admission.initialGeneratedToken = sampledToken(
                    from: output.logits,
                    sampling: admission.sampling
                )
            }
            asyncEval(admission.cache)
            remainingBudget -= end - admission.nextPrefillIndex
            admission.nextPrefillIndex = end
            // Copied from mlx-lm's `_prefill` (guest/mlx-lm/mlx_lm/generate.py:579),
            // which calls `mx.clear_cache()` at the end of every chunk. Each
            // chunk attends over a longer key range than the last, so no two
            // chunks allocate the same attention-scratch shape and MLX's buffer
            // cache can never reuse one — it just accumulates every shape for
            // the whole prefill. Freeing between chunks is what makes chunked
            // prefill cheap for them and was the difference from us.
            //
            // Only between chunks, and only when chunking: the final chunk's
            // buffers are about to be reused by decode, and the single-pass path
            // allocates nothing worth freeing mid-loop.
            if chunked, admission.nextPrefillIndex < admission.prefillRange.upperBound {
                Memory.clearCache()
            }
        }

        if admission.nextPrefillIndex < admission.prefillRange.upperBound {
            admissionInProgress = admission
            return nil
        }

        eval(admission.cache)
        guard let initialGeneratedToken = admission.initialGeneratedToken else {
            preconditionFailure("completed prompt prefill without an initial generated token")
        }
        return PreparedBatchRow(
            cache: admission.cache,
            lastToken: initialGeneratedToken.token,
            promptTokens: admission.storedPromptTokens,
            prefixHit: admission.prefixHit,
            initialGeneratedToken: initialGeneratedToken,
            modelState: admission.state
        )
    }

    private func prepareForInsert(_ request: Request, sampling: SamplingParameters) throws
        -> PreparedAdmission
    {
        let rowCache = try model.newCache(parameters: parameters)
        let prefixCacheEligible = isPrefixCacheEligible(request.input)
        let canUseRawTextTokens = schedulerManagedTextPrefill
            && prefixCacheEligible
            && request.input.text.tokens.ndim == 1
            && request.input.text.mask == nil
        let promptText: LMInput.Text
        if canUseRawTextTokens {
            promptText = request.input.text
        } else {
            // state: nil — `rowCache` was just created above, so there is no
            // prior model state to carry into the prefill.
            switch try model.prepare(
                request.input, cache: rowCache, state: nil,
                prefill: parameters.prefill) {
            case .tokens(let tokens):
                promptText = tokens
            case .logits(let output):
                let firstToken = sampledToken(from: output.logits, sampling: sampling)
                return .ready(PreparedBatchRow(
                    cache: rowCache,
                    lastToken: firstToken.token,
                    promptTokens: [],
                    prefixHit: nil,
                    initialGeneratedToken: firstToken,
                    modelState: output.state
                ))
            }
        }

        let promptTokensArray = promptText.tokens
        let promptTokenCount = promptTokensArray.dim(0)
        guard promptTokenCount > 0 else {
            throw BatchGeneratorError.promptTooShortForExternalPrefill
        }
        let promptTokens = promptTokensArray.asArray(Int.self)

        if promptTokenCount == 1 {
            let output = model(promptText[text: .newAxis], cache: rowCache, state: nil)
            eval(rowCache)
            let firstToken = sampledToken(from: output.logits, sampling: sampling)
            return .ready(PreparedBatchRow(
                cache: rowCache,
                lastToken: firstToken.token,
                promptTokens: [],
                prefixHit: nil,
                initialGeneratedToken: firstToken,
                modelState: output.state
            ))
        }

        if prefixCacheEnabled,
            prefixCacheEligible,
            let prefixStore,
            let hit = prefixStore.fetch(tokens: promptTokens, sessionKey: request.cacheSession)
        {
            if hit.matchedTokenCount == promptTokens.count {
                prefixStore.release(hit)
                return prefillMissRow(
                    request: request,
                    sampling: sampling,
                    promptTokens: promptTokens,
                    promptTokensArray: promptTokensArray,
                    storedPromptTokens: promptTokens,
                    rowCache: rowCache
                )
            }
            do {
                let serialized = try prefixStore.reconstructCache(from: hit)
                let reconstructedCache = try serialized.map {
                    try BlockAwarePrefixKVStore.cache(from: $0)
                }
                return try prepareHitRow(
                    request: request,
                    sampling: sampling,
                    promptTokens: promptTokens,
                    promptTokensArray: promptTokensArray,
                    hit: hit,
                    reconstructedCache: reconstructedCache
                )
            } catch {
                prefixStore.release(hit)
                logCacheFailure("prefix hit reconstruction failed; falling back to cache miss", error)
            }
        }

        return prefillMissRow(
            request: request,
            sampling: sampling,
            promptTokens: promptTokens,
            promptTokensArray: promptTokensArray,
            storedPromptTokens: prefixCacheEligible ? promptTokens : [],
            rowCache: rowCache
        )
    }

    private func prepareHitRow(
        request: Request,
        sampling: SamplingParameters,
        promptTokens: [Int],
        promptTokensArray: MLXArray,
        hit: PrefixKVStoreHit,
        reconstructedCache: [any KVCache]
    ) throws -> PreparedAdmission {
        let matched = hit.matchedTokenCount
        guard matched <= promptTokens.count else {
            throw SchedulerError.invalidPrefixHit
        }

        return .pending(AdmissionInProgress(
            request: request,
            sampling: sampling,
            cache: reconstructedCache,
            promptTokens: promptTokens,
            promptTokensArray: promptTokensArray,
            storedPromptTokens: promptTokens,
            prefixHit: hit,
            prefillRange: matched ..< promptTokens.count,
            nextPrefillIndex: matched,
            state: nil,
            initialGeneratedToken: nil
        ))
    }

    private func prefillMissRow(
        request: Request,
        sampling: SamplingParameters,
        promptTokens: [Int],
        promptTokensArray: MLXArray,
        storedPromptTokens: [Int],
        rowCache: [any KVCache]
    ) -> PreparedAdmission {
        .pending(AdmissionInProgress(
            request: request,
            sampling: sampling,
            cache: rowCache,
            promptTokens: promptTokens,
            promptTokensArray: promptTokensArray,
            storedPromptTokens: storedPromptTokens,
            prefixHit: nil,
            prefillRange: 0 ..< promptTokens.count,
            nextPrefillIndex: 0,
            state: nil,
            initialGeneratedToken: nil
        ))
    }

    private func cacheSnapshot(uid: String) -> [SerializedKVLayer]? {
        guard let cache = generator.extractCache(uid: uid) else {
            return nil
        }
        return cache.map {
            SerializedKVLayer(
                state: $0.state,
                metaState: $0.metaState,
                className: String(describing: type(of: $0))
            )
        }
    }

    private func storeCompletedAdmissionPrefix(_ row: PreparedBatchRow, request: Request) {
        guard prefixCacheEnabled,
            let prefixStore,
            !row.promptTokens.isEmpty
        else {
            return
        }

        let snapshot = row.cache.map {
            SerializedKVLayer(
                state: $0.state,
                metaState: $0.metaState,
                className: String(describing: type(of: $0))
            )
        }
        do {
            try prefixStore.store(
                tokens: row.promptTokens,
                sessionKey: request.cacheSession,
                cache: snapshot
            )
        } catch {
            logCacheFailure("completed admission prefix cache store failed", error)
        }
    }

    /// Snapshotting a row's KV cache is not free: `extractCache` slices every
    /// layer and the stored copy holds live references to the KV arrays, which
    /// blocks MLX buffer donation on the next in-place cache update. Publishing
    /// on every decode step measured a ~14% single-request throughput loss, so
    /// mid-generation publishes only fire once this many new tokens accumulate.
    /// The first publish (prompt prefix) and the finish/preemption publishes
    /// are exempt: callers pass `minimumNewTokens: 1` there.
    static let midGenerationPublishInterval = 256

    private func publishAvailablePrefixBlocks(
        uid: String,
        runningRequest: RunningRequest,
        minimumNewTokens: Int = 1
    ) {
        guard prefixCacheEnabled,
            let prefixStore,
            !runningRequest.promptTokens.isEmpty
        else {
            return
        }

        // The live KV cache has consumed the token from the previous step. The
        // token sampled by the current step is now `currentTokens`, so it is not
        // part of the cache snapshot until the next decode call.
        let generatedTokensAfterPrompt = runningRequest.generatedTokens
            .dropFirst(runningRequest.generatedTokensIncludedInPrompt)
        let cachedGeneratedTokens = generatedTokensAfterPrompt.dropLast()
        let availableTokens = runningRequest.promptTokens + cachedGeneratedTokens
        let requiredNewTokens = runningRequest.cachedTokenCount == 0 ? 1 : minimumNewTokens
        guard availableTokens.count >= runningRequest.cachedTokenCount + requiredNewTokens,
            let snapshot = cacheSnapshot(uid: uid)
        else {
            return
        }

        do {
            try prefixStore.store(
                tokens: availableTokens,
                sessionKey: runningRequest.request.cacheSession,
                cache: snapshot
            )
            runningRequest.cachedTokenCount = availableTokens.count
        } catch {
            logCacheFailure("prefix cache store failed", error)
        }
    }

    private func preemptYoungestResumableRequest() -> Bool {
        guard let uid = generator.uids.reversed().first(where: { uid in
            guard let request = running[uid] else { return false }
            return canResumeFromTokenPrompt(request)
        }),
            let runningRequest = running[uid]
        else {
            return false
        }

        publishAvailablePrefixBlocks(uid: uid, runningRequest: runningRequest)
        if let hit = runningRequest.prefixHit {
            prefixStore?.release(hit)
        }

        let resumedTokens = currentContextTokens(for: runningRequest)
        let resumedInput = LMInput(tokens: MLXArray(resumedTokens.map(Int32.init)))
        let resumedRequest = Request(
            uid: runningRequest.request.uid,
            input: resumedInput,
            maxTokens: runningRequest.request.maxTokens,
            sampling: runningRequest.request.sampling,
            eosTokenIds: runningRequest.request.eosTokenIds,
            cacheSession: runningRequest.request.cacheSession
        )

        generator.remove(uid: uid)
        running.removeValue(forKey: uid)
        resumeGeneratedTokens[uid] = runningRequest.generatedTokens
        waiting.insert(resumedRequest, at: 0)
        return true
    }

    private func canResumeFromTokenPrompt(_ runningRequest: RunningRequest) -> Bool {
        !runningRequest.promptTokens.isEmpty
            && runningRequest.request.input.image == nil
            && runningRequest.request.input.video == nil
            && runningRequest.request.input.audio == nil
            && runningRequest.request.input.text.tokens.ndim == 1
            && runningRequest.request.input.text.mask == nil
    }

    private func currentContextTokens(for runningRequest: RunningRequest) -> [Int] {
        runningRequest.promptTokens
            + runningRequest.generatedTokens.dropFirst(runningRequest.generatedTokensIncludedInPrompt)
    }

    private func pressureSnapshot() -> PressureSnapshot {
        PressureSnapshot(
            runningUIDs: generator.uids,
            waitingCount: waiting.count,
            admissionInProgressUID: admissionInProgress?.request.uid
        )
    }

    private func logCacheFailure(_ message: String, _ error: Error) {
        let line = "MLXCat cache warning: \(message): \(error)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private func finishReason(
        token: Int,
        generatedTokenCount: Int,
        request: Request
    ) -> FinishReason? {
        if request.eosTokenIds.contains(token) {
            return .stop
        }
        if generatedTokenCount >= request.maxTokens {
            return .length
        }
        return nil
    }

    private func sampledToken(
        from logits: MLXArray,
        sampling: SamplingParameters
    ) -> PreparedGeneratedToken {
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
        return PreparedGeneratedToken(
            token: token,
            tokenID: tokenID,
            thinkingBudgetState: thinkingBudgetState
        )
    }

    private func isPrefixCacheEligible(_ input: LMInput) -> Bool {
        input.image == nil && input.video == nil && input.audio == nil
    }

}

private enum PreparedAdmission {
    case ready(PreparedBatchRow)
    case pending(AdmissionInProgress)
}

private struct AdmissionInProgress {
    let request: Request
    let sampling: SamplingParameters
    let cache: [any KVCache]
    let promptTokens: [Int]
    let promptTokensArray: MLXArray
    let storedPromptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
    let prefillRange: Range<Int>
    var nextPrefillIndex: Int
    var state: LMOutput.State?
    var initialGeneratedToken: PreparedGeneratedToken?
}

private struct PreparedBatchRow {
    let cache: [any KVCache]
    let lastToken: MLXArray
    let promptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
    let initialGeneratedToken: PreparedGeneratedToken?
    /// Model state produced by the prefill (e.g. Qwen3.5/Qwen-VL M-RoPE
    /// ropeDeltas). Stateful models refuse to decode a warm cache without it.
    let modelState: LMOutput.State?
}

private struct PreparedGeneratedToken {
    let token: MLXArray
    let tokenID: Int
    let thinkingBudgetState: ThinkingBudgetState?
}
