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
    private let prefillsLastTokenAlone: Bool
    /// Hybrid prefix reuse (default on; `MLXCAT_HYBRID_PREFIX_REUSE=0` disables): capture recurrent-state checkpoints during
    /// prefill so hybrid models can resume a follow-up turn from the prefix cache.
    private let capturesRecurrentCheckpoints: Bool
    /// See ``fullPromptMatchReuseEnabled(environment:)``.
    private let reusesFullPromptMatch: Bool
    private let pressurePolicy: PressurePolicy
    private var waiting: [Request] = []
    private var running: [String: RunningRequest] = [:]
    private var admissionInProgress: AdmissionInProgress?
    private var resumeGeneratedTokens: [String: [Int]] = [:]
    private var droppedStaleResponseCount = 0
    private var pendingCancellation: Set<String> = []
    private let cacheReleasePolicy: CacheReleasePolicy
    private let kvQuantization: KVQuantizationPolicy
    private var decodeStepsSinceCacheClear = 0
    /// Set when the drain-to-idle clear has already fired for the current idle
    /// stretch, so a scheduler polled while idle clears once rather than on
    /// every call.
    private var hasReleasedCacheForThisIdleStretch = true
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
        cacheReleasePolicy: CacheReleasePolicy? = nil,
        kvQuantization: KVQuantizationPolicy = .off,
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
        self.kvQuantization = kvQuantization
        // Stage 1 of KV quantization is single-stream, and the gate is here
        // rather than in a comment. A quantized row that reaches
        // `BatchLayerCache` is misrouted by its state array count: rows of
        // different lengths throw, and — worse — two rows of the SAME length
        // pass shape validation and then have their `[step, offset, groupSize,
        // bits]` metaState read as slot metadata, which is silent wrong output.
        // `.always` keeps every row alone, so that path is unreachable rather
        // than merely unlikely.
        //
        // The prefix stores are excluded for a milder reason: both require the
        // 2-array keys/values schema and would throw on every publish, which the
        // scheduler catches and logs. That is not a crash, it is a cache that
        // has silently stopped working while filling the log — so it is turned
        // off explicitly instead.
        self.prefixCacheEnabled =
            prefixStore != nil && !cacheCapabilities.usesWindowedKVCache && !kvQuantization.isEnabled
        if kvQuantization.isEnabled {
            self.serializationPolicy = .always
        } else {
            self.serializationPolicy = serializationPolicy ?? (serializedDecode ? .always : .never)
        }
        self.schedulerManagedTextPrefill = schedulerManagedTextPrefill
        self.chunkIdlePrefill = chunkIdlePrefill
        self.capturesRecurrentCheckpoints = Self.hybridPrefixReuseEnabled()
        self.reusesFullPromptMatch = Self.fullPromptMatchReuseEnabled()
        self.prefillsLastTokenAlone = Self.prefillsLastTokenAlone(
            usesWindowedKVCache: cacheCapabilities.usesWindowedKVCache
        )
        self.pressurePolicy = pressurePolicy
        self.cacheReleasePolicy = cacheReleasePolicy ?? .fromEnvironment()
    }

    /// Whether the last prompt token is prefilled alone so the one forward whose
    /// logits we read is `[1, 1, vocab]` rather than `[1, chunk, vocab]`.
    ///
    /// mlx-lm always does this — it loops `while y.size > 1` and hands the single
    /// remaining token to the step that samples
    /// (`guest/mlx-lm/mlx_lm/generate.py:580-587`). It is safe for them because
    /// that remaining token goes through `_step()`, the SAME single-token decode
    /// path every later token uses.
    ///
    /// Ours does not: the extra one-token forward still goes through the PREFILL
    /// call site. A rotating (sliding-window) KV cache has separate multi-token
    /// and single-token update paths, so that extra prefill-path `S == 1` update
    /// desynchronises the ring. Measured on gpt-oss-20b (window 128):
    /// `SlidingWindowBatchIntegrationTests` went from 3/3 green to 60 of 160
    /// tokens diverging from serial with a sustained run of 37, starting at
    /// token 97 — and every non-windowed gate stayed green, which is what
    /// localises it. Bisected: disabling this alone restores 3/3.
    ///
    /// So it is capability-gated rather than reverted: windowed caches keep the
    /// old path, everything else takes the smaller tensor.
    /// `MLXCAT_PREFILL_LAST_TOKEN_ALONE=always|never` forces it either way, so
    /// the trade-off can be measured on a windowed model rather than argued —
    /// `always` is how you reproduce the divergence above.
    /// On by default since the before/after numbers came in (packet P1, 2026-09-23:
    /// the 9K cached-prefix follow-up went from 46 s to 3 s with `grid` placement,
    /// token-identical to a cold run). `MLXCAT_HYBRID_PREFIX_REUSE=0|false|off`
    /// turns it off; that is the A/B lever, not a safety switch.
    public static func hybridPrefixReuseEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        switch environment["MLXCAT_HYBRID_PREFIX_REUSE"]?.lowercased() {
        case "0", "false", "off": return false
        default: return true
        }
    }

    /// Whether a prefix hit covering the WHOLE prompt is reused (off by default;
    /// `MLXCAT_PREFIX_FULL_MATCH_REUSE=1` turns it on).
    ///
    /// A request identical to a stored one (a retry, a regenerate, a benchmark's
    /// warm repeat) matches every prompt token, which leaves no token to prefill
    /// and so no logits to sample from. Today that hit is released and the whole
    /// prompt is prefilled cold. With the lever on, the store is asked again for
    /// the prompt minus its last token: a trimmable cache comes back one token
    /// short and only that token is prefilled; a hybrid slot resumes from its
    /// deepest checkpoint, as any shorter match does. The same "reuse at most
    /// N-1" rule is what Trans-N-ai/swama#123 and mlx-lm's prompt cache use.
    public static func fullPromptMatchReuseEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        switch environment["MLXCAT_PREFIX_FULL_MATCH_REUSE"]?.lowercased() {
        case "1", "true", "on": return true
        default: return false
        }
    }

    /// Where prefill leaves recurrent checkpoints.
    ///
    /// `grid` (default, `MLXCAT_HYBRID_PREFIX_CHECKPOINTS=grid`): the last prefill
    /// chunk boundary below the prompt end. Token-identical to a cold run, because a
    /// resume re-creates the cold run's chunk boundaries; costs up to one chunk (511
    /// tokens at the idle step) of re-prefill per follow-up turn.
    ///
    /// `last` (`MLXCAT_HYBRID_PREFIX_CHECKPOINTS=last`): `promptCount - 1` plus the
    /// last 2048 boundary. Re-prefills almost nothing, but the resume's chunk
    /// boundaries differ from a cold run's, so greedy output can flip on a near-tie
    /// (measured 3/20 turns on 2026-09-23, P1.md).
    enum CheckpointPlacement { case grid, lastToken }

    static let checkpointPlacement: CheckpointPlacement =
        ProcessInfo.processInfo.environment["MLXCAT_HYBRID_PREFIX_CHECKPOINTS"]?.lowercased() == "last"
        ? .lastToken : .grid

    /// A recurrent checkpoint lands on the last multiple of this below the prompt
    /// end, besides the one at `promptCount - 1`: the fallback for a follow-up
    /// whose template diverges from the previous prompt earlier than its final
    /// token (oMLX checkpoints on fixed blocks the same way).
    static let recurrentCheckpointBlock = 2048

    /// Test hook: `MLXCAT_HYBRID_PREFIX_SABOTAGE=1` records every checkpoint one
    /// position EARLIER than the state it holds, so a resume feeds a token twice. An
    /// exactness check that cannot see that is not a check.
    private static let checkpointSabotage =
        ProcessInfo.processInfo.environment["MLXCAT_HYBRID_PREFIX_SABOTAGE"] == "1"

    static func prefillsLastTokenAlone(
        usesWindowedKVCache: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        switch environment["MLXCAT_PREFILL_LAST_TOKEN_ALONE"]?.lowercased() {
        case "always", "1", "true": return true
        case "never", "0", "false": return false
        default: return !usesWindowedKVCache
        }
    }

    /// When the MLX buffer cache is handed back to the OS.
    ///
    /// MLX keeps freed Metal buffers in a cache and does not return them on its
    /// own, so a server that once served a 16k-token prompt keeps that footprint
    /// resident forever. Measured on the 2026-08-23 board: gemma-4-E2B's 16k
    /// tier peaked at 8.39 GiB, and every later cell in the same process
    /// reported ~7.8 GiB no matter how little it actually needed — the same
    /// process, hours later, still holding gigabytes for nobody. On a laptop
    /// that is the whole complaint.
    ///
    /// The references both do more than we did, which was to clear between
    /// prefill chunks and nowhere else:
    ///
    ///   * mlx-lm clears every 512 decode steps
    ///     (`guest/mlx-lm/mlx_lm/generate.py:1779`), so a long generation does
    ///     not accumulate.
    ///   * omlx clears on engine transitions
    ///     (`guest/omlx/omlx/engine_core.py:85`).
    ///
    /// Both knobs are on by default and both are switchable, because the cost is
    /// real and worth measuring rather than asserting: a cleared cache means the
    /// next allocation goes to the Metal allocator instead of a free list.
    /// `MLXCAT_DECODE_CLEAR_CACHE_STEPS=0` disables the periodic clear (any other
    /// non-negative integer sets the interval); `MLXCAT_IDLE_CLEAR_CACHE=0`
    /// disables the drain-to-idle clear.
    ///
    /// `MLXCAT_ADMISSION_CLEAR_CACHE=1` (off by default) also clears once each
    /// time a prefilled row joins the decode batch. mlx-swift-lm does the
    /// single-stream version of this, clearing on the first generated token
    /// (ml-explore/mlx-swift-lm#620): a request shorter than the decode interval
    /// otherwise never clears. Our idle release already covers back-to-back
    /// requests; the case it does not cover is a server that never drains, where
    /// each admission's prefill scratch (shaped `[1, chunk, ...]`, never reused
    /// by `[B, 1, ...]` decode) sits on the free list until the next interval.
    public struct CacheReleasePolicy: Sendable, Equatable {
        /// Clear every N decode steps. Zero never clears.
        public var decodeStepInterval: Int
        /// Clear once each time the scheduler drains to fully idle.
        public var releasesWhenIdle: Bool
        /// Clear once each time an admitted row finishes prefill.
        public var releasesAfterAdmission: Bool

        public init(
            decodeStepInterval: Int = 512,
            releasesWhenIdle: Bool = true,
            releasesAfterAdmission: Bool = false
        ) {
            self.decodeStepInterval = max(0, decodeStepInterval)
            self.releasesWhenIdle = releasesWhenIdle
            self.releasesAfterAdmission = releasesAfterAdmission
        }

        /// mlx-lm's interval, plus the idle release it has no need for (its
        /// server does not hold one process across unrelated workloads).
        public static let `default` = CacheReleasePolicy()
        public static let never = CacheReleasePolicy(decodeStepInterval: 0, releasesWhenIdle: false)

        public static func fromEnvironment(
            _ environment: [String: String] = ProcessInfo.processInfo.environment
        ) -> CacheReleasePolicy {
            var policy = CacheReleasePolicy.default
            if let raw = environment["MLXCAT_DECODE_CLEAR_CACHE_STEPS"],
                let steps = Int(raw.trimmingCharacters(in: .whitespaces)), steps >= 0
            {
                policy.decodeStepInterval = steps
            }
            switch environment["MLXCAT_IDLE_CLEAR_CACHE"]?.lowercased() {
            case "0", "false", "never": policy.releasesWhenIdle = false
            case "1", "true", "always": policy.releasesWhenIdle = true
            default: break
            }
            switch environment["MLXCAT_ADMISSION_CLEAR_CACHE"]?.lowercased() {
            case "1", "true", "always": policy.releasesAfterAdmission = true
            case "0", "false", "never": policy.releasesAfterAdmission = false
            default: break
            }
            return policy
        }
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
        decodeStepsSinceCacheClear += 1
        if cacheReleasePolicy.decodeStepInterval > 0,
            decodeStepsSinceCacheClear >= cacheReleasePolicy.decodeStepInterval
        {
            Memory.clearCache()
            decodeStepsSinceCacheClear = 0
        }
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
        releaseCacheIfDrained()
        return processed
    }

    /// Hand the buffer cache back once per idle stretch, not once per poll.
    private func releaseCacheIfDrained() {
        guard isIdle else {
            hasReleasedCacheForThisIdleStretch = false
            return
        }
        guard cacheReleasePolicy.releasesWhenIdle, !hasReleasedCacheForThisIdleStretch else {
            return
        }
        hasReleasedCacheForThisIdleStretch = true
        decodeStepsSinceCacheClear = 0
        Memory.clearCache()
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
        if cacheReleasePolicy.releasesAfterAdmission {
            // Only buffers already on MLX's free list go back; the row's cache
            // and any pending graph keep theirs.
            Memory.clearCache()
        }
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
                toolGrammarState: seededGeneratedTokens.isEmpty
                    ? row.initialGeneratedToken?.toolGrammarState
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
            running[request.uid]?.checkpoints = row.checkpoints
        } else {
            storeCompletedAdmissionPrefix(row, request: request)
            // A row that finishes at admission (max_tokens 1, or EOS first) never
            // reaches `running`, so the finish path below never releases its
            // prefix lease. Without this the slot stays leased for good: no
            // later fetch can see it and eviction skips it.
            if let hit = row.prefixHit {
                prefixStore?.release(hit)
            }
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
        // The LAST prompt token is prefilled alone, so the one forward whose
        // logits we actually read produces `[1, 1, vocab]` instead of
        // `[1, chunk, vocab]`.
        //
        // mlx-lm structures prefill exactly this way — it loops `while y.size > 1`
        // and hands the single remaining token to the step that samples
        // (`guest/mlx-lm/mlx_lm/generate.py:580-587`, and the same shape at
        // `:430-452`). Its prefill chunks evaluate only the cache
        // (`mx.eval([c.state for c in cache])`), never the logits, so MLX's
        // laziness means a prefill chunk's logits are never computed at all.
        //
        // We read `output.logits` on whichever chunk ends the range, which at a
        // 262k vocab is ~268 MB of fp16 for a 512-token chunk and gigabytes on
        // the single-pass path — all of it discarded except one row.
        let logitsBoundary =
            prefillsLastTokenAlone
            ? admission.prefillRange.upperBound - 1
            : admission.prefillRange.upperBound
        let checkpointTargets = recurrentCheckpointTargets(for: admission, prefillStep: prefillStep)
        while admission.nextPrefillIndex < admission.prefillRange.upperBound, remainingBudget > 0 {
            var end: Int
            if admission.nextPrefillIndex == logitsBoundary {
                // The final token, alone. This is the forward we sample from.
                end = admission.prefillRange.upperBound
            } else if chunked {
                end = min(
                    admission.nextPrefillIndex + min(prefillStep, remainingBudget),
                    logitsBoundary
                )
            } else {
                end = logitsBoundary
            }
            if let target = checkpointTargets.first(where: { $0 > admission.nextPrefillIndex }),
                target < end
            {
                end = target
            }
            let input = LMInput.Text(
                tokens: admission.promptTokensArray[admission.nextPrefillIndex ..< end]
            )
            let output = model(input[text: .newAxis], cache: admission.cache, state: admission.state)
            admission.state = output.state
            if checkpointTargets.contains(end) {
                admission.checkpoints.append(recurrentCheckpoint(of: admission.cache, at: end))
                Self.prefixDebug("checkpoint uid=\(admission.request.uid) at=\(end)")
            }
            if end == admission.prefillRange.upperBound {
                admission.initialGeneratedToken = sampledToken(
                    from: output.logits,
                    sampling: admission.sampling,
                    priorGeneratedTokens: resumeGeneratedTokens[admission.request.uid] ?? []
                )
            }
            asyncEval(admission.cache)
            // Convert inside the chunk loop, as mlx-lm does
            // (`guest/mlx-lm/mlx_lm/generate.py:418,441`), not once at the end:
            // conversion allocates the quantized copy while the fp16 original is
            // still live, so converting per chunk caps that doubling at one
            // chunk's worth of KV instead of the whole prompt's. A no-op until
            // the cache passes `startTokens`, and a no-op forever for rotating
            // and recurrent layers, which upstream declines to quantize.
            kvQuantization.apply(to: &admission.cache)
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
            modelState: admission.state,
            checkpoints: admission.checkpoints
        )
    }

    private func prepareForInsert(_ request: Request, sampling: SamplingParameters) throws
        -> PreparedAdmission
    {
        let rowCache = try model.newCache(parameters: parameters)
        let prefixCacheEligible = isPrefixCacheEligible(request.input)
        // A VLM processor hands text-only prompts over as a batch of one
        // (`[1, L]`); that is the same prompt as `[L]`, and flattening it is what
        // lets a text-only turn on a VLM (the Qwen3.8-27B checkpoint ships with a
        // vision tower) reach scheduler-managed prefill and the prefix cache.
        let requestText: LMInput.Text
        if request.input.text.tokens.ndim == 2, request.input.text.tokens.dim(0) == 1,
            request.input.text.mask == nil
        {
            requestText = LMInput.Text(tokens: request.input.text.tokens.reshaped([-1]))
        } else {
            requestText = request.input.text
        }
        let canUseRawTextTokens = schedulerManagedTextPrefill
            && prefixCacheEligible
            && requestText.tokens.ndim == 1
            && requestText.mask == nil
        Self.prefixDebug(
            "admission uid=\(request.uid) rawText=\(canUseRawTextTokens) managed=\(schedulerManagedTextPrefill) "
                + "eligible=\(prefixCacheEligible) enabled=\(prefixCacheEnabled) ndim=\(request.input.text.tokens.ndim) "
                + "mask=\(request.input.text.mask != nil)")
        let promptText: LMInput.Text
        if canUseRawTextTokens {
            promptText = requestText
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
            var hit = prefixStore.fetch(tokens: promptTokens, sessionKey: request.cacheSession)
        {
            Self.prefixDebug("hit uid=\(request.uid) matched=\(hit.matchedTokenCount) of \(promptTokens.count)")
            if hit.matchedTokenCount == promptTokens.count,
                reusesFullPromptMatch,
                promptTokens.count > 2
            {
                prefixStore.release(hit)
                if let shorter = prefixStore.fetch(
                    tokens: Array(promptTokens.dropLast()), sessionKey: request.cacheSession)
                {
                    Self.prefixDebug(
                        "full-match reuse uid=\(request.uid) matched=\(shorter.matchedTokenCount) of \(promptTokens.count)")
                    hit = shorter
                } else {
                    return prefillMissRow(
                        request: request,
                        sampling: sampling,
                        promptTokens: promptTokens,
                        promptTokensArray: promptTokensArray,
                        storedPromptTokens: promptTokens,
                        rowCache: rowCache
                    )
                }
            }
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
            // A hit resumes on a WARM cache, which is the shape that trapped
            // batched qwen3_5 inside the model. Seeded rather than left nil —
            // see ``BatchPositionalState/textOnlyResumeState()``.
            state: BatchPositionalState.textOnlyResumeState(),
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

    /// Positions inside this admission's prefill where recurrent state is worth
    /// keeping. Empty unless the flag is on and the cache has a layer that cannot trim.
    private func recurrentCheckpointTargets(
        for admission: AdmissionInProgress,
        prefillStep: Int
    ) -> [Int] {
        guard capturesRecurrentCheckpoints,
            !admission.storedPromptTokens.isEmpty,
            admission.cache.contains(where: { !$0.isTrimmable })
        else { return [] }
        let promptEnd = admission.prefillRange.upperBound
        let lastToken = promptEnd - 1
        let candidates: [Int]
        switch Self.checkpointPlacement {
        case .grid:
            // Only on the prefill chunk grid, which a cold admission (starting at
            // 0) crosses anyway: the checkpoint adds no boundary, and a resume from
            // it replays exactly the chunks a cold run of the follow-up would.
            let grid = max(1, prefillStep)
            candidates = [(lastToken / grid) * grid]
        case .lastToken:
            let block = (lastToken / Self.recurrentCheckpointBlock) * Self.recurrentCheckpointBlock
            candidates = [block, lastToken]
        }
        return Set(candidates)
            .filter { $0 > admission.nextPrefillIndex && $0 < promptEnd }
            .sorted()
    }

    private func recurrentCheckpoint(of cache: [any KVCache], at position: Int)
        -> PrefixRecurrentCheckpoint
    {
        PrefixRecurrentCheckpoint(
            position: Self.checkpointSabotage ? position - 1 : position,
            layers: cache.map { layer in
                guard !layer.isTrimmable else { return nil }
                return SerializedKVLayer(
                    state: layer.state,
                    metaState: layer.metaState,
                    className: String(describing: type(of: layer))
                )
            }
        )
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
                cache: snapshot,
                checkpoints: row.checkpoints
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
                cache: snapshot,
                checkpoints: runningRequest.checkpoints
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

    private static let prefixDebugEnabled =
        ProcessInfo.processInfo.environment["MLXCAT_PREFIX_DEBUG"] == "1"

    static func prefixDebug(_ message: @autoclosure () -> String) {
        guard prefixDebugEnabled else { return }
        FileHandle.standardError.write(Data("MLXCat prefix: \(message())\n".utf8))
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
        sampling: SamplingParameters,
        priorGeneratedTokens: [Int] = []
    ) -> PreparedGeneratedToken {
        let nextTokenLogits = logits[0..., -1, 0...]
        let matcher = sampling.jsonGrammar?.makeMatcher()
        let regexMatcher = sampling.regexGrammar?.makeMatcher()
        let gbnfMatcher = sampling.gbnfGrammar?.makeMatcher()
        var toolGrammarState = sampling.toolGrammar.map { QwenXMLToolGrammarState(configuration: $0) }
        for tokenID in priorGeneratedTokens {
            toolGrammarState?.observeGeneratedToken(tokenID)
        }
        var thinkingBudgetState = sampling.thinkingBudget.map(ThinkingBudgetState.init(configuration:))
        let token = TokenSampler.sample(
            logits: nextTokenLogits[0, 0...],
            parameters: sampling,
            generatedTokens: priorGeneratedTokens,
            jsonGrammarMatcher: matcher,
            regexGrammarMatcher: regexMatcher,
            gbnfGrammarMatcher: gbnfMatcher,
            toolGrammarMatcher: toolGrammarState?.activeMatcher,
            postToolAllowedTokenIDs: toolGrammarState?.activeMatcher == nil
                ? toolGrammarState?.activeAllowedTokenIDs
                : nil,
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
        toolGrammarState?.observeGeneratedToken(tokenID)
        thinkingBudgetState?.advance(tokenID: tokenID)
        return PreparedGeneratedToken(
            token: token,
            tokenID: tokenID,
            toolGrammarState: toolGrammarState,
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
    /// `var` because KV quantization REPLACES layers in place partway through
    /// prefill: `KVCacheSimple` becomes `QuantizedKVCache` once the cache passes
    /// the threshold, so the array cannot be a constant.
    var cache: [any KVCache]
    let promptTokens: [Int]
    let promptTokensArray: MLXArray
    let storedPromptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
    let prefillRange: Range<Int>
    var nextPrefillIndex: Int
    var state: LMOutput.State?
    var initialGeneratedToken: PreparedGeneratedToken?
    var checkpoints: [PrefixRecurrentCheckpoint] = []
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
    var checkpoints: [PrefixRecurrentCheckpoint] = []
}

private struct PreparedGeneratedToken {
    let token: MLXArray
    let tokenID: Int
    let toolGrammarState: QwenXMLToolGrammarState?
    let thinkingBudgetState: ThinkingBudgetState?
}
