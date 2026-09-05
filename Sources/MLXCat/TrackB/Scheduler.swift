import Foundation
import MLX
import MLXLMCommon

public actor Scheduler {
    private let model: any LanguageModel
    private let generator: ContinuousBatchGenerator
    private let maxConcurrentRequests: Int
    private let queueLimit: Int
    private let prefixStore: (any PrefixKVStore)?
    private let prefixCacheEnabled: Bool
    private let serializationPolicy: SerializationPolicy
    private let promptPreparation: SchedulerPromptPreparation
    private let prefillConfiguration: SchedulerPrefill.Configuration
    private let pressurePolicy: PressurePolicy
    private var waiting: [Request] = []
    private var running: [String: RunningRequest] = [:]
    private var admissionInProgress: AdmissionInProgress?
    private var resumeGeneratedTokens: [String: [Int]] = [:]
    private var pendingCancellation: Set<String> = []
    private var decoder: SchedulerDecode
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
        self.generator = ContinuousBatchGenerator(
            model: modelBox.model,
            parameters: parameters,
            speculativeDecoding: speculativeDecoding
        )
        self.maxConcurrentRequests = maxConcurrentRequests
        self.queueLimit = max(maxConcurrentRequests * 4, 32)
        self.prefixStore = prefixStore
        self.prefixCacheEnabled =
            prefixStore != nil && !cacheCapabilities.usesWindowedKVCache && !kvQuantization.isEnabled
        self.serializationPolicy = Self.configuredSerializationPolicy(
            kvQuantization: kvQuantization, serializationPolicy: serializationPolicy, serializedDecode: serializedDecode
        )
        self.promptPreparation = SchedulerPromptPreparation(
            model: modelBox.model, parameters: parameters, prefixStore: prefixStore,
            prefixCacheEnabled: self.prefixCacheEnabled, schedulerManagedTextPrefill: schedulerManagedTextPrefill
        )
        self.prefillConfiguration = SchedulerPrefill.Configuration(
            stepSize: parameters.prefill.stepSize, chunkIdlePrefill: chunkIdlePrefill,
            prefillsLastTokenAlone: Self.prefillsLastTokenAlone(
                usesWindowedKVCache: cacheCapabilities.usesWindowedKVCache
            ),
            kvQuantization: kvQuantization
        )
        self.pressurePolicy = pressurePolicy
        self.decoder = SchedulerDecode(
            generator: self.generator, collector: self.collector, prefixStore: prefixStore,
            prefixCacheEnabled: self.prefixCacheEnabled, cacheReleasePolicy: cacheReleasePolicy ?? .fromEnvironment()
        )
    }
}

// Request lifecycle and decode steps.
extension Scheduler {
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
            || admissionInProgress?.request.uid == request.uid {
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
        if admissionInProgress?.request.uid == uid {
            Stream.gpu.synchronize()
            let hit = admissionInProgress?.prefixHit
            admissionInProgress = nil
            if let hit { prefixStore?.release(hit) }
            resumeGeneratedTokens.removeValue(forKey: uid)
            collector.record(Response(uid: uid, token: -1, finishReason: .cancelled))
            return
        }
        if running[uid] != nil {
            pendingCancellation.insert(uid)
        }
    }

    /// Cancellation acknowledgement uses the same terminal removal path as
    /// step(). Actor isolation waits for an in-flight step; that path synchronizes
    /// the GPU and removes KV rows before this method can return.
    public func cancelAndWait(uid: String) {
        cancel(uid: uid)
        _ = applyPendingCancellation(only: uid)
        decoder.releaseCacheIfDrained(isIdle: isIdle)
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
           preemptYoungestResumableRequest() {
            return processed
        }

        processed.append(contentsOf: decoder.processGeneratedResponses(
            generator.next(), running: &running, resumedTokens: &resumeGeneratedTokens
        ))

        if generator.isEmpty {
            processed.append(contentsOf: admitWaiting(allowPartialPrefill: false))
        }
        decoder.releaseCacheIfDrained(isIdle: isIdle)
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
        decoder.droppedStaleResponseCount
    }

    public var speculativeDecodingStats: SpeculativeDecodingStats {
        generator.speculationStats
    }
}

// Queue admission and terminal cancellation.
extension Scheduler {
    private func admitWaiting(allowPartialPrefill: Bool) -> [Response] {
        var admittedResponses: [Response] = []
        while running.count < maxConcurrentRequests {
            let busy = !running.isEmpty || !generator.isEmpty || !admittedResponses.isEmpty
            if Self.refusesAdmission(
                policy: serializationPolicy, running: running, nextRequest: waiting.first, busy: busy
            ) {
                return admittedResponses
            }

            if admissionInProgress == nil {
                guard !waiting.isEmpty else { return admittedResponses }
                switch beginAdmission(waiting.removeFirst()) {
                case .ready(let response):
                    if let response { admittedResponses.append(response) }
                    continue
                case .pending: break
                }
            }
            switch resumeAdmission(allowPartialPrefill: allowPartialPrefill) {
            case .ready(let response):
                if let response { admittedResponses.append(response) }
            case .pending: return admittedResponses
            }
        }
        return admittedResponses
    }

    private enum AdmissionOutcome {
        case ready(Response?)
        case pending
    }

    private func beginAdmission(_ request: Request) -> AdmissionOutcome {
        do {
            var sampling = request.sampling
            sampling.eosTokenIds.formUnion(request.eosTokenIds)
            if !request.eosTokenIds.isEmpty {
                sampling.xtcSpecialTokens = Array(Set(sampling.xtcSpecialTokens).union(request.eosTokenIds))
            }
            switch try promptPreparation.prepare(request, sampling: sampling) {
            case .ready(let row):
                return .ready(try completeAdmission(row, request: request, sampling: sampling))
            case .pending(let admission):
                admissionInProgress = admission
                return .pending
            }
        } catch {
            return .ready(recordAdmissionFailure(request, error: error))
        }
    }

    private func resumeAdmission(allowPartialPrefill: Bool) -> AdmissionOutcome {
        do {
            guard let row = try advanceAdmission(allowPartialPrefill: allowPartialPrefill),
                  let admission = admissionInProgress else { return .pending }
            let response = try completeAdmission(row, request: admission.request, sampling: admission.sampling)
            admissionInProgress = nil
            return .ready(response)
        } catch {
            let request = admissionInProgress?.request
            let hit = admissionInProgress?.prefixHit
            Stream.gpu.synchronize()
            admissionInProgress = nil
            if let hit { prefixStore?.release(hit) }
            return .ready(request.map { recordAdmissionFailure($0, error: error) })
        }
    }

    private func recordAdmissionFailure(_ request: Request, error: Error) -> Response {
        let response = Response(uid: request.uid, token: -1, finishReason: .failed(String(describing: error)))
        collector.record(response)
        return response
    }

    private func applyPendingCancellation(only uid: String? = nil) -> [Response] {
        let cancellations = uid.map { pendingCancellation.intersection([$0]) } ?? pendingCancellation
        guard !cancellations.isEmpty else { return [] }

        Stream.gpu.synchronize()
        var responses: [Response] = []
        for uid in cancellations {
            guard running[uid] != nil else { continue }
            let hit = running[uid]?.prefixHit
            generator.remove(uid: uid)
            running.removeValue(forKey: uid)
            if let hit { prefixStore?.release(hit) }
            resumeGeneratedTokens.removeValue(forKey: uid)
            let response = Response(uid: uid, token: -1, finishReason: .cancelled)
            collector.record(response)
            responses.append(response)
        }
        pendingCancellation.subtract(cancellations)
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
            SchedulerPrefill.finishReason(
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
        let row = SchedulerPrefill.advance(
            &admission,
            model: model,
            configuration: prefillConfiguration,
            allowPartialPrefill: allowPartialPrefill,
            priorGeneratedTokens: resumeGeneratedTokens[admission.request.uid] ?? []
        )
        if row == nil { admissionInProgress = admission }
        return row
    }

    private func storeCompletedAdmissionPrefix(_ row: PreparedBatchRow, request: Request) {
        guard prefixCacheEnabled else { return }
        SchedulerPrefixCache.storeCompletedAdmissionPrefix(row, request: request, store: prefixStore)
    }

    private func publishAvailablePrefixBlocks(
        uid: String, runningRequest: RunningRequest, minimumNewTokens: Int = 1
    ) {
        guard prefixCacheEnabled else { return }
        SchedulerPrefixCache.publishAvailablePrefixBlocks(
            uid: uid, runningRequest: runningRequest, generator: generator,
            store: prefixStore, minimumNewTokens: minimumNewTokens
        )
    }

    private func preemptYoungestResumableRequest() -> Bool {
        guard let uid = generator.uids.reversed().first(where: { uid in
            guard let request = running[uid] else { return false }
            return SchedulerPromptPreparation.canResumeFromTokenPrompt(request)
        }),
        let runningRequest = running[uid]
        else {
            return false
        }

        publishAvailablePrefixBlocks(uid: uid, runningRequest: runningRequest)
        let hit = runningRequest.prefixHit
        let resumedRequest = SchedulerPromptPreparation.resumedRequest(for: runningRequest)

        Stream.gpu.synchronize()
        generator.remove(uid: uid)
        running.removeValue(forKey: uid)
        if let hit { prefixStore?.release(hit) }
        resumeGeneratedTokens[uid] = runningRequest.generatedTokens
        waiting.insert(resumedRequest, at: 0)
        return true
    }

    private func pressureSnapshot() -> PressureSnapshot {
        PressureSnapshot(
            runningUIDs: generator.uids,
            waitingCount: waiting.count,
            admissionInProgressUID: admissionInProgress?.request.uid
        )
    }
}

private typealias AdmissionInProgress = SchedulerPrefill.Admission
private typealias PreparedBatchRow = SchedulerPrefill.Row
