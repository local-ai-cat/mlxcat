import MLXLMCommon

public final class MLXCatEngine: @unchecked Sendable {
    private let resident: AsyncResourceOwner<EngineResident>

    public init(
        model: any LanguageModel,
        parameters: GenerateParameters = GenerateParameters(),
        maxConcurrentRequests: Int,
        prefixStore: (any PrefixKVStore)? = nil,
        cacheCapabilities: ModelCacheCapabilities = .default,
        serializedDecode: Bool = false,
        serializationPolicy: SerializationPolicy? = nil,
        schedulerManagedTextPrefill: Bool = true,
        chunkIdlePrefill: Bool = true,
        pressurePolicy: Scheduler.PressurePolicy = .disabled,
        cacheReleasePolicy: Scheduler.CacheReleasePolicy? = nil,
        kvQuantization: KVQuantizationPolicy = .off,
        speculativeDecoding: SpeculativeDecodingConfiguration = SpeculativeDecodingConfiguration()
    ) {
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: parameters,
            maxConcurrentRequests: maxConcurrentRequests,
            prefixStore: prefixStore,
            cacheCapabilities: cacheCapabilities,
            serializedDecode: serializedDecode,
            serializationPolicy: serializationPolicy,
            schedulerManagedTextPrefill: schedulerManagedTextPrefill,
            chunkIdlePrefill: chunkIdlePrefill,
            pressurePolicy: pressurePolicy,
            cacheReleasePolicy: cacheReleasePolicy,
            kvQuantization: kvQuantization,
            speculativeDecoding: speculativeDecoding
        )
        self.resident = AsyncResourceOwner(EngineResident(
            scheduler: scheduler, streamDemux: EngineStreamDemux(scheduler: scheduler)
        ))
    }

    public func submit(_ request: Request) async throws {
        try await resident.withValue { try await $0.scheduler.submit(request) }
    }

    public func cancel(uid: String) async {
        try? await resident.withValue { await $0.scheduler.cancel(uid: uid) }
    }

    @discardableResult
    public func step() async throws -> [Response] {
        try await resident.withValue { try await $0.scheduler.step() }
    }

    public func generate(_ requests: [Request]) async throws -> [String: [Int]] {
        try await resident.withValue { state in
            for request in requests { try await state.scheduler.submit(request) }
            while await !state.scheduler.isIdle { _ = try await state.scheduler.step() }
            var result: [String: [Int]] = [:]
            for request in requests { result[request.uid] = await state.scheduler.consumeTokens(for: request.uid) }
            return result
        }
    }

    public struct StreamSession: Sendable {
        public let responses: AsyncThrowingStream<Response, Error>
        public let cancelAndWait: @Sendable () async -> Void
    }

    public func stream(_ request: Request) -> AsyncThrowingStream<Response, Error> {
        streamSession(request).responses
    }

    public func streamSession(_ request: Request) -> StreamSession {
        let (responses, continuation) = AsyncThrowingStream<Response, Error>.makeStream()
        let registration = Task { [resident] in
            do {
                try await resident.withValue { await $0.streamDemux.add(request, continuation: continuation) }
            } catch { continuation.finish(throwing: error) }
        }
        let uid = request.uid
        let cancelAndWait: @Sendable () async -> Void = { [resident] in
            await registration.value
            await resident.withValueOrAwaitInvalidation { await $0.streamDemux.cancelAndWait(uid: uid) }
        }
        continuation.onTermination = { _ in Task { await cancelAndWait() } }
        return StreamSession(responses: responses, cancelAndWait: cancelAndWait)
    }

    public func responses(for uid: String) async -> [Response] {
        (try? await resident.withValue { await $0.scheduler.responses(for: uid) }) ?? []
    }

    public func tokens(for uid: String) async -> [Int] {
        (try? await resident.withValue { await $0.scheduler.tokens(for: uid) }) ?? []
    }

    public var isIdle: Bool {
        get async { (try? await resident.withValue { await $0.scheduler.isIdle }) ?? true }
    }

    public func shutdown() async {
        await resident.invalidate { await $0.streamDemux.shutdown() }
    }
}

private struct EngineResident: Sendable {
    let scheduler: Scheduler
    let streamDemux: EngineStreamDemux
}

private actor EngineStreamDemux {
    private let scheduler: Scheduler
    private var continuations: [String: AsyncThrowingStream<Response, Error>.Continuation] = [:]
    private var registeringUIDs: Set<String> = []
    private var pumpTask: Task<Void, Never>?
    private var closing = false

    init(scheduler: Scheduler) {
        self.scheduler = scheduler
    }

    func add(
        _ request: Request,
        continuation: AsyncThrowingStream<Response, Error>.Continuation
    ) async {
        guard !closing else { continuation.finish(throwing: ResourceOwnerError.unloaded); return }
        registeringUIDs.insert(request.uid)
        do {
            try await scheduler.submit(request)
            registeringUIDs.remove(request.uid)
            continuations[request.uid] = continuation
            startPumpIfNeeded()
        } catch {
            registeringUIDs.remove(request.uid)
            continuation.finish(throwing: error)
        }
    }

    func cancelAndWait(uid: String) async {
        let continuation = continuations.removeValue(forKey: uid)
        await scheduler.cancelAndWait(uid: uid)
        await scheduler.discardResponses(for: uid)
        continuation?.finish()
    }

    func shutdown() async {
        closing = true
        let ids = Array(continuations.keys)
        for id in ids { await cancelAndWait(uid: id) }
        let pump = pumpTask
        pump?.cancel()
        await pump?.value
    }

    private func startPumpIfNeeded() {
        guard !closing, pumpTask == nil else { return }
        pumpTask = Task { [weak self] in
            await self?.pump()
        }
    }

    private func pump() async {
        do {
            while !closing {
                let responses = try await scheduler.step()
                for response in responses {
                    await deliver(response)
                }
                if responses.isEmpty {
                    let idle = await scheduler.isIdle
                    if idle {
                        if continuations.isEmpty && registeringUIDs.isEmpty {
                            break
                        }
                        if registeringUIDs.isEmpty {
                            finishRemaining()
                            break
                        }
                    }
                    await Task.yield()
                }
            }
        } catch {
            finishAll(throwing: error)
        }
        pumpTask = nil
        if !continuations.isEmpty || !registeringUIDs.isEmpty {
            startPumpIfNeeded()
        }
    }

    private func deliver(_ response: Response) async {
        guard let continuation = continuations[response.uid] else { return }
        continuation.yield(response)
        if response.finishReason != nil {
            continuation.finish()
            continuations.removeValue(forKey: response.uid)
            await scheduler.discardResponses(for: response.uid)
        }
    }

    private func finishRemaining() {
        for continuation in continuations.values {
            continuation.finish()
        }
        continuations.removeAll()
    }

    private func finishAll(throwing error: Error) {
        for continuation in continuations.values {
            continuation.finish(throwing: error)
        }
        continuations.removeAll()
    }
}
