import MLXLMCommon

public final class MLXCatEngine: @unchecked Sendable {
    private let scheduler: Scheduler
    private let streamDemux: EngineStreamDemux

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
            speculativeDecoding: speculativeDecoding
        )
        self.scheduler = scheduler
        self.streamDemux = EngineStreamDemux(scheduler: scheduler)
    }

    public func submit(_ request: Request) async throws {
        try await scheduler.submit(request)
    }

    public func cancel(uid: String) async {
        await scheduler.cancel(uid: uid)
    }

    @discardableResult
    public func step() async throws -> [Response] {
        try await scheduler.step()
    }

    public func generate(_ requests: [Request]) async throws -> [String: [Int]] {
        for request in requests {
            try await scheduler.submit(request)
        }

        while await !scheduler.isIdle {
            _ = try await scheduler.step()
        }

        var result: [String: [Int]] = [:]
        for request in requests {
            result[request.uid] = await scheduler.consumeTokens(for: request.uid)
        }
        return result
    }

    public func stream(_ request: Request) -> AsyncThrowingStream<Response, Error> {
        AsyncThrowingStream { continuation in
            continuation.onTermination = { [streamDemux] _ in
                Task {
                    await streamDemux.cancel(uid: request.uid)
                }
            }
            Task { [streamDemux] in
                await streamDemux.add(request, continuation: continuation)
            }
        }
    }

    public func responses(for uid: String) async -> [Response] {
        await scheduler.responses(for: uid)
    }

    public func tokens(for uid: String) async -> [Int] {
        await scheduler.tokens(for: uid)
    }

    public var isIdle: Bool {
        get async {
            await scheduler.isIdle
        }
    }
}

private actor EngineStreamDemux {
    private let scheduler: Scheduler
    private var continuations: [String: AsyncThrowingStream<Response, Error>.Continuation] = [:]
    private var registeringUIDs: Set<String> = []
    private var pumpTask: Task<Void, Never>?

    init(scheduler: Scheduler) {
        self.scheduler = scheduler
    }

    func add(
        _ request: Request,
        continuation: AsyncThrowingStream<Response, Error>.Continuation
    ) async {
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

    func cancel(uid: String) async {
        continuations.removeValue(forKey: uid)
        registeringUIDs.remove(uid)
        await scheduler.cancel(uid: uid)
    }

    private func startPumpIfNeeded() {
        guard pumpTask == nil else { return }
        pumpTask = Task { [weak self] in
            await self?.pump()
        }
    }

    private func pump() async {
        do {
            while true {
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
