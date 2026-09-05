import Foundation
import MLXCat
import MLXCatHTTP

public final class PoolBackedChatBackend<Loader: EnginePoolModelLoader>: OpenAIModelLifecycleBackend,
    OpenAICompletionBackend, OpenAIEmbeddingsBackend, OpenAIRerankBackend, AudioTranscriptionBackend,
    AnthropicTokenCountingBackend, OpenAIHealthProviding, @unchecked Sendable
where Loader.Engine == NativeModelEngine {
    public let models: [OpenAIModelInfo]
    public var embeddingModels: [OpenAIModelInfo] { embeddingsBackend?.embeddingModels ?? [] }
    public var rerankModels: [OpenAIModelInfo] { rerankBackend?.rerankModels ?? [] }
    public var transcriptionModels: [OpenAIModelInfo] { speechBackend?.transcriptionModels ?? [] }
    public var healthInfo: OpenAIHealthInfo {
        OpenAIHealthInfo(
            defaultModel: models.first?.id,
            enginePool: OpenAIHealthEnginePool(modelCount: models.count, loadedCount: models.count)
        )
    }

    private let pool: EnginePool<Loader>
    private let embeddingsBackend: (any OpenAIEmbeddingsBackend)?
    private let rerankBackend: (any OpenAIRerankBackend)?
    private let speechBackend: (any AudioTranscriptionBackend)?
    private let memoryWatchdog: MemoryWatchdog?

    public init(
        pool: EnginePool<Loader>,
        modelIDs: [String],
        embeddingsBackend: (any OpenAIEmbeddingsBackend)? = nil,
        rerankBackend: (any OpenAIRerankBackend)? = nil,
        speechBackend: (any AudioTranscriptionBackend)? = nil,
        memoryWatchdog: MemoryWatchdog? = nil
    ) {
        self.pool = pool
        self.embeddingsBackend = embeddingsBackend
        self.rerankBackend = rerankBackend
        self.speechBackend = speechBackend
        self.memoryWatchdog = memoryWatchdog
        self.models = modelIDs.sorted().map { OpenAIModelInfo(id: $0, maxModelLength: nil) }
    }

    /// Consult the live memory watchdog before starting generation. For text
    /// requests, callers pass the projected KV cache footprint so the watchdog
    /// reserves both prompt and generated-token growth before decode begins.
    private func admitOrThrow(additionalBytes: Int64 = 0) async throws {
        guard let memoryWatchdog else { return }
        do {
            try await memoryWatchdog.checkAdmission(additionalBytes: additionalBytes)
        } catch let error as MemoryWatchdogError {
            switch error {
            case .admissionDenied(_, let current, let ceiling):
                throw OpenAIHTTPError(
                    status: 507,
                    message: "Insufficient memory: usage \(current) exceeds ceiling \(ceiling) after reclaim.",
                    retryAfterSeconds: 1
                )
            }
        }
    }

    public func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        let ticket: EnginePoolQueueTicket
        do {
            ticket = try await pool.admitWaitingRequest()
        } catch {
            throw openAIError(from: error)
        }

        do {
            let lease = try await pool.acquire(request.model)
            let stream: OpenAIChatStream
            var reservation: UUID?
            do {
                let prepared = try await lease.engine.prepareChatCompletion(request)
                let admitted = try await prepared.admitAndStart(
                    pool: pool,
                    additionalBytes: lease.engine.estimatedKVCacheBytes(
                        promptTokens: prepared.promptTokens,
                        maxGeneratedTokens: request.maxTokens
                    ),
                    checkPressure: { [self] in try await admitOrThrow(additionalBytes: $0) }
                )
                reservation = admitted.reservation
                stream = admitted.stream
            } catch {
                if let reservation { await pool.releaseRequest(reservation) }
                await pool.release(lease)
                await pool.finishWaitingRequest(ticket)
                throw error
            }
            return leasedStream(stream, lease: lease, ticket: ticket, reservation: reservation)
        } catch {
            await pool.finishWaitingRequest(ticket)
            throw openAIError(from: error)
        }
    }

    public func startCompletion(_ request: OpenAICompletionRequest) async throws -> OpenAIChatStream {
        let ticket: EnginePoolQueueTicket
        do {
            ticket = try await pool.admitWaitingRequest()
        } catch {
            throw openAIError(from: error)
        }
        do {
            let lease = try await pool.acquire(request.model)
            let stream: OpenAIChatStream
            var reservation: UUID?
            do {
                let prepared = try await lease.engine.prepareCompletion(request)
                let admitted = try await prepared.admitAndStart(
                    pool: pool,
                    additionalBytes: lease.engine.estimatedKVCacheBytes(
                        promptTokens: prepared.promptTokens,
                        maxGeneratedTokens: request.maxTokens
                    ),
                    checkPressure: { [self] in try await admitOrThrow(additionalBytes: $0) }
                )
                reservation = admitted.reservation
                stream = admitted.stream
            } catch {
                if let reservation { await pool.releaseRequest(reservation) }
                await pool.release(lease)
                await pool.finishWaitingRequest(ticket)
                throw error
            }
            return leasedStream(stream, lease: lease, ticket: ticket, reservation: reservation)
        } catch {
            await pool.finishWaitingRequest(ticket)
            throw openAIError(from: error)
        }
    }

    public func embed(_ request: OpenAIEmbeddingsRequest) async throws -> OpenAIEmbeddingsResult {
        guard let embeddingsBackend else {
            throw OpenAIHTTPError(status: 404, message: "embeddings backend unavailable")
        }
        return try await embeddingsBackend.embed(request)
    }

    public func rerank(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult {
        guard let rerankBackend else {
            throw OpenAIHTTPError(status: 404, message: "rerank backend unavailable")
        }
        return try await rerankBackend.rerank(request)
    }

    public func transcribe(_ request: AudioTranscriptionRequest) async throws -> AudioTranscriptionResult {
        guard let speechBackend else {
            throw OpenAIHTTPError(status: 501, message: "transcription backend not configured")
        }
        return try await speechBackend.transcribe(request)
    }

    public func countTokens(_ request: AnthropicCountTokensRequest) async throws -> AnthropicCountTokensResult {
        do {
            let lease = try await pool.acquire(request.model)
            do {
                let count = try await lease.engine.countPromptTokens(for: request.openAIRequest())
                await pool.release(lease)
                return AnthropicCountTokensResult(inputTokens: count)
            } catch {
                await pool.release(lease)
                throw error
            }
        } catch let poolError as EnginePoolError {
            if case .modelNotFound = poolError {
                return AnthropicCountTokensResult(
                    inputTokens: request.estimatedInputTokens(),
                    estimated: true
                )
            }
            throw openAIError(from: poolError)
        }
    }

    public func modelPoolStatus() async throws -> OpenAIModelPoolStatus {
        let prefixMemory = (await pool.loadedEngines()).reduce(Int64(0)) { total, loaded in
            total + loaded.engine.prefixCacheStats().currentBytes
        }
        var status = OpenAIModelPoolStatus(await pool.status()).addingPrefixCacheMemory(prefixMemory)
        let resources = await pool.resourceStatuses()
        let resourceModels = resources.map(OpenAIModelRuntimeStatus.init)
        status = status.appendingSpeechModels(resourceModels, speechMemory: 0)
        if let speechBackend = speechBackend as? SpeechModelLifecycleRouting {
            let accountedPaths = Set(resourceModels.map(\.modelPath))
            let speechModels = await speechBackend.speechModelStatuses().filter {
                !accountedPaths.contains($0.modelPath)
            }
            let speechMemory = speechModels.compactMap(\.actualSize).reduce(Int64(0), +)
            status = status.appendingSpeechModels(speechModels, speechMemory: speechMemory)
        }
        return status
    }

    public func loadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        if let result = try await PoolResourceModelLifecycle(pool: pool).load(
            id, allowBareModelID: await allowBareResourceModelID(id)
        ) { return result }
        if let registrySpeechBackend = speechBackend as? SpeechModelLifecycleRouting,
           await shouldRouteSpeechLifecycle(id, speechBackend: registrySpeechBackend) {
            return try await registrySpeechBackend.loadModel(id)
        }
        try await admitOrThrow()
        do {
            let result = try await pool.load(id)
            let message = result.alreadyLoaded ? "Already loaded: \(id)" : "Loaded: \(id)"
            return OpenAIModelLifecycleResult(modelID: id, message: message)
        } catch {
            throw openAIError(from: error)
        }
    }

    public func unloadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        if let result = try await PoolResourceModelLifecycle(pool: pool).unload(
            id, allowBareModelID: await allowBareResourceModelID(id)
        ) { return result }
        if let registrySpeechBackend = speechBackend as? SpeechModelLifecycleRouting,
           await shouldRouteSpeechLifecycle(id, speechBackend: registrySpeechBackend) {
            return try await registrySpeechBackend.unloadModel(id)
        }
        do {
            try await pool.unload(id)
            return OpenAIModelLifecycleResult(modelID: id)
        } catch {
            throw openAIError(from: error)
        }
    }

    private func isLLMModel(_ id: String) -> Bool {
        models.contains { $0.id == id }
    }

    private func shouldRouteSpeechLifecycle(_ id: String, speechBackend: SpeechModelLifecycleRouting) async -> Bool {
        if !isLLMModel(id) {
            return true
        }
        guard speechBackend.isNamespacedSpeechModelReference(id) else {
            return false
        }
        return await speechBackend.canResolveModelReference(id)
    }

    private func leasedStream(
        _ stream: OpenAIChatStream,
        lease: EnginePoolLease<NativeModelEngine>,
        ticket: EnginePoolQueueTicket,
        reservation: UUID?
    ) -> OpenAIChatStream {
        PoolLeasedChatStream.make(stream, pool: pool, lease: lease, ticket: ticket, reservation: reservation)
    }

    private func openAIError(from error: Error) -> OpenAIHTTPError {
        if case let MemoryWatchdogError.admissionDenied(required, current, ceiling) = error {
            return OpenAIHTTPError(status: 507,
                                   message: "Insufficient memory: \(required) requested bytes, "
                                    + "\(current) used, \(ceiling) ceiling.",
                                   retryAfterSeconds: 1)
        }
        guard let poolError = error as? EnginePoolError else {
            if let httpError = error as? OpenAIHTTPError {
                return httpError
            }
            return OpenAIHTTPError(status: 500, message: String(describing: error))
        }
        return OpenAIHTTPError(
            status: poolError.httpStatus,
            message: poolError.message,
            retryAfterSeconds: poolError.retryAfterSeconds,
            code: poolError.modelNotFoundCode
        )
    }
}

private extension PoolBackedChatBackend {
    func allowBareResourceModelID(_ id: String) async -> Bool {
        guard !isLLMModel(id) else { return false }
        if let speech = speechBackend as? SpeechModelLifecycleRouting,
           await speech.canResolveModelReference(id) { return false }
        return true
    }
}

private extension EnginePoolError {
    var modelNotFoundCode: String? {
        if case .modelNotFound = self { return "model_not_found" }
        return nil
    }
}

private extension OpenAIModelPoolStatus {
    init(_ status: EnginePoolStatus) {
        self.init(
            finalCeiling: status.finalCeiling,
            currentModelMemory: status.currentModelMemory,
            modelCount: status.modelCount,
            loadedCount: status.loadedCount,
            models: status.models.map(OpenAIModelRuntimeStatus.init)
        )
    }

    func appendingSpeechModels(
        _ speechModels: [OpenAIModelRuntimeStatus],
        speechMemory: Int64
    ) -> OpenAIModelPoolStatus {
        OpenAIModelPoolStatus(
            finalCeiling: finalCeiling,
            currentModelMemory: currentModelMemory + speechMemory,
            modelCount: modelCount + speechModels.count,
            loadedCount: loadedCount + speechModels.filter(\.loaded).count,
            models: (models + speechModels).sorted { lhs, rhs in
                if lhs.id == rhs.id { return lhs.modelType < rhs.modelType }
                return lhs.id < rhs.id
            }
        )
    }

    func addingPrefixCacheMemory(_ prefixMemory: Int64) -> OpenAIModelPoolStatus {
        OpenAIModelPoolStatus(
            finalCeiling: finalCeiling,
            currentModelMemory: currentModelMemory + prefixMemory,
            modelCount: modelCount,
            loadedCount: loadedCount,
            models: models
        )
    }
}

private extension OpenAIModelRuntimeStatus {
    init(_ state: EnginePoolResourceStatus) {
        self.init(
            id: state.resource.modelID, modelType: state.resource.modelType, modelPath: state.resource.id,
            loaded: state.loaded, isLoading: state.isLoading || state.isUnloading,
            estimatedSize: state.resource.footprint.bytes ?? 0, actualSize: nil, pinned: false,
            lastAccess: state.lastAccess?.timeIntervalSince1970, inUse: state.inUse,
            footprintSource: state.resource.footprint.source.rawValue,
            accountedBytes: (state.loaded || state.isLoading) ? state.resource.footprint.bytes : 0
        )
    }

    init(_ status: EnginePoolModelStatus) {
        self.init(
            id: status.id,
            modelPath: status.modelPath,
            loaded: status.loaded,
            isLoading: status.isLoading,
            estimatedSize: status.estimatedSize,
            actualSize: status.actualSize,
            pinned: status.pinned,
            lastAccess: status.lastAccess,
            inUse: status.inUse
        )
    }
}
