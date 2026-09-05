import Foundation
import MLXCat
import MLXCatSpeech

/// All entrypoints, including registry streaming, share the chat pool's owner.
final class PoolBackedSpeechAdapter: SpeechEngineAdapter, Sendable {
    let adapter: any SpeechEngineAdapter
    let pool: any EnginePoolResourceManaging
    var engineID: String { adapter.engineID }
    var displayName: String { adapter.displayName }
    var capabilities: SpeechEngineCapabilities { adapter.capabilities }

    init(adapter: any SpeechEngineAdapter, pool: any EnginePoolResourceManaging) async {
        self.adapter = adapter
        self.pool = pool
        _ = await availableModels()
    }

    func availableModels() async -> [SpeechModelInfo] {
        let models = await adapter.availableModels()
        for model in models { await register(model) }
        return models
    }

    private func register(_ model: SpeechModelInfo) async {
        let modelID = model.id
        let footprint = EnginePoolFootprint(bytes: model.footprint,
                                            source: model
                                            .footprintSource == .diskEstimate ? .diskEstimate : .configuredEstimate)
        let adapter = self.adapter
        await pool.registerResource(EnginePoolResource(
            id: key(modelID), modelID: modelID, modelType: "audio_stt", footprint: footprint,
            // Speech adapters do not promise allocation through MLX. CoreML,
            // ANE and CPU estimates must be included alongside MLX observations.
            usesMLXAllocator: false,
            load: { try await adapter.loadModel(modelID) },
            unload: { await adapter.unloadModel(modelID) }
        ))
    }

    private func acquire(_ modelID: String) async throws -> EnginePoolResourceLease {
        guard await availableModels().contains(where: { $0.id == modelID }) else {
            throw SpeechEngineError.unknownModel(modelID)
        }
        return try await pool.acquireResource(key(modelID))
    }

    func key(_ modelID: String) -> String { "speech://\(engineID)/\(modelID)" }

    func loadModel(_ modelID: String) async throws {
        let lease = try await acquire(modelID)
        await pool.releaseResource(lease)
    }

    func unloadModel(_ modelID: String) async {
        // Protocol compatibility: the HTTP lifecycle uses the throwing owner
        // directly so busy errors remain visible to callers.
        try? await pool.unloadResource(key(modelID))
    }

    func unloadModelChecked(_ modelID: String) async throws {
        try await pool.unloadResource(key(modelID))
    }

    func loadedFootprint() async -> Int64? {
        let statuses = await pool.resourceStatuses().filter {
            $0.resource.id.hasPrefix("speech://\(engineID)/") && $0.loaded
        }
        guard statuses.allSatisfy({ $0.resource.footprint.bytes != nil }) else { return nil }
        return statuses.reduce(0) { $0 + ($1.resource.footprint.bytes ?? 0) }
    }

    func transcribeFile(_ request: SpeechFileTranscriptionRequest) async throws -> SpeechTranscriptionResult {
        let lease = try await acquire(request.model)
        do {
            let result = try await adapter.transcribeFile(request)
            await pool.releaseResource(lease)
            return result
        } catch {
            await pool.releaseResource(lease)
            throw error
        }
    }

    func makeStreamSession(modelID: String, language: String?) async throws -> any SpeechStreamSession {
        let lease = try await acquire(modelID)
        do {
            let session = try await adapter.makeStreamSession(modelID: modelID, language: language)
            return PoolLeasedSpeechSession(session: session, pool: pool, lease: lease)
        } catch {
            await pool.releaseResource(lease)
            throw error
        }
    }
}

/// A cancellation request is not completion: in-flight push/finish operations
/// retain the lease until their adapter calls have actually returned.
private actor PoolLeasedSpeechSession: SpeechStreamSession {
    nonisolated let segments: AsyncThrowingStream<SpeechStreamSegment, Error>
    private let session: any SpeechStreamSession
    private let pool: any EnginePoolResourceManaging
    private let lease: EnginePoolResourceLease
    private var activeCalls = 0
    private var closed = false
    private var released = false
    private var cancellationAcknowledged = false

    init(session: any SpeechStreamSession, pool: any EnginePoolResourceManaging, lease: EnginePoolResourceLease) {
        self.session = session
        self.pool = pool
        self.lease = lease
        self.segments = session.segments
    }

    deinit {
        guard !released else { return }
        let session = self.session
        let pool = self.pool
        let lease = self.lease
        Task {
            await session.cancel()
            await pool.releaseResource(lease)
        }
    }

    func push(_ buffer: SpeechPCMBuffer) async throws {
        guard !closed else { return }
        activeCalls += 1
        do {
            try await session.push(buffer)
            activeCalls -= 1
            await releaseIfFinished()
        } catch {
            activeCalls -= 1
            await cancel()
            throw error
        }
    }

    func finish() async throws {
        guard !closed else { return }
        closed = true
        activeCalls += 1
        do {
            try await session.finish()
            activeCalls -= 1
            cancellationAcknowledged = true
            await releaseIfFinished()
        } catch {
            activeCalls -= 1
            await cancel()
            throw error
        }
    }

    func cancel() async {
        closed = true
        await session.cancel()
        cancellationAcknowledged = true
        await releaseIfFinished()
    }

    func latencyStats() async -> SpeechSessionLatencyStats { await session.latencyStats() }

    private func releaseIfFinished() async {
        guard closed, cancellationAcknowledged, activeCalls == 0, !released else { return }
        released = true
        await pool.releaseResource(lease)
    }
}
