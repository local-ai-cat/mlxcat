import Foundation
import MLXCatHTTP
import MLXCatSpeech

/// Serves `/v1/audio/transcriptions` from the speech engine registry — the
/// route stops 501ing the moment any adapter is registered.
final class RegistrySpeechBackend: AudioTranscriptionBackend, SpeechModelLifecycleRouting, @unchecked Sendable {
    private let registry: SpeechEngineRegistry
    private let modelsLock = NSLock()
    private var cachedModels: [OpenAIModelInfo]
    private var loadedModelKeys: Set<String> = []

    init(registry: SpeechEngineRegistry) async {
        self.registry = registry
        self.cachedModels = await Self.snapshotModels(registry)
    }

    /// The route gate and /v1/models read this synchronously; it is a cache,
    /// refreshed after every transcribe so adapter catalog changes (new model
    /// folders appearing) converge without a restart.
    var transcriptionModels: [OpenAIModelInfo] {
        modelsLock.lock()
        defer { modelsLock.unlock() }
        return cachedModels
    }

    func transcribe(_ request: AudioTranscriptionRequest) async throws -> AudioTranscriptionResult {
        defer { Task { await self.refreshModels() } }

        let candidates: [any SpeechEngineAdapter]
        let modelID: String
        do {
            (candidates, modelID) = try await registry.resolveCandidates(model: request.model)
        } catch let error as SpeechEngineError {
            throw Self.httpError(from: error, model: request.model, models: transcriptionModels)
        }

        // A claimed model can still fail to load (corrupt folder) — fall through
        // to the next candidate; only surface the last failure.
        var lastError: Error = SpeechEngineError.unknownModel(request.model)
        for adapter in candidates {
            do {
                let result = try await adapter.transcribeFile(
                    SpeechFileTranscriptionRequest(
                        model: modelID,
                        fileName: request.fileName,
                        fileData: request.fileData,
                        language: request.language,
                        temperature: request.temperature
                    )
                )
                return AudioTranscriptionResult(
                    text: result.text,
                    language: result.language,
                    duration: result.duration,
                    segments: segments(from: result)
                )
            } catch {
                lastError = error
            }
        }
        if let speechError = lastError as? SpeechEngineError {
            throw Self.httpError(from: speechError, model: request.model, models: transcriptionModels)
        }
        throw lastError
    }

    private func refreshModels() async {
        let fresh = await Self.snapshotModels(registry)
        storeModels(fresh)
    }

    private func storeModels(_ fresh: [OpenAIModelInfo]) {
        modelsLock.lock()
        cachedModels = fresh
        modelsLock.unlock()
    }

    private static func snapshotModels(_ registry: SpeechEngineRegistry) async -> [OpenAIModelInfo] {
        await registry.allModels().map { OpenAIModelInfo(id: $0.id, maxModelLength: nil, modelType: "audio_stt") }
    }

    func speechModelStatuses() async -> [OpenAIModelRuntimeStatus] {
        let adapters = await registry.allAdapters()
        var statuses: [OpenAIModelRuntimeStatus] = []
        for adapter in adapters {
            let footprint = await adapter.loadedFootprint()
            var reportedAdapterFootprint = false
            for model in await adapter.availableModels().sorted(by: { $0.id < $1.id }) {
                let loaded = isLoaded(engineID: model.engineID, modelID: model.id)
                let actualSize: Int64?
                if loaded && !reportedAdapterFootprint {
                    // Adapter APIs expose one process/ANE/CoreML working-set estimate,
                    // not separable per-model allocations. Attribute it to the first
                    // loaded model for the adapter so status totals do not double count.
                    actualSize = footprint
                    reportedAdapterFootprint = true
                } else {
                    actualSize = nil
                }
                statuses.append(
                    OpenAIModelRuntimeStatus(
                        id: model.id,
                        modelType: "audio_stt",
                        modelPath: "speech://\(model.engineID)/\(model.id)",
                        loaded: loaded,
                        isLoading: false,
                        estimatedSize: model.footprint ?? 0,
                        actualSize: actualSize,
                        pinned: false,
                        lastAccess: nil,
                        inUse: 0
                    )
                )
            }
        }
        return statuses.sorted { lhs, rhs in
            if lhs.id == rhs.id { return lhs.modelPath < rhs.modelPath }
            return lhs.id < rhs.id
        }
    }

    func isNamespacedSpeechModelReference(_ id: String) -> Bool {
        let parts = id.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
            return false
        }
        return true
    }

    func canResolveModelReference(_ id: String) async -> Bool {
        do {
            _ = try await registry.resolveCandidates(model: id)
            return true
        } catch {
            return false
        }
    }

    func loadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        let candidates: [any SpeechEngineAdapter]
        let modelID: String
        do {
            (candidates, modelID) = try await registry.resolveCandidates(model: id)
        } catch let error as SpeechEngineError {
            throw Self.httpError(from: error, model: id, models: transcriptionModels)
        }

        var lastError: Error = SpeechEngineError.unknownModel(id)
        for adapter in candidates {
            do {
                try await adapter.loadModel(modelID)
                markLoaded(engineID: adapter.engineID, modelID: modelID)
                return OpenAIModelLifecycleResult(modelID: id, message: "Loaded: \(id)")
            } catch {
                lastError = error
            }
        }
        if let speechError = lastError as? SpeechEngineError {
            throw Self.httpError(from: speechError, model: id, models: transcriptionModels)
        }
        throw lastError
    }

    func unloadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        let candidates: [any SpeechEngineAdapter]
        let modelID: String
        do {
            (candidates, modelID) = try await registry.resolveCandidates(model: id)
        } catch let error as SpeechEngineError {
            throw Self.httpError(from: error, model: id, models: transcriptionModels)
        }

        for adapter in candidates {
            await adapter.unloadModel(modelID)
            markUnloaded(engineID: adapter.engineID, modelID: modelID)
        }
        return OpenAIModelLifecycleResult(modelID: id)
    }

    private func isLoaded(engineID: String, modelID: String) -> Bool {
        modelsLock.lock()
        defer { modelsLock.unlock() }
        return loadedModelKeys.contains(modelKey(engineID: engineID, modelID: modelID))
    }

    private func markLoaded(engineID: String, modelID: String) {
        modelsLock.lock()
        loadedModelKeys.insert(modelKey(engineID: engineID, modelID: modelID))
        modelsLock.unlock()
    }

    private func markUnloaded(engineID: String, modelID: String) {
        modelsLock.lock()
        loadedModelKeys.remove(modelKey(engineID: engineID, modelID: modelID))
        modelsLock.unlock()
    }

    private func modelKey(engineID: String, modelID: String) -> String {
        "\(engineID):\(modelID)"
    }

    private static func httpError(
        from error: SpeechEngineError,
        model: String,
        models: [OpenAIModelInfo]
    ) -> OpenAIHTTPError {
        switch error {
        case .unknownModel:
            let known = models.map(\.id).sorted().joined(separator: ", ")
            return OpenAIHTTPError(
                status: 404,
                message: "model '\(model)' not found. Available transcription models: \(known)"
            )
        case .fileTranscriptionUnsupported, .streamingUnsupported:
            return OpenAIHTTPError(status: 400, message: String(describing: error))
        case .modelNotLoaded, .engineFailure:
            return OpenAIHTTPError(status: 500, message: String(describing: error))
        }
    }

    private func segments(from result: SpeechTranscriptionResult) -> [AudioTranscriptionSegment]? {
        guard let words = result.words, !words.isEmpty else { return nil }
        return words.enumerated().map { index, word in
            AudioTranscriptionSegment(id: index, start: word.start, end: word.end, text: word.text)
        }
    }
}
