import Foundation
@testable import MLXCat
@testable import MLXCatHTTP
@testable import MLXCatNative
@testable import MLXCatHTTPServer
@testable import MLXCatSpeech
import XCTest

// MARK: - Fake adapter

/// In-memory adapter used both to test the registry and as the reference
/// implementation of the conformance battery below.
final class FakeSpeechAdapter: SpeechEngineAdapter, @unchecked Sendable {
    let engineID: String
    let capabilities: SpeechEngineCapabilities
    private let models: [SpeechModelInfo]
    private let footprintBytes: Int64
    private let state = FakeSpeechAdapterState()

    var displayName: String { "Fake (\(engineID))" }

    init(
        engineID: String,
        silicon: SpeechEngineSilicon,
        streaming: Bool = true,
        modelIDs: [String],
        footprintBytes: Int64 = 0
    ) {
        self.engineID = engineID
        self.footprintBytes = footprintBytes
        self.capabilities = SpeechEngineCapabilities(
            supportsFileTranscription: true,
            supportsStreaming: streaming,
            silicon: silicon,
            wordTimestamps: true
        )
        self.models = modelIDs.map {
            SpeechModelInfo(id: $0, engineID: engineID, displayName: $0)
        }
    }

    func availableModels() async -> [SpeechModelInfo] { models }

    func loadModel(_ modelID: String) async throws {
        guard models.contains(where: { $0.id == modelID }) else {
            throw SpeechEngineError.unknownModel(modelID)
        }
        await state.markLoaded(modelID)
    }

    func unloadModel(_ modelID: String) async {
        await state.markUnloaded(modelID)
    }

    func loadedFootprint() async -> Int64? {
        await state.hasLoadedModels ? footprintBytes : 0
    }

    func isLoadedForTest(_ modelID: String) async -> Bool {
        await state.isLoaded(modelID)
    }

    func loadCallsForTest(_ modelID: String) async -> Int {
        await state.loadCalls(modelID)
    }

    func unloadCallsForTest(_ modelID: String) async -> Int {
        await state.unloadCalls(modelID)
    }

    func transcribeFile(_ request: SpeechFileTranscriptionRequest) async throws -> SpeechTranscriptionResult {
        SpeechTranscriptionResult(
            text: "fake transcript via \(engineID)/\(request.model)",
            language: request.language ?? "en",
            duration: 1.0,
            words: [
                SpeechWord(text: "fake", start: 0.0, end: 0.4),
                SpeechWord(text: "transcript", start: 0.4, end: 1.0)
            ]
        )
    }

    func makeStreamSession(modelID: String, language: String?) async throws -> any SpeechStreamSession {
        guard capabilities.supportsStreaming else {
            throw SpeechEngineError.streamingUnsupported(engineID: engineID)
        }
        return FakeStreamSession()
    }
}

private actor FakeSpeechAdapterState {
    private var loadedModelIDs: Set<String> = []
    private var loadCallCounts: [String: Int] = [:]
    private var unloadCallCounts: [String: Int] = [:]

    var hasLoadedModels: Bool {
        !loadedModelIDs.isEmpty
    }

    func markLoaded(_ modelID: String) {
        loadedModelIDs.insert(modelID)
        loadCallCounts[modelID, default: 0] += 1
    }

    func markUnloaded(_ modelID: String) {
        loadedModelIDs.remove(modelID)
        unloadCallCounts[modelID, default: 0] += 1
    }

    func isLoaded(_ modelID: String) -> Bool {
        loadedModelIDs.contains(modelID)
    }

    func loadCalls(_ modelID: String) -> Int {
        loadCallCounts[modelID] ?? 0
    }

    func unloadCalls(_ modelID: String) -> Int {
        unloadCallCounts[modelID] ?? 0
    }
}

/// Emits one partial per pushed second of audio, then finalizes on finish() —
/// exercising the partial→final identity contract.
actor FakeStreamSession: SpeechStreamSession {
    nonisolated let segments: AsyncThrowingStream<SpeechStreamSegment, Error>
    private let continuation: AsyncThrowingStream<SpeechStreamSegment, Error>.Continuation
    private var fedSeconds: Double = 0
    private var emissions = 0

    init() {
        var continuation: AsyncThrowingStream<SpeechStreamSegment, Error>.Continuation!
        self.segments = AsyncThrowingStream { continuation = $0 }
        self.continuation = continuation
    }

    func push(_ buffer: SpeechPCMBuffer) async throws {
        fedSeconds += buffer.duration
        emissions += 1
        continuation.yield(
            SpeechStreamSegment(
                id: 0,
                kind: .partial,
                text: "partial-\(emissions)",
                start: 0,
                end: fedSeconds
            )
        )
    }

    func finish() async throws {
        continuation.yield(
            SpeechStreamSegment(
                id: 0,
                kind: .final,
                text: "final transcript",
                start: 0,
                end: fedSeconds,
                words: [SpeechWord(text: "final", start: 0, end: fedSeconds)]
            )
        )
        continuation.finish()
    }

    func cancel() async {
        continuation.finish()
    }

    func latencyStats() async -> SpeechSessionLatencyStats {
        SpeechSessionLatencyStats(sampleCount: emissions, p50Seconds: 0.1, p95Seconds: 0.2)
    }
}

/// Claims models in its catalog but fails every load — the corrupt-folder case.
final class BrokenSpeechAdapter: SpeechEngineAdapter, @unchecked Sendable {
    let engineID: String
    let capabilities: SpeechEngineCapabilities
    private let models: [SpeechModelInfo]

    var displayName: String { "Broken (\(engineID))" }

    init(engineID: String, silicon: SpeechEngineSilicon, modelIDs: [String]) {
        self.engineID = engineID
        self.capabilities = SpeechEngineCapabilities(
            supportsFileTranscription: true,
            supportsStreaming: false,
            silicon: silicon
        )
        self.models = modelIDs.map { SpeechModelInfo(id: $0, engineID: engineID, displayName: $0) }
    }

    func availableModels() async -> [SpeechModelInfo] { models }

    func transcribeFile(_ request: SpeechFileTranscriptionRequest) async throws -> SpeechTranscriptionResult {
        throw SpeechEngineError.engineFailure("corrupt model folder")
    }

    func makeStreamSession(modelID: String, language: String?) async throws -> any SpeechStreamSession {
        throw SpeechEngineError.streamingUnsupported(engineID: engineID)
    }
}

// MARK: - Registry tests
