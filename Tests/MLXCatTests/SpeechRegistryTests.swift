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

    func loadedFootprint() async -> Int64 {
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
                SpeechWord(text: "transcript", start: 0.4, end: 1.0),
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

final class SpeechRegistryTests: XCTestCase {
    func testResolveFindsModelAcrossAdapters() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny", "base"]))
        await registry.register(FakeSpeechAdapter(engineID: "parakeet", silicon: .ane, modelIDs: ["nemotron"]))

        let (adapter, modelID) = try await registry.resolve(model: "nemotron")
        XCTAssertEqual(adapter.engineID, "parakeet")
        XCTAssertEqual(modelID, "nemotron")
    }

    func testResolveUnknownModelThrows() async {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"]))

        do {
            _ = try await registry.resolve(model: "nope")
            XCTFail("expected unknownModel")
        } catch let error as SpeechEngineError {
            XCTAssertEqual(error, .unknownModel("nope"))
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }

    func testExplicitEngineNamespaceDisambiguates() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"]))
        await registry.register(FakeSpeechAdapter(engineID: "whispercpp", silicon: .cpu, modelIDs: ["tiny"]))

        let (adapter, _) = try await registry.resolve(model: "whispercpp:tiny")
        XCTAssertEqual(adapter.engineID, "whispercpp")
    }

    func testSiliconPreferenceSteersResolution() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "mlxasr", silicon: .gpu, modelIDs: ["shared"]))
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["shared"]))

        let (defaulted, _) = try await registry.resolve(model: "shared")
        XCTAssertEqual(defaulted.engineID, "mlxasr", "registration order wins without a preference")

        let (preferred, _) = try await registry.resolve(
            model: "shared",
            preferences: SpeechSessionPreferences(preferredSilicon: .ane)
        )
        XCTAssertEqual(preferred.engineID, "whisperkit")
    }

    func testRequireStreamingFiltersNonStreamingAdapters() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(
            FakeSpeechAdapter(engineID: "fileonly", silicon: .cpu, streaming: false, modelIDs: ["m"])
        )
        await registry.register(FakeSpeechAdapter(engineID: "streamer", silicon: .ane, modelIDs: ["m"]))

        let (adapter, _) = try await registry.resolve(
            model: "m",
            preferences: SpeechSessionPreferences(requireStreaming: true)
        )
        XCTAssertEqual(adapter.engineID, "streamer")
    }

    func testModelReferenceSplitting() {
        XCTAssertEqual(SpeechEngineRegistry.splitModelReference("tiny").engineID, nil)
        XCTAssertEqual(SpeechEngineRegistry.splitModelReference("wk:tiny").engineID, "wk")
        XCTAssertEqual(SpeechEngineRegistry.splitModelReference("wk:tiny").modelID, "tiny")
        XCTAssertEqual(SpeechEngineRegistry.splitModelReference(":tiny").engineID, nil)
    }
}

// MARK: - Stream session conformance

final class SpeechStreamSessionConformanceTests: XCTestCase {
    func testPartialsThenFinalWithStableIdentityAndMonotoneTimes() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "fake", silicon: .ane, modelIDs: ["m"]))
        let (adapter, modelID) = try await registry.resolve(model: "m")
        let session = try await adapter.makeStreamSession(modelID: modelID, language: nil)

        let tone = SpeechPCMBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000)
        try await session.push(tone)
        try await session.push(tone)
        try await session.finish()

        var collected: [SpeechStreamSegment] = []
        for try await segment in session.segments {
            collected.append(segment)
        }

        XCTAssertEqual(collected.count, 3)
        XCTAssertEqual(collected.map(\.kind), [.partial, .partial, .final])
        XCTAssertEqual(Set(collected.map(\.id)).count, 1, "final must share the partials' identity")
        let ends = collected.map(\.end)
        XCTAssertEqual(ends, ends.sorted(), "segment end times must be monotone")
        XCTAssertEqual(collected.last?.end ?? 0, 2.0, accuracy: 0.001)

        let stats = await session.latencyStats()
        XCTAssertGreaterThan(stats.sampleCount, 0)
    }
}

// MARK: - HTTP bridge

final class RegistrySpeechBackendTests: XCTestCase {
    func testBridgeServesRegistryModelsAndTranscribes() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"]))
        let backend = await RegistrySpeechBackend(registry: registry)

        XCTAssertEqual(backend.transcriptionModels.map(\.id), ["tiny"])

        let result = try await backend.transcribe(
            AudioTranscriptionRequest(model: "tiny", fileName: "t.wav", fileData: Data([0x00]))
        )
        XCTAssertEqual(result.text, "fake transcript via whisperkit/tiny")
        XCTAssertEqual(result.segments?.count, 2)
        XCTAssertEqual(result.segments?.first?.start, 0.0)
    }

    func testBridgeUnknownModelMapsTo404WithModelList() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"]))
        let backend = await RegistrySpeechBackend(registry: registry)

        do {
            _ = try await backend.transcribe(
                AudioTranscriptionRequest(model: "ghost", fileName: "t.wav", fileData: Data())
            )
            XCTFail("expected 404")
        } catch let error as OpenAIHTTPError {
            XCTAssertEqual(error.status, 404)
            XCTAssertTrue(error.message.contains("tiny"), "404 body should list available models")
        }
    }

    func testBridgeFallsThroughToNextCandidateOnLoadFailure() async throws {
        let registry = SpeechEngineRegistry()
        await registry.register(
            BrokenSpeechAdapter(engineID: "corrupt-ane", silicon: .ane, modelIDs: ["shared"])
        )
        await registry.register(FakeSpeechAdapter(engineID: "healthy", silicon: .cpu, modelIDs: ["shared"]))
        let backend = await RegistrySpeechBackend(registry: registry)

        let result = try await backend.transcribe(
            AudioTranscriptionRequest(model: "shared", fileName: "t.wav", fileData: Data([0x00]))
        )
        XCTAssertEqual(result.text, "fake transcript via healthy/shared")
    }

    func testModelsRouteListsSpeechModelsAsAudioSTT() async throws {
        let adapter = FakeSpeechAdapter(
            engineID: "whisperkit",
            silicon: .ane,
            modelIDs: ["tiny"]
        )
        let server = try await makeServer(speechAdapter: adapter)

        let body = server.modelsResponseForTesting()
        let data = try XCTUnwrap(body["data"] as? [[String: Any]])
        let tiny = try XCTUnwrap(data.first { $0["id"] as? String == "tiny" })
        let alpha = try XCTUnwrap(data.first { $0["id"] as? String == "alpha" })

        XCTAssertEqual(tiny["model_type"] as? String, "audio_stt")
        XCTAssertEqual(alpha["model_type"] as? String, "llm")
    }

    func testSpeechLifecycleRoutesLoadUnloadAdapterAndStatusIncludesFootprint() async throws {
        let adapter = FakeSpeechAdapter(
            engineID: "whisperkit",
            silicon: .ane,
            modelIDs: ["tiny"],
            footprintBytes: 123_456
        )
        let server = try await makeServer(speechAdapter: adapter)

        let load = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/tiny/load", headers: [:], body: Data())
        )
        XCTAssertEqual(load.status, 200)
        XCTAssertEqual(load.body["model_id"] as? String, "tiny")
        let loadCalls = await adapter.loadCallsForTest("tiny")
        let loadedAfterLoad = await adapter.isLoadedForTest("tiny")
        XCTAssertEqual(loadCalls, 1)
        XCTAssertTrue(loadedAfterLoad)

        let status = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "GET", path: "/v1/models/status", headers: [:], body: Data())
        )
        XCTAssertEqual(status.status, 200)
        XCTAssertEqual(status.body["model_count"] as? Int, 2)
        XCTAssertEqual(status.body["loaded_count"] as? Int, 1)
        XCTAssertEqual(status.body["current_model_memory"] as? Int64, 123_456)
        let models = try XCTUnwrap(status.body["models"] as? [[String: Any]])
        let tiny = try XCTUnwrap(models.first { $0["id"] as? String == "tiny" })
        XCTAssertEqual(tiny["model_type"] as? String, "audio_stt")
        XCTAssertEqual(tiny["loaded"] as? Bool, true)
        XCTAssertEqual(tiny["actual_size"] as? Int64, 123_456)

        let unload = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/tiny/unload", headers: [:], body: Data())
        )
        XCTAssertEqual(unload.status, 200)
        let unloadCalls = await adapter.unloadCallsForTest("tiny")
        let loadedAfterUnload = await adapter.isLoadedForTest("tiny")
        XCTAssertEqual(unloadCalls, 1)
        XCTAssertFalse(loadedAfterUnload)
    }

    func testSpeechLifecycleTracksDuplicateModelIDsPerAdapter() async throws {
        let registry = SpeechEngineRegistry()
        let first = FakeSpeechAdapter(
            engineID: "first",
            silicon: .ane,
            modelIDs: ["shared"],
            footprintBytes: 10
        )
        let second = FakeSpeechAdapter(
            engineID: "second",
            silicon: .cpu,
            modelIDs: ["shared"],
            footprintBytes: 20
        )
        await registry.register(first)
        await registry.register(second)
        let backend = await RegistrySpeechBackend(registry: registry)

        _ = try await backend.loadModel("second:shared")
        let statuses = await backend.speechModelStatuses()
        let loaded = statuses.filter { $0.loaded }
        let firstLoadCalls = await first.loadCallsForTest("shared")
        let secondLoadCalls = await second.loadCallsForTest("shared")

        XCTAssertEqual(loaded.count, 1)
        XCTAssertEqual(loaded.first?.modelPath, "speech://second/shared")
        XCTAssertEqual(firstLoadCalls, 0)
        XCTAssertEqual(secondLoadCalls, 1)
    }

    func testNamespacedSpeechLifecycleRemainsReachableWhenBareIDCollidesWithLLM() async throws {
        let adapter = FakeSpeechAdapter(
            engineID: "whisperkit",
            silicon: .ane,
            modelIDs: ["tiny"]
        )
        let server = try await makeServer(speechAdapter: adapter, llmModelIDs: ["tiny"])

        let response = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/whisperkit:tiny/load", headers: [:], body: Data())
        )

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(response.body["model_id"] as? String, "whisperkit:tiny")
        let loadCalls = await adapter.loadCallsForTest("tiny")
        let loaded = await adapter.isLoadedForTest("tiny")
        XCTAssertEqual(loadCalls, 1)
        XCTAssertTrue(loaded)
    }

    func testSpeechStatusReportsAdapterFootprintOnlyOnceForMultipleLoadedModels() async throws {
        let adapter = FakeSpeechAdapter(
            engineID: "whisperkit",
            silicon: .ane,
            modelIDs: ["base", "tiny"],
            footprintBytes: 123_456
        )
        let server = try await makeServer(speechAdapter: adapter)

        _ = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/base/load", headers: [:], body: Data())
        )
        _ = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/tiny/load", headers: [:], body: Data())
        )

        let status = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "GET", path: "/v1/models/status", headers: [:], body: Data())
        )
        XCTAssertEqual(status.status, 200)
        XCTAssertEqual(status.body["current_model_memory"] as? Int64, 123_456)
        let models = try XCTUnwrap(status.body["models"] as? [[String: Any]])
        let speechRows = models.filter { $0["model_type"] as? String == "audio_stt" }
        XCTAssertEqual(speechRows.filter { $0["loaded"] as? Bool == true }.count, 2)
        XCTAssertEqual(speechRows.compactMap { $0["actual_size"] as? Int64 }, [123_456])
    }

    func testUnknownSpeechLifecycleModelMapsTo404() async throws {
        let adapter = FakeSpeechAdapter(engineID: "whisperkit", silicon: .ane, modelIDs: ["tiny"])
        let server = try await makeServer(speechAdapter: adapter)

        let response = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/ghost/load", headers: [:], body: Data())
        )
        XCTAssertEqual(response.status, 404)
        let error = try XCTUnwrap(response.body["error"] as? [String: Any])
        XCTAssertEqual(error["type"] as? String, "not_found_error")
    }
}

private func makeServer(
    speechAdapter: FakeSpeechAdapter,
    llmModelIDs: [String] = ["alpha"]
) async throws -> OpenAIServer {
    let registry = SpeechEngineRegistry()
    await registry.register(speechAdapter)
    let speechBackend = await RegistrySpeechBackend(registry: registry)
    let pool = EnginePool(
        models: Dictionary(
            uniqueKeysWithValues: llmModelIDs.map {
                (
                    $0,
                    DiscoveredModel(
                        id: $0,
                        modelURL: URL(fileURLWithPath: "/models/\($0)", isDirectory: true),
                        estimatedSize: 100
                    )
                )
            }
        ),
        loader: NativeModelLoader(maxConcurrentRequests: 1),
        finalCeiling: 1_000
    )
    let backend = PoolBackedChatBackend(
        pool: pool,
        modelIDs: llmModelIDs,
        speechBackend: speechBackend
    )
    return try OpenAIServer(port: 0, backend: backend)
}
