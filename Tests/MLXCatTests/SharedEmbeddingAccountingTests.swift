import Foundation
import MLXCat
@testable import MLXCatHTTP
@testable import MLXCatNative
@testable import MLXCatHTTPServer
import Testing

extension SharedModelAccountingTests {
    @Test func fourModelClassesShareAdmissionAndEmbeddingEvictsOnlyIdleSpeech() async throws {
        let pool = EnginePool(models: ["chat": model("chat", 25)], loader: AccountingChatLoader(), finalCeiling: 100)
        let chat = try await pool.acquire("chat")
        let speechWork = AccountingBarrier()
        let speech = await PoolBackedSpeechAdapter(
            adapter: AccountingSpeechAdapter(bytes: 20, work: speechWork), pool: pool
        )
        let speechTask = Task {
            try await speech.transcribeFile(.init(model: "speech", fileName: "fixture.wav", fileData: Data()))
        }
        await speechWork.waitUntilEntered()
        let rankWork = AccountingBarrier()
        let rerank = await NativeRerankBackend(models: ["rank": model("rank", 25)], pool: pool,
                                               cache: AccountingRerankCache(work: rankWork))
        let rankTask = Task { try await rerank.rerank(rerankRequest()) }
        await rankWork.waitUntilEntered()
        let embedWork = AccountingBarrier()
        let cache = AccountingEmbeddingCache(work: embedWork)
        let embedding = embeddingBackend(bytes: 35, pool: pool, cache: cache)
        await embedding.registerModel()
        #expect(await cache.loads == 0)
        do {
            _ = try await embedding.embed(embeddingRequest())
            Issue.record("Four individually fitting models exceeded the shared ceiling")
        } catch let error as OpenAIHTTPError { #expect(error.status == 507) }
        #expect(await cache.loads == 0)
        await speechWork.open()
        _ = try await speechTask.value
        let embedTask = Task { try await embedding.embed(embeddingRequest()) }
        await embedWork.waitUntilEntered()
        #expect(await pool.status().currentModelMemory == 85)
        #expect(await pool.resourceStatuses().first(where: { $0.resource.modelType == "audio_stt" })?.loaded == false)
        let state = await pool.resourceStatuses().first { $0.resource.modelType == "embedding" }
        #expect(state?.loaded == true)
        #expect(state?.inUse == 1)
        #expect(state?.resource.footprint == EnginePoolFootprint(bytes: 35, source: .configuredEstimate))
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 0)
        await embedWork.open()
        #expect(try await embedTask.value == AccountingEmbeddingCache.result)
        await rankWork.open()
        _ = try await rankTask.value
        await pool.release(chat)
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 85)
        #expect(await cache.unloads == 1)
    }

    @Test func embeddingPendingLoadIsChargedAndCancelledWaiterDoesNotCancelSharedFactory() async throws {
        let pool = emptyPool(100)
        let loading = AccountingBarrier()
        let cache = AccountingEmbeddingCache(loadGate: loading)
        let backend = embeddingBackend(bytes: 60, pool: pool, cache: cache)
        let first = Task { try await backend.embed(embeddingRequest()) }
        await loading.waitUntilEntered()
        #expect(await pool.status().currentModelMemory == 60)
        #expect(await pool.resourceStatuses().first?.isLoading == true)
        let second = Task { try await backend.embed(embeddingRequest()) }
        first.cancel()
        await loading.open()
        do {
            _ = try await first.value
            Issue.record("Cancelled embedding waiter received a result")
        } catch is CancellationError {}
        #expect(try await second.value == AccountingEmbeddingCache.result)
        #expect(await cache.loads == 1)
        #expect(await cache.factoryWasCancelled == false)
        #expect(await cache.requests == [embeddingRequest()])
        #expect(await pool.resourceStatuses().first?.inUse == 0)
    }

    @Test func embeddingLoadFailureRetainsChargeThroughCleanupAndCanRetry() async throws {
        let pool = emptyPool(100)
        let cleanup = AccountingBarrier()
        let cache = AccountingEmbeddingCache(unloadGate: cleanup, failFirstLoad: true)
        let backend = embeddingBackend(bytes: 40, pool: pool, cache: cache)
        let failed = Task { try await backend.embed(embeddingRequest()) }
        await cleanup.waitUntilEntered()
        #expect(await pool.status().currentModelMemory == 40)
        await cleanup.open()
        do {
            _ = try await failed.value
            Issue.record("Injected embedding load failure was swallowed")
        } catch is AccountingFailure {}
        #expect(await pool.status().currentModelMemory == 0)
        #expect(try await backend.embed(embeddingRequest()) == AccountingEmbeddingCache.result)
        #expect(await cache.loads == 2)
        #expect(await cache.requests.count == 1)
    }

    @Test func cancelledEmbeddingScoringAndExplicitUnloadRetainBytesUntilAcknowledged() async throws {
        let pool = emptyPool(100)
        let work = AccountingBarrier()
        let unloading = AccountingBarrier()
        let cache = AccountingEmbeddingCache(work: work, unloadGate: unloading)
        let backend = embeddingBackend(bytes: 60, pool: pool, cache: cache)
        let router = PoolResourceModelLifecycle(pool: pool)
        let embedding = Task { try await backend.embed(embeddingRequest()) }
        await work.waitUntilEntered()
        embedding.cancel()
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 0)
        do {
            _ = try await router.unload("embedding://embed", allowBareModelID: true)
            Issue.record("Explicit unload evicted an active embedding model")
        } catch let error as OpenAIHTTPError { #expect(error.status == 409) }
        #expect(await pool.resourceStatuses().first?.inUse == 1)
        await work.open()
        do {
            _ = try await embedding.value
            Issue.record("Cancelled scoring delivered a result")
        } catch is CancellationError {}
        #expect(await pool.resourceStatuses().first?.inUse == 0)
        let unload = Task { try await router.unload("embedding://embed", allowBareModelID: true) }
        await unloading.waitUntilEntered()
        #expect(await pool.status().currentModelMemory == 60)
        #expect(await pool.resourceStatuses().first?.isUnloading == true)
        await unloading.open()
        #expect(try await unload.value?.modelID == "embedding://embed")
        #expect(await pool.status().currentModelMemory == 0)
        _ = try await router.unload("embedding://embed", allowBareModelID: true)
        #expect(await cache.unloads == 1)
    }

    @Test func embeddingDescriptorAndExplicitLifecyclePreserveChatNamespacing() async throws {
        let pool = emptyPool(100)
        let cache = AccountingEmbeddingCache()
        let backend = embeddingBackend(bytes: 30, pool: pool, cache: cache)
        await backend.registerModel()
        #expect(await cache.loads == 0)
        #expect(await pool.resourceStatuses().first?.loaded == false)
        let router = PoolResourceModelLifecycle(pool: pool)
        #expect(try await router.load("embed", allowBareModelID: false) == nil)
        #expect(try await router.load("embedding://embed", allowBareModelID: false)?.modelID == "embedding://embed")
        #expect(await cache.loads == 1)
        #expect(await pool.resourceStatuses().first?.inUse == 0)
        _ = try await router.unload("embed", allowBareModelID: true)
        #expect(await cache.unloads == 1)
        _ = try await router.load("embed", allowBareModelID: true)
        #expect(await cache.loads == 2)
        #expect(try await router.load("absent", allowBareModelID: true) == nil)
        let other = AccountingResourceState()
        await pool.registerResource(EnginePoolResource(
            id: "speech://fixture/embed", modelID: "embed", modelType: "audio_stt",
            footprint: EnginePoolFootprint(bytes: 10, source: .configuredEstimate),
            usesMLXAllocator: false, load: { try await other.load() }, unload: { await other.unload() }
        ))
        #expect(try await router.load("embed", allowBareModelID: true) == nil)
        #expect(try await router.load("embedding://embed", allowBareModelID: true) != nil)
        #expect(await other.loads == 0)
    }

    @Test func invalidAndUnbudgetableEmbeddingRequestsNeverInvokeFactory() async throws {
        let footprints: [Int64?] = [nil, 101]
        for bytes in footprints {
            let pool = emptyPool(100)
            let cache = AccountingEmbeddingCache()
            let backend = embeddingBackend(bytes: bytes, pool: pool, cache: cache)
            do {
                _ = try await backend.embed(.init(input: .string("input"), model: "absent"))
                Issue.record("Invalid embedding model reached the factory")
            } catch let error as OpenAIHTTPError { #expect(error.status == 404) }
            do {
                _ = try await backend.embed(embeddingRequest())
                Issue.record("Unbudgetable embedding model reached the factory")
            } catch let error as OpenAIHTTPError { #expect(error.status == 507) }
            #expect(await cache.loads == 0)
            #expect(await pool.status().currentModelMemory == 0)
        }
    }

    @Test func configuredEmbeddingDirectoryIsExcludedFromChatDiscoveryEvenThroughSymlink() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["chat", "embed"] {
            let directory = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data("{\"model_type\":\"bert\"}".utf8).write(to: directory.appendingPathComponent("config.json"))
            try Data(repeating: 0, count: 64).write(to: directory.appendingPathComponent("fixture.safetensors"))
        }
        let models = try ModelDiscovery.discoverModels(in: root)
        #expect(Set(models.keys) == Set(["chat", "embed"]))
        let alias = root.appendingPathComponent("embedding-alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: root.appendingPathComponent("embed"))
        let chatModels = ModelDiscovery.excludingModelDirectory(models, directory: alias)
        #expect(Array(chatModels.keys) == ["chat"])
        #expect(ModelDiscovery.excludingModelDirectory(models, directory: nil) == models)
    }

    @Test func encodedEmbeddingLifecycleReferenceIsDecodedExactlyOnce() throws {
        let raw = Data("POST /v1/models/embedding%3A%2F%2Fembed/load HTTP/1.1\r\nHost: localhost\r\n\r\n".utf8)
        let request = try HTTPRequest.parseComplete(raw)
        #expect(OpenAIServer.lifecycleModelID(from: request.path, action: "load") == "embedding://embed")
        #expect(OpenAIServer.lifecycleModelID(from: "/v1/models/embedding://embed/unload", action: "unload")
                    == "embedding://embed")
        #expect(OpenAIServer.lifecycleModelID(from: "/v1/models/name%252Fpart/load", action: "load") == "name%2Fpart")
        #expect(OpenAIServer.lifecycleModelID(from: "/v1/models/bad%/load", action: "load") == nil)
    }

    private func embeddingBackend(bytes: Int64?, pool: any EnginePoolResourceManaging,
                                  cache: AccountingEmbeddingCache) -> NativeEmbeddingsBackend {
        NativeEmbeddingsBackend(modelURL: URL(fileURLWithPath: "/fixture/embed"),
                                footprint: EnginePoolFootprint(bytes: bytes, source: .configuredEstimate), pool: pool,
                                cache: cache)
    }

    private func embeddingRequest() -> OpenAIEmbeddingsRequest {
        OpenAIEmbeddingsRequest(input: .strings(["first", "second"]), model: "embed", dimensions: 1)
    }
}

actor AccountingEmbeddingCache: NativeEmbeddingModelServing {
    static let result = OpenAIEmbeddingsResult(embeddings: [[1, 2], [3, 4]], promptTokens: 7)
    let loadGate: AccountingBarrier?
    let work: AccountingBarrier?
    let unloadGate: AccountingBarrier?
    let failFirstLoad: Bool
    private(set) var loads = 0
    private(set) var unloads = 0
    private(set) var factoryWasCancelled = false
    private(set) var requests: [OpenAIEmbeddingsRequest] = []

    init(loadGate: AccountingBarrier? = nil, work: AccountingBarrier? = nil,
         unloadGate: AccountingBarrier? = nil, failFirstLoad: Bool = false) {
        self.loadGate = loadGate
        self.work = work
        self.unloadGate = unloadGate
        self.failFirstLoad = failFirstLoad
    }

    func load(modelURL: URL) async throws {
        loads += 1
        await loadGate?.wait()
        factoryWasCancelled = Task.isCancelled
        if failFirstLoad && loads == 1 { throw AccountingFailure.injected }
    }

    func unload() async {
        unloads += 1
        await unloadGate?.wait()
    }

    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> OpenAIEmbeddingsResult {
        requests.append(request)
        await work?.wait()
        return Self.result
    }
}
