import Foundation
import MLXCat
import MLXCatHTTP
import MLXCatSpeech
@testable import MLXCatNative
@testable import MLXCatHTTPServer
import Testing

@Suite("Shared chat, speech and rerank accounting", .timeLimit(.minutes(1)))
struct SharedModelAccountingTests {
    @Test func mixedAdmissionProtectsActiveWorkAndEvictsIdleSpeech() async throws {
        let pool = EnginePool(models: ["chat": model("chat", 45)], loader: AccountingChatLoader(), finalCeiling: 100)
        let chat = try await pool.acquire("chat")
        let speechWork = AccountingBarrier()
        let adapter = AccountingSpeechAdapter(bytes: 30, work: speechWork)
        let speech = await PoolBackedSpeechAdapter(adapter: adapter, pool: pool)
        let registry = SpeechEngineRegistry()
        await registry.register(speech)
        let route = await RegistrySpeechBackend(registry: registry)
        let transcription = Task { try await route.transcribe(audioRequest()) }
        await speechWork.waitUntilEntered()
        let states = await route.speechModelStatuses()
        #expect(states.first?.loaded == true)
        #expect(states.first?.inUse == 1)
        #expect(states.first?.accountedBytes == 30)
        #expect(states.first?.actualSize == nil)
        #expect(states.first?.footprintSource == "configuredEstimate")
        let cache = AccountingRerankCache()
        let reranker = await NativeRerankBackend(models: ["rank": model("rank", 35)], pool: pool, cache: cache)
        do {
            _ = try await reranker.rerank(rerankRequest())
            Issue.record("A third individually fitting model exceeded the shared total")
        } catch let error as OpenAIHTTPError { #expect(error.status == 507) }
        #expect(await cache.loads == 0)
        #expect(await pool.status().currentModelMemory == 75)
        await speechWork.open()
        _ = try await transcription.value
        _ = try await reranker.rerank(rerankRequest())
        #expect(await adapter.unloads == 1)
        #expect(await pool.status().currentModelMemory == 80)
        #expect(try await pool.isLoaded("chat"))
        await pool.release(chat)
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 80)
        #expect(await pool.status().currentModelMemory == 0)
    }

    @Test func sharedLoadReservesPendingBytesAndOneCancelledWaiterDoesNotCancelIt() async throws {
        let pool = emptyPool(100)
        let loading = AccountingBarrier()
        let state = AccountingResourceState(loadGate: loading)
        await pool.registerResource(resource("speech", 60, state))
        let cancelled = Task { try await pool.acquireResource("speech") }
        await loading.waitUntilEntered()
        #expect(await pool.status().currentModelMemory == 60)
        #expect(await pool.nonMLXAccountedBytes() == 60)
        #expect(await pool.resourceStatuses().first?.isLoading == true)
        let other = Task { try await pool.acquireResource("speech") }
        cancelled.cancel()
        await loading.open()
        do {
            _ = try await cancelled.value
            Issue.record("Cancelled waiter received a lease")
        } catch is CancellationError {}
        let lease = try await other.value
        #expect(await state.loads == 1)
        #expect(await state.loaderWasCancelled == false)
        #expect(await pool.resourceStatuses().first?.inUse == 1)
        await pool.releaseResource(lease)
        await pool.releaseResource(lease)
        try await pool.unloadResource("speech")
        #expect(await state.unloads == 1)
        #expect(await pool.status().currentModelMemory == 0)
    }

    @Test func unloadingStaysChargedUntilAcknowledged() async throws {
        let pool = emptyPool(100)
        let unloading = AccountingBarrier()
        let state = AccountingResourceState(unloadGate: unloading)
        await pool.registerResource(resource("speech", 60, state))
        let lease = try await pool.acquireResource("speech")
        await pool.releaseResource(lease)
        let unload = Task { try await pool.unloadResource("speech") }
        await unloading.waitUntilEntered()
        #expect(await pool.status().currentModelMemory == 60)
        #expect(await pool.nonMLXAccountedBytes() == 60)
        #expect(await pool.resourceStatuses().first?.isUnloading == true)
        do {
            _ = try await pool.reserveRequest(additionalBytes: 50)
            Issue.record("Unacknowledged teardown was treated as free memory")
        } catch let error as EnginePoolError { #expect(error.httpStatus == 507) }
        await unloading.open()
        try await unload.value
        let request = try await pool.reserveRequest(additionalBytes: 100)
        await pool.releaseRequest(request)
        await pool.releaseRequest(request)
        #expect(await pool.reservedRequestBytes == 0)
        #expect(await pool.status().currentModelMemory == 0)
    }

    @Test func failedLoadKeepsReservationThroughPartialObjectCleanupThenRetries() async throws {
        let pool = emptyPool(100)
        let cleanup = AccountingBarrier()
        let state = AccountingResourceState(unloadGate: cleanup, failFirstLoad: true)
        await pool.registerResource(resource("rank", 40, state))
        let failed = Task { try await pool.acquireResource("rank") }
        await cleanup.waitUntilEntered()
        #expect(await pool.status().currentModelMemory == 40)
        await cleanup.open()
        do {
            _ = try await failed.value
            Issue.record("Injected load failure disappeared")
        } catch AccountingFailure.injected {}
        #expect(await pool.status().currentModelMemory == 0)
        let retry = try await pool.acquireResource("rank")
        #expect(await state.loads == 2)
        await pool.releaseResource(retry)
        try await pool.unloadResource("rank")
        #expect(await pool.status().currentModelMemory == 0)
    }

    @Test func unknownIsDeniedWithFiniteCeilingButZeroAndUnboundedRemainSupported() async throws {
        let state = AccountingResourceState()
        let pool = emptyPool(100)
        await pool.registerResource(resource("unknown", nil, state))
        do {
            _ = try await pool.acquireResource("unknown")
            Issue.record("Unknown was charged as zero")
        } catch let error as EnginePoolError { #expect(error == .unknownFootprint(id: "unknown")) }
        #expect(await state.loads == 0)
        #expect(await pool.resourceStatuses().first?.resource.footprint.source == .unknown)
        await pool.registerResource(resource("zero", 0, state))
        let zero = try await pool.acquireResource("zero")
        await pool.releaseResource(zero)
        let unbounded = emptyPool(0)
        await unbounded.registerResource(resource("unknown", nil, state))
        let unknown = try await unbounded.acquireResource("unknown")
        #expect(await unbounded.resourceStatuses().first?.resource.footprint.bytes == nil)
        await unbounded.releaseResource(unknown)
    }

    @Test func activeChatGrowthCompetesWithAuxiliaryLoadsAndConcurrentRequests() async throws {
        let pool = EnginePool(models: ["chat": model("chat", 40)], loader: AccountingChatLoader(), finalCeiling: 100)
        let chat = try await pool.acquire("chat")
        let growth = try await pool.reserveRequest(additionalBytes: 35)
        let state = AccountingResourceState()
        await pool.registerResource(resource("rank", 30, state))
        do {
            _ = try await pool.acquireResource("rank")
            Issue.record("Model ignored projected chat growth")
        } catch let error as EnginePoolError { #expect(error.httpStatus == 507) }
        do {
            _ = try await pool.reserveRequest(additionalBytes: 30)
            Issue.record("Requests spent the same headroom")
        } catch let error as EnginePoolError { #expect(error.httpStatus == 507) }
        #expect(await state.loads == 0)
        await pool.releaseRequest(growth)
        let rank = try await pool.acquireResource("rank")
        await pool.releaseResource(rank)
        await pool.release(chat)
    }

    @Test func simultaneousRequestAdmissionsCannotSpendTheSameHeadroom() async throws {
        let pool = EnginePool(models: ["chat": model("chat", 40)], loader: AccountingChatLoader(), finalCeiling: 100)
        let chat = try await pool.acquire("chat")
        let tokens = await withTaskGroup(of: UUID?.self) { group in
            for _ in 0..<3 {
                group.addTask { try? await pool.reserveRequest(additionalBytes: 35) }
            }
            var granted: [UUID] = []
            for await token in group { if let token { granted.append(token) } }
            return granted
        }
        #expect(tokens.count == 1)
        #expect(await pool.reservedRequestBytes == 35)
        for token in tokens { await pool.releaseRequest(token) }
        await pool.release(chat)
    }

    @Test func sharedLRUReclaimsOlderChatBeforeIdleAuxiliaryAndHonorsChatPin() async throws {
        let pool = EnginePool(models: ["chat": model("chat", 40)], loader: AccountingChatLoader(), finalCeiling: 100)
        let chat = try await pool.acquire("chat", now: Date(timeIntervalSince1970: 1))
        let state = AccountingResourceState()
        await pool.registerResource(resource("speech", 30, state))
        let speech = try await pool.acquireResource("speech")
        await pool.releaseResource(speech)
        await pool.release(chat)
        #expect(await pool.reclaimIdleModels(targetBytes: 1) == 40)
        #expect(try await pool.isLoaded("chat") == false)
        #expect(await state.unloads == 0)
        _ = try await pool.load("chat", now: Date(timeIntervalSince1970: 1))
        try await pool.setPinned(true, for: "chat")
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 30)
        #expect(try await pool.isLoaded("chat"))
        #expect(await state.unloads == 1)
    }

    @Test func pressureSamplerDoesNotDoubleCountNonMLXPendingBytes() async throws {
        let pool = emptyPool(100)
        let watchdog = MemoryWatchdog(configuration: MemoryWatchdogConfiguration(ceilingBytes: 100),
                                      sampler: { await pool.nonMLXAccountedBytes() },
                                      reclaimer: ClosureMemoryWatchdogReclaimer(trimReclaimableCaches: { _ in 0 },
                                                                                evictIdleModels: {
                                                                                await pool
                                                                                .reclaimIdleModels(targetBytes: $0) }))
        await pool.setPressureAdmission { [weak watchdog] in
            try await watchdog?.checkAdmission(additionalBytes: $0)
        }
        let state = AccountingResourceState()
        await pool.registerResource(resource("speech", 60, state))
        let speech = try await pool.acquireResource("speech")
        #expect(await state.loads == 1)
        #expect(await pool.nonMLXAccountedBytes() == 60)
        await pool.releaseResource(speech)
    }

}

extension SharedModelAccountingTests {
    func emptyPool(_ ceiling: Int64) -> EnginePool<AccountingChatLoader> {
        EnginePool(models: [:], loader: AccountingChatLoader(), finalCeiling: ceiling)
    }

    func model(_ id: String, _ bytes: Int64) -> DiscoveredModel {
        DiscoveredModel(id: id, modelURL: URL(fileURLWithPath: "/unused/\(id)"), estimatedSize: bytes)
    }

    func resource(_ id: String, _ bytes: Int64?, _ state: AccountingResourceState) -> EnginePoolResource {
        EnginePoolResource(id: id, modelID: id, modelType: "test", footprint: EnginePoolFootprint(
            bytes: bytes, source: .configuredEstimate
        ), usesMLXAllocator: false,
                           load: { try await state.load() }, unload: { await state.unload() })
    }

    func rerankRequest() -> OpenAIRerankRequest {
        OpenAIRerankRequest(model: "rank", query: .string("query"), documents: [.string("document")])
    }

    func audioRequest() -> AudioTranscriptionRequest {
        AudioTranscriptionRequest(model: "speech", fileName: "fixture.wav", fileData: Data())
    }
}
