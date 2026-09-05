import Foundation
import MLXCat
import MLXCatHTTP
import MLXCatSpeech
@testable import MLXCatNative
@testable import MLXCatHTTPServer
import Testing

extension SharedModelAccountingTests {
    @Test func pressureKeepsUnattributedGrowthConservativeAndPendingGrowthAdditive() async throws {
        // No allocation telemetry exists: even materialized KV cannot safely be subtracted.
        let pool = EnginePool(models: ["chat": model("chat", 40)], loader: AccountingChatLoader(), finalCeiling: 100)
        let lease = try await pool.acquire("chat")
        let active = try await pool.reserveRequest(additionalBytes: 30)
        let watchdog = MemoryWatchdog(configuration: MemoryWatchdogConfiguration(ceilingBytes: 100),
                                      sampler: { 70 }, reclaimer: ClosureMemoryWatchdogReclaimer(
                                          trimReclaimableCaches: { _ in 0 }, evictIdleModels: { _ in 0 }
                                      ))
        let starts = AccountingCounter()
        let prepared = PreparedNativeChat(start: { _ in
            starts.increment()
            return OpenAIChatStream(promptTokens: 1, chunks: AsyncThrowingStream { $0.finish() })
        }, promptTokens: 1)
        do {
            _ = try await prepared.admitAndStart(pool: pool, additionalBytes: 10,
                                                 checkPressure: {
                                                 try await watchdog.checkAdmission(additionalBytes: $0) })
            Issue.record("Unattributed growth was silently assumed materialized")
        } catch is MemoryWatchdogError {}
        #expect(await pool.reservedRequestBytes == 30)
        #expect(starts.value == 0)
        await pool.releaseRequest(active)
        await pool.release(lease)

        // Observed excess (e.g. embeddings/cache) is additive to a pending allocation.
        // max(observed40, pool reservation20) would incorrectly fit a ceiling of50.
        let pendingPool = emptyPool(50)
        let pendingWatchdog = MemoryWatchdog(configuration: MemoryWatchdogConfiguration(ceilingBytes: 50),
                                             sampler: { 40 }, reclaimer: ClosureMemoryWatchdogReclaimer(
                                                 trimReclaimableCaches: { _ in 0 }, evictIdleModels: { _ in 0 }
                                             ))
        do {
            _ = try await prepared.admitAndStart(pool: pendingPool, additionalBytes: 20,
                                                 checkPressure: {
                                                 try await pendingWatchdog.checkAdmission(additionalBytes: $0) })
            Issue.record("Observed excess plus pending growth exceeded the pressure ceiling")
        } catch is MemoryWatchdogError {}
        #expect(await pendingPool.reservedRequestBytes == 0)
        #expect(starts.value == 0)
    }

    @Test func pinChangedDuringChatUnloadSurvivesAcknowledgement() async throws {
        let barrier = AccountingBarrier()
        let pool = EnginePool(models: ["chat": model("chat", 40)],
                              loader: AccountingChatLoader(unloadGate: barrier), finalCeiling: 100)
        let lease = try await pool.acquire("chat")
        await pool.release(lease)
        let unload = Task { try await pool.unload("chat") }
        await barrier.waitUntilEntered()
        try await pool.setPinned(true, for: "chat")
        await barrier.open()
        try await unload.value
        #expect(await pool.status().models.first?.pinned == true)
    }

    @Test func chatFactoryStartsOnlyAfterSharedAdmissionAndRollsBackPressureDenial() async throws {
        let pool = EnginePool(models: ["chat": model("chat", 40)], loader: AccountingChatLoader(), finalCeiling: 100)
        let lease = try await pool.acquire("chat")
        let starts = AccountingCounter()
        let prepared = PreparedNativeChat(start: { _ in
            starts.increment()
            return OpenAIChatStream(promptTokens: 1, chunks: AsyncThrowingStream { $0.finish() })
        }, promptTokens: 1)
        do {
            _ = try await prepared.admitAndStart(pool: pool, additionalBytes: 70, checkPressure: { _ in })
            Issue.record("Oversized request started")
        } catch let error as EnginePoolError { #expect(error.httpStatus == 507) }
        do {
            _ = try await prepared.admitAndStart(pool: pool, additionalBytes: 35,
                                                 checkPressure: { _ in throw AccountingFailure.injected })
            Issue.record("Pressure-denied request started")
        } catch AccountingFailure.injected {}
        #expect(starts.value == 0)
        #expect(await pool.reservedRequestBytes == 0)
        let admitted = try await prepared.admitAndStart(
            pool: pool, additionalBytes: 35, checkPressure: { _ in }
        )
        #expect(starts.value == 1)
        #expect(await pool.reservedRequestBytes == 35)
        await admitted.stream.cancelAndWait()
        await pool.releaseRequest(admitted.reservation)
        await pool.release(lease)
    }

    @Test func cancellationDuringChatPressureCheckReleasesGrowthWithoutStarting() async throws {
        let pool = emptyPool(100)
        let pressure = AccountingBarrier()
        let starts = AccountingCounter()
        let prepared = PreparedNativeChat(start: { _ in
            starts.increment()
            return OpenAIChatStream(promptTokens: 1, chunks: AsyncThrowingStream { $0.finish() })
        }, promptTokens: 1)
        let admission = Task {
            try await prepared.admitAndStart(pool: pool, additionalBytes: 35,
                                             checkPressure: { _ in await pressure.wait() })
        }
        await pressure.waitUntilEntered()
        #expect(await pool.reservedRequestBytes == 35)
        admission.cancel()
        await pressure.open()
        do {
            _ = try await admission.value
            Issue.record("Cancelled preparation started generation")
        } catch is CancellationError {}
        #expect(starts.value == 0)
        #expect(await pool.reservedRequestBytes == 0)
    }

    @Test func chatCancellationRetainsLeaseUntilProducerTeardownAcknowledges() async throws {
        let pool = EnginePool(models: ["chat": model("chat", 40)], loader: AccountingChatLoader(), finalCeiling: 100)
        let lease = try await pool.acquire("chat")
        let reservation = try await pool.reserveRequest(additionalBytes: 35)
        let ticket = try await pool.admitWaitingRequest()
        let teardown = AccountingBarrier()
        let (chunks, continuation) = AsyncThrowingStream<OpenAIChatChunk, Error>.makeStream()
        let producer = OpenAIChatStream(promptTokens: 1, chunks: chunks, cancelAndWait: {
            await teardown.wait()
            continuation.finish()
        })
        let stream = PoolLeasedChatStream.make(producer, pool: pool, lease: lease,
                                               ticket: ticket, reservation: reservation)
        let first = Task { await stream.cancelAndWait() }
        await teardown.waitUntilEntered()
        let second = Task { await stream.cancelAndWait() }
        #expect(await pool.status().models.first?.inUse == 1)
        #expect(await pool.reservedRequestBytes == 35)
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 0)
        await teardown.open()
        await first.value
        await second.value
        #expect(await teardown.entries == 1)
        #expect(await pool.reservedRequestBytes == 0)
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 40)
    }

    @Test(arguments: [false, true])
    func httpConsumptionClosesProducerAfterEarlyBreakOrWriteFailure(failWrite: Bool) async throws {
        let pool = EnginePool(models: ["chat": model("chat", 40)], loader: AccountingChatLoader(), finalCeiling: 100)
        let lease = try await pool.acquire("chat")
        let ticket = try await pool.admitWaitingRequest()
        let teardown = AccountingBarrier()
        let (chunks, continuation) = AsyncThrowingStream<OpenAIChatChunk, Error>.makeStream()
        let producer = OpenAIChatStream(promptTokens: 1, chunks: chunks, cancelAndWait: {
            await teardown.wait()
            continuation.finish()
        })
        let stream = PoolLeasedChatStream.make(producer, pool: pool, lease: lease,
                                               ticket: ticket, reservation: nil)
        let consumer = Task {
            try await stream.withLifetime { stream in
                for try await _ in stream.chunks {
                    if failWrite { throw AccountingFailure.injected }
                    break
                }
            }
        }
        continuation.yield(OpenAIChatChunk(text: "stop", tokenID: 1))
        await teardown.waitUntilEntered()
        #expect(await pool.status().models.first?.inUse == 1)
        await teardown.open()
        do {
            try await consumer.value
            #expect(!failWrite)
        } catch is AccountingFailure { #expect(failWrite) }
        #expect(await pool.status().models.first?.inUse == 0)
        #expect(await teardown.entries == 1)
    }

    @Test func oversizedRerankerReturns507BeforeInvokingTheFactory() async throws {
        let pool = emptyPool(100)
        let cache = AccountingRerankCache()
        let backend = await NativeRerankBackend(models: ["rank": model("rank", 101)], pool: pool, cache: cache)
        do {
            _ = try await backend.rerank(rerankRequest())
            Issue.record("Oversized reranker loaded")
        } catch let error as OpenAIHTTPError {
            #expect(error.status == 507)
            let expected = EnginePoolError.modelTooLarge(id: "rerank://rank", size: 101, ceiling: 100)
            #expect(error.message == expected.description)
        }
        #expect(await cache.loads == 0)
        #expect(await pool.status().currentModelMemory == 0)
    }

    @Test func cancellingRerankDoesNotFreeItsModelUntilScoringReturns() async throws {
        let pool = emptyPool(100)
        let scoring = AccountingBarrier()
        let cache = AccountingRerankCache(work: scoring)
        let backend = await NativeRerankBackend(models: ["rank": model("rank", 60)], pool: pool, cache: cache)
        let request = Task { try await backend.rerank(rerankRequest()) }
        await scoring.waitUntilEntered()
        request.cancel()
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 0)
        #expect(await pool.resourceStatuses().first?.inUse == 1)
        await scoring.open()
        do { _ = try await request.value; Issue.record("Cancelled scoring succeeded") } catch is CancellationError {}
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 60)
        #expect(await cache.unloads == 1)
    }

    @Test func streamingCancelWaitsForActivePushAndReleasesExactlyOnce() async throws {
        let pool = emptyPool(100)
        let work = AccountingBarrier()
        let adapter = AccountingSpeechAdapter(bytes: 60, work: work)
        let speech = await PoolBackedSpeechAdapter(adapter: adapter, pool: pool)
        let session = try await speech.makeStreamSession(modelID: "speech", language: nil)
        let push = Task { try await session.push(SpeechPCMBuffer(samples: [0], sampleRate: 16_000)) }
        await work.waitUntilEntered()
        await session.cancel()
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 0)
        await work.open()
        try await push.value
        await session.cancel()
        try await session.finish()
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 60)
        #expect(await adapter.unloads == 1)
    }

}
