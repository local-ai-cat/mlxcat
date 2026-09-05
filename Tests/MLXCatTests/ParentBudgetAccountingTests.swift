import Foundation
import MLXCat
import MLXCatHTTP
@testable import MLXCatNative
import Testing

@Suite("Parent budget allocation lifetimes", .timeLimit(.minutes(1)))
struct ParentBudgetAccountingTests {
    @Test func deniedModelNeverInvokesFactory() async throws {
        let parent = ParentBudgetProbe(limit: 50)
        let loader = ParentAccountingLoader()
        let pool = EnginePool(models: ["chat": parentBudgetModel()], loader: loader, parentBudget: parent)
        await #expect(throws: ParentBudgetFailure.self) { try await pool.acquire("chat") }
        #expect(await loader.loads == 0)
        #expect(await parent.used == 0)
        #expect(await pool.status().currentModelMemory == 0)
    }

    @Test func pendingModelIsChargedToParentBeforeLoaderCompletes() async throws {
        let gate = AccountingBarrier()
        let parent = ParentBudgetProbe(limit: 80)
        let loader = ParentAccountingLoader(loadGate: gate)
        let pool = EnginePool(models: ["chat": parentBudgetModel()], loader: loader, parentBudget: parent)
        let loading = Task { try await pool.acquire("chat") }
        await gate.waitUntilEntered()
        #expect(await parent.used == 60)
        #expect(await parent.loaded.isEmpty)
        #expect(await pool.status().currentModelMemory == 60)
        let competitor = EnginePool(models: ["chat": parentBudgetModel(30)], loader: ParentAccountingLoader(),
                                    parentBudget: parent)
        await #expect(throws: ParentBudgetFailure.self) { try await competitor.acquire("chat") }
        await gate.open()
        let lease = try await loading.value
        #expect(await parent.loaded.count == 1)
        await pool.release(lease)
        try await pool.unload("chat")
        #expect(await parent.used == 0)
    }

    @Test func usageLeasesNeverDoubleChargeResidentModel() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: ["chat": parentBudgetModel()], loader: ParentAccountingLoader(),
                              parentBudget: parent)
        let first = try await pool.acquire("chat")
        let second = try await pool.acquire("chat")
        #expect(await parent.allocations.count == 1)
        #expect(await parent.used == 60)
        await pool.release(first)
        await pool.release(first)
        #expect(await parent.used == 60)
        await pool.release(second)
        try await pool.unload("chat")
        #expect(await parent.releaseCalls == 1)
    }

    @Test func cancelledChatWaiterPreservesSharedLoadAndParentCharge() async throws {
        let gate = AccountingBarrier()
        let loader = ParentAccountingLoader(loadGate: gate)
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: ["chat": parentBudgetModel()], loader: loader, parentBudget: parent)
        let first = Task { try await pool.acquire("chat") }
        await gate.waitUntilEntered()
        let second = Task { try await pool.acquire("chat") }
        first.cancel()
        #expect(await parent.used == 60)
        await gate.open()
        await #expect(throws: CancellationError.self) { try await first.value }
        let lease = try await second.value
        #expect(await loader.loads == 1)
        #expect(await loader.loaderWasCancelled == false)
        #expect(await parent.allocations.count == 1)
        #expect(await parent.used == 60)
        await pool.release(lease)
        try await pool.unload("chat")
    }

    @Test func cancelledParentAdmissionReleasesBeforeAnyFactoryStarts() async throws {
        let gate = AccountingBarrier()
        let parent = ParentBudgetProbe(limit: 100, admissionGate: gate)
        let loader = ParentAccountingLoader()
        let pool = EnginePool(models: ["chat": parentBudgetModel()], loader: loader, parentBudget: parent)
        let loading = Task { try await pool.acquire("chat") }
        await gate.waitUntilEntered()
        loading.cancel()
        #expect(await parent.used == 60)
        #expect(await loader.loads == 0)
        await gate.open()
        await #expect(throws: CancellationError.self) { try await loading.value }
        #expect(await parent.used == 0)
        #expect(await parent.releaseCalls == 1)
        #expect(await loader.loads == 0)
    }

    @Test func modelUnloadRetainsBothChargesUntilAcknowledged() async throws {
        let gate = AccountingBarrier()
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: ["chat": parentBudgetModel()],
                              loader: ParentAccountingLoader(unloadGate: gate), parentBudget: parent)
        _ = try await pool.load("chat")
        let unloading = Task { try await pool.unload("chat") }
        await gate.waitUntilEntered()
        #expect(await parent.used == 60)
        #expect(await pool.status().currentModelMemory == 60)
        await #expect(throws: ParentBudgetFailure.self) { try await pool.reserveRequest(additionalBytes: 50) }
        await gate.open()
        try await unloading.value
        #expect(await parent.used == 0)
        #expect(await pool.status().currentModelMemory == 0)
        #expect(await parent.releaseCalls == 1)
    }

    @Test func measuredOvershootStaysChargedThroughRejectedLoadCleanup() async throws {
        let cleanup = AccountingBarrier()
        let parent = ParentBudgetProbe(limit: 50, rejectOvershoot: true)
        let loader = ParentAccountingLoader(unloadGate: cleanup, actual: 70)
        let pool = EnginePool(models: ["chat": parentBudgetModel(40)], loader: loader, parentBudget: parent)
        let loading = Task { try await pool.acquire("chat") }
        await cleanup.waitUntilEntered()
        #expect(await parent.used == 70)
        #expect(await pool.status().currentModelMemory == 70)
        #expect(await pool.status().models.first?.isLoading == true)
        await cleanup.open()
        await #expect(throws: ParentBudgetFailure.self) { try await loading.value }
        #expect(await parent.used == 0)
        #expect(await parent.releaseCalls == 1)
        #expect(await loader.unloads == 1)
        #expect(await pool.status().currentModelMemory == 0)
    }

    @Test func requestDenialDoesNotInvokeLazyFactory() async throws {
        let parent = ParentBudgetProbe(limit: 10)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let started = AccountingCounter()
        let prepared = PreparedNativeChat(start: { _ in
            started.increment()
            return OpenAIChatStream(promptTokens: 1, chunks: AsyncThrowingStream { $0.finish() })
        }, promptTokens: 1)
        await #expect(throws: ParentBudgetFailure.self) {
            try await prepared.admitAndStart(pool: pool, additionalBytes: 20, checkPressure: { _ in })
        }
        #expect(started.value == 0)
        #expect(await pool.reservedRequestBytes == 0)
    }

    @Test func requestReleaseWaitsForStreamTeardownAndIsIdempotent() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: ["chat": parentBudgetModel()], loader: ParentAccountingLoader(),
                              parentBudget: parent)
        let lease = try await pool.acquire("chat")
        let ticket = try await pool.admitWaitingRequest()
        let request = try await pool.reserveRequest(additionalBytes: 20)
        let teardown = AccountingBarrier()
        let pair = AsyncThrowingStream<OpenAIChatChunk, Error>.makeStream()
        let original = OpenAIChatStream(promptTokens: 1, chunks: pair.stream, cancelAndWait: {
            await teardown.wait()
            pair.continuation.finish()
        })
        let stream = PoolLeasedChatStream.make(original, pool: pool, lease: lease, ticket: ticket, reservation: request)
        let cancelled = Task { await stream.cancelAndWait() }
        await teardown.waitUntilEntered()
        #expect(await parent.used == 80)
        #expect(await pool.reservedRequestBytes == 20)
        await teardown.open()
        await cancelled.value
        await stream.cancelAndWait()
        await pool.releaseRequest(request)
        #expect(await parent.used == 60)
        #expect(await parent.releaseCalls == 1)
        try await pool.unload("chat")
        #expect(await parent.used == 0)
    }

    @Test func concurrentRequestReleaseJoinsAcknowledgement() async throws {
        let gate = AccountingBarrier()
        let parent = ParentBudgetProbe(limit: 100, releaseGate: gate)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let request = try await pool.reserveRequest(additionalBytes: 20)
        let first = Task { await pool.releaseRequest(request) }
        await gate.waitUntilEntered()
        let completed = AccountingCounter()
        let second = Task {
            await pool.releaseRequest(request)
            completed.increment()
        }
        // Queue an actor round trip after the second release enters the pool.
        for _ in 0..<100 { await Task.yield() }
        #expect(completed.value == 0)
        #expect(await parent.releaseCalls == 1)
        await gate.open()
        await first.value
        await second.value
        #expect(completed.value == 1)
        #expect(await parent.used == 0)
    }

}
