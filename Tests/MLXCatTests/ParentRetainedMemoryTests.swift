import Foundation
import MLXCat
import MLXCatHTTP
@testable import MLXCatNative
import Testing

@Suite("Parent retained memory ownership", .timeLimit(.minutes(1)))
struct ParentRetainedMemoryTests {
    @Test func lazyNativeFactoryReceivesItsAdmittedCacheOwnership() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let cache = CapturedCachePayload()
        let prepared = PreparedNativeChat(start: { allocation in
            try await cache.store(try #require(allocation))
            return OpenAIChatStream(promptTokens: 1, chunks: AsyncThrowingStream { $0.finish() })
        }, promptTokens: 1)
        let result = try await prepared.admitAndStart(pool: pool, additionalBytes: 40, checkPressure: { _ in })
        await result.stream.cancelAndWait()
        await pool.releaseRequest(result.reservation)
        #expect(await parent.used == 40)
        await cache.discard()
        #expect(await parent.used == 0)
    }

    @Test func cachedPayloadKeepsChargeAcrossCompletionAndReplacement() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let first = try await pool.reserveRequest(additionalBytes: 60)
        let allocation = try #require(await pool.retainedAllocation(for: first))
        let slot = try RetainedCachePayload(ReferencePayload(), allocation: allocation)
        let anotherSlot = try RetainedCachePayload(ReferencePayload(), allocation: allocation)
        await pool.releaseRequest(first)
        #expect(await parent.used == 60)
        #expect(await parent.allocations.count == 1)
        #expect(await parent.loaded.count == 1)
        await #expect(throws: ParentBudgetFailure.self) { try await pool.reserveRequest(additionalBytes: 50) }
        slot.discard()
        await allocation.waitForReleaseIfUnused()
        #expect(await parent.used == 60)
        anotherSlot.discard()
        await allocation.waitForReleaseIfUnused()
        #expect(await parent.used == 0)
        for _ in 0..<20 {
            let next = try await pool.reserveRequest(additionalBytes: 60)
            let nextAllocation = try #require(await pool.retainedAllocation(for: next))
            let nextSlot = try RetainedCachePayload(ReferencePayload(), allocation: nextAllocation)
            await pool.releaseRequest(next)
            #expect(await parent.used == 60)
            nextSlot.discard()
            await nextAllocation.waitForReleaseIfUnused()
            #expect(await parent.used == 0)
        }
    }

    @Test func retainedSnapshotExtentsRecordDebtBeforeCompletion() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let request = try await pool.reserveRequest(additionalBytes: 40)
        let allocation = try #require(await pool.retainedAllocation(for: request))
        let first = try RetainedCachePayload(ReferencePayload(), allocation: allocation, bytes: 60)
        let second = try RetainedCachePayload(ReferencePayload(), allocation: allocation, bytes: 70)
        await pool.releaseRequest(request)
        #expect(await parent.used == 130)
        #expect(await pool.reservedRequestBytes == 130)
        await #expect(throws: ParentBudgetFailure.self) { try await pool.reserveRequest(additionalBytes: 1) }
        first.discard()
        second.discard()
        await allocation.waitForReleaseIfUnused()
        #expect(await parent.used == 0)
    }

    @Test func unknownRequestProjectionIsNotZeroForParent() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        await #expect(throws: ParentBudgetFailure.self) { try await pool.reserveRequest(additionalBytes: nil) }
        #expect(await pool.reservedRequestBytes == 0)
        #expect(await parent.allocations.isEmpty)
    }

    @Test func cachedPayloadIsDiscardedBeforeParentReleaseAcknowledgement() async throws {
        let gate = AccountingBarrier()
        let parent = ParentBudgetProbe(limit: 100, releaseGate: gate)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let request = try await pool.reserveRequest(additionalBytes: 40)
        let allocation = try #require(await pool.retainedAllocation(for: request))
        let freed = AccountingCounter()
        let slot = try RetainedCachePayload(ReferencePayload(onDeinit: { freed.increment() }), allocation: allocation)
        await pool.releaseRequest(request)
        #expect(freed.value == 0)
        slot.discard()
        await gate.waitUntilEntered()
        #expect(freed.value == 1)
        #expect(slot.value == nil)
        #expect(await parent.used == 40)
        await gate.open()
        await allocation.waitForReleaseIfUnused()
        #expect(await parent.used == 0)
    }

    @Test func discardedSlotDoesNotReleaseAnActiveBorrowedHit() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let request = try await pool.reserveRequest(additionalBytes: 40)
        let allocation = try #require(await pool.retainedAllocation(for: request))
        let slot = try RetainedCachePayload(ReferencePayload(), allocation: allocation)
        let hit = try allocation.retainCache()
        await pool.releaseRequest(request)
        slot.discard()
        await allocation.waitForReleaseIfUnused()
        #expect(await parent.used == 40)
        hit.releaseAfterTeardown()
        await allocation.waitForReleaseIfUnused()
        #expect(await parent.used == 0)
    }

    @Test func nativeOwnerInvalidatesPayloadDespiteRetainedReleasedHandle() async throws {
        let parentGate = AccountingBarrier()
        let parent = ParentBudgetProbe(limit: 100, releaseGate: parentGate)
        let loader = ReferenceOwnerLoader()
        let pool = EnginePool(models: ["chat": parentBudgetModel()], loader: loader, parentBudget: parent)
        let lease = try await pool.acquire("chat")
        let staleHandle = lease.engine
        #expect(await loader.payloadIsAlive)
        await pool.release(lease)
        let unload = Task { try await pool.unload("chat") }
        await parentGate.waitUntilEntered()
        #expect(await loader.payloadIsAlive == false)
        #expect(await parent.used == 60)
        await #expect(throws: ResourceOwnerError.self) {
            try await staleHandle.withValue { _ in true }
        }
        await parentGate.open()
        try await unload.value
        #expect(await parent.used == 0)
        _ = staleHandle // The invalidated public handle remains strongly referenced.
    }

    @Test func invalidationWaitsForActiveOperationAndConcurrentCallers() async throws {
        let borrowed = AccountingBarrier()
        let teardown = AccountingBarrier()
        let owner = AsyncResourceOwner(ReferencePayload())
        let operation = Task {
            try await owner.withValue { _ in await borrowed.wait() }
        }
        await borrowed.waitUntilEntered()
        let calls = AccountingCounter()
        let first = Task { await owner.invalidate { _ in calls.increment(); await teardown.wait() } }
        for _ in 0..<100 { await Task.yield() }
        #expect(calls.value == 0)
        await borrowed.open()
        try await operation.value
        await teardown.waitUntilEntered()
        let completed = AccountingCounter()
        let second = Task {
            await owner.invalidate { _ in calls.increment() }
            completed.increment()
        }
        let cancellation = Task {
            await owner.withValueOrAwaitInvalidation { _ in calls.increment() }
            completed.increment()
        }
        for _ in 0..<100 { await Task.yield() }
        #expect(completed.value == 0)
        await teardown.open()
        await first.value
        await second.value
        await cancellation.value
        #expect(calls.value == 1)
        #expect(completed.value == 2)
    }
}

private final class ReferencePayload: Sendable {
    let onDeinit: (@Sendable () -> Void)?
    init(onDeinit: (@Sendable () -> Void)? = nil) { self.onDeinit = onDeinit }
    deinit { onDeinit?() }
}

private actor ReferenceOwnerLoader: EnginePoolModelLoader {
    private weak var payload: ReferencePayload?
    var payloadIsAlive: Bool { payload != nil }
    func loadModel(id: String, modelURL: URL) async throws -> AsyncResourceOwner<ReferencePayload> {
        let payload = ReferencePayload()
        self.payload = payload
        return AsyncResourceOwner(payload)
    }
    func unloadModel(_ engine: AsyncResourceOwner<ReferencePayload>, id: String) async {
        await engine.invalidate { _ in }
    }
}

private actor CapturedCachePayload {
    var slot: RetainedCachePayload<ReferencePayload>?
    func store(_ allocation: RetainedAllocation) throws {
        slot = try RetainedCachePayload(ReferencePayload(), allocation: allocation)
    }
    func discard() async {
        let allocation = slot?.allocation
        slot?.discard()
        slot = nil
        await allocation?.waitForReleaseIfUnused()
    }
}
