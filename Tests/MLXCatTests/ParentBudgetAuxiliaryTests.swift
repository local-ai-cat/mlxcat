import Foundation
import MLXCat
import Testing

@Suite("Parent budget auxiliary lifetimes", .timeLimit(.minutes(1)))
struct ParentBudgetAuxiliaryTests {
    private func resource(_ state: AccountingResourceState, bytes: Int64? = 40) -> EnginePoolResource {
        EnginePoolResource(id: "speech:fixture", modelID: "fixture", modelType: "speech",
                           footprint: .init(bytes: bytes, source: .configuredEstimate), usesMLXAllocator: false,
                           load: { try await state.load() }, unload: { await state.unload() })
    }

    @Test func deniedAuxiliaryNeverInvokesFactoryAndUnknownIsNotZero() async throws {
        let parent = ParentBudgetProbe(limit: 30)
        for bytes: Int64? in [40, nil] {
            let state = AccountingResourceState()
            let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
            await pool.registerResource(resource(state, bytes: bytes))
            await #expect(throws: ParentBudgetFailure.self) { try await pool.acquireResource("speech:fixture") }
            #expect(await state.loads == 0)
            #expect(await parent.used == 0)
            #expect(await pool.status().currentModelMemory == 0)
        }
    }

    @Test func auxiliarySharedWaiterCancellationDoesNotReleaseResidentToken() async throws {
        let loading = AccountingBarrier()
        let state = AccountingResourceState(loadGate: loading)
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        await pool.registerResource(resource(state))
        let first = Task { try await pool.acquireResource("speech:fixture") }
        await loading.waitUntilEntered()
        let second = Task { try await pool.acquireResource("speech:fixture") }
        first.cancel()
        #expect(await parent.used == 40)
        #expect(await parent.allocations.first?.usesMLXAllocator == false)
        await loading.open()
        await #expect(throws: CancellationError.self) { try await first.value }
        let lease = try await second.value
        #expect(await parent.allocations.count == 1)
        #expect(await state.loaderWasCancelled == false)
        #expect(await parent.used == 40)
        await pool.releaseResource(lease)
        try await pool.unloadResource("speech:fixture")
        #expect(await parent.releaseCalls == 1)
    }

    @Test func auxiliaryFailureAndUnloadRetainChargesThroughCleanup() async throws {
        let cleanup = AccountingBarrier()
        let state = AccountingResourceState(unloadGate: cleanup, failFirstLoad: true)
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        await pool.registerResource(resource(state))
        let failed = Task { try await pool.acquireResource("speech:fixture") }
        await cleanup.waitUntilEntered()
        #expect(await parent.used == 40)
        #expect(await pool.status().currentModelMemory == 40)
        await cleanup.open()
        await #expect(throws: AccountingFailure.self) { try await failed.value }
        #expect(await parent.used == 0)
        #expect(await parent.releaseCalls == 1)
        let lease = try await pool.acquireResource("speech:fixture")
        #expect(await parent.allocations.count == 2)
        #expect(await parent.used == 40)
        await pool.releaseResource(lease)
        try await pool.unloadResource("speech:fixture")
        #expect(await parent.used == 0)
        #expect(await parent.releaseCalls == 2)
    }

    @Test func explicitAuxiliaryUnloadAcknowledgesBeforeParentRelease() async throws {
        let unloading = AccountingBarrier()
        let state = AccountingResourceState(unloadGate: unloading)
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        await pool.registerResource(resource(state))
        let lease = try await pool.acquireResource("speech:fixture")
        await pool.releaseResource(lease)
        let stopped = Task { try await pool.unloadResource("speech:fixture") }
        await unloading.waitUntilEntered()
        #expect(await parent.used == 40)
        #expect(await pool.resourceStatuses().first?.isUnloading == true)
        await unloading.open()
        try await stopped.value
        #expect(await parent.used == 0)
        #expect(await parent.releaseCalls == 1)
    }
}
