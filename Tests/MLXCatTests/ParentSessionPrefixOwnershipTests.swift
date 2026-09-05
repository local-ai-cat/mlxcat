import Foundation
import MLXCat
import Testing

/// Actual store wiring with metadata-only Mamba state. No MLXArray is created or
/// evaluated; payload/debt tests separately cover retained reference objects and bytes.
@Suite("Parent session prefix ownership", .timeLimit(.minutes(1)))
struct ParentSessionPrefixOwnershipTests {
    @Test func sessionReplacementAndClearRetireActualSlotOwnership() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let store = SessionPrefixKVStore(maxAnonymousSlots: 1)
        let state = [SerializedKVLayer(state: [], metaState: [], className: "MambaCache")]
        let first = try await pool.reserveRequest(additionalBytes: 40)
        let firstOwner = try #require(await pool.retainedAllocation(for: first))
        try store.store(tokens: [1, 2], sessionKey: "session", cache: state, allocation: firstOwner)
        await pool.releaseRequest(first)
        #expect(await parent.used == 40)
        #expect(store.stats.slotCount == 1)
        let hit = try #require(store.fetch(tokens: [1, 2, 3], sessionKey: "session"))
        store.clearEntry(hit)
        await firstOwner.waitForReleaseIfUnused()
        #expect(await parent.used == 40)
        store.release(hit)
        await firstOwner.waitForReleaseIfUnused()
        #expect(await parent.used == 0)
        var previousOwner: RetainedAllocation?
        for _ in 0..<5 {
            let request = try await pool.reserveRequest(additionalBytes: 40)
            let owner = try #require(await pool.retainedAllocation(for: request))
            try store.store(tokens: [1, 2], sessionKey: "session", cache: state, allocation: owner)
            await previousOwner?.waitForReleaseIfUnused()
            await pool.releaseRequest(request)
            #expect(await parent.used == 40)
            previousOwner = owner
        }
        await store.clearAndWait()
        #expect(await parent.used == 0)
        #expect(store.stats.slotCount == 0)
    }

    @Test func anonymousEvictionReleasesOldGenerationWithoutAccumulation() async throws {
        let parent = ParentBudgetProbe(limit: 100)
        let pool = EnginePool(models: [:], loader: ParentAccountingLoader(), parentBudget: parent)
        let store = SessionPrefixKVStore(maxAnonymousSlots: 1)
        let state = [SerializedKVLayer(state: [], metaState: [], className: "MambaCache")]
        let first = try await pool.reserveRequest(additionalBytes: 40)
        let firstOwner = try #require(await pool.retainedAllocation(for: first))
        try store.store(tokens: [1, 2], sessionKey: nil, cache: state, allocation: firstOwner)
        await pool.releaseRequest(first)
        let second = try await pool.reserveRequest(additionalBytes: 40)
        let secondOwner = try #require(await pool.retainedAllocation(for: second))
        try store.store(tokens: [3, 4], sessionKey: nil, cache: state, allocation: secondOwner)
        await firstOwner.waitForReleaseIfUnused()
        await pool.releaseRequest(second)
        #expect(await parent.used == 40)
        #expect(store.stats.evictionCount == 1)
        await store.clearAndWait()
        #expect(await parent.used == 0)
    }
}
