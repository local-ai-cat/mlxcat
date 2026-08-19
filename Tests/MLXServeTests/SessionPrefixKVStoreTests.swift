import MLX
import MLXLMCommon
@testable import MLXServe
import XCTest

final class SessionPrefixKVStoreTests: XCTestCase {
    func testMambaLayerRoundTripsWithoutCoercingMetadataIntoSimpleCache() throws {
        let original = MambaCache()
        original[0] = MLXArray.zeros([1, 2, 3])
        original[1] = MLXArray.zeros([1, 2, 3])
        let layer = SerializedKVLayer(
            state: original.state,
            metaState: original.metaState,
            className: "MambaCache"
        )

        let restored = try BlockAwarePrefixKVStore.cache(from: layer)

        XCTAssertTrue(restored is MambaCache)
        XCTAssertEqual(restored.metaState, original.metaState)
        XCTAssertEqual(restored.state.map(\.shape), original.state.map(\.shape))
    }

    func testMambaLayerRejectsUnsafeRewind() throws {
        let original = MambaCache()
        original[0] = MLXArray.zeros([1, 2, 3])
        original[1] = MLXArray.zeros([1, 2, 3])
        let store = SessionPrefixKVStore()
        try store.store(
            tokens: [1, 2, 3],
            sessionKey: "mamba",
            cache: [
                SerializedKVLayer(
                    state: original.state,
                    metaState: original.metaState,
                    className: "MambaCache"
                )
            ]
        )
        let hit = try XCTUnwrap(store.fetch(tokens: [1, 2, 9], sessionKey: "mamba"))

        XCTAssertThrowsError(try store.reconstructCache(from: hit)) { error in
            XCTAssertEqual(error as? PrefixKVStoreError, .unsupportedCacheTrim("MambaCache"))
        }
        store.release(hit)
    }

    func testSessionPlannerExtendsTrimsAndResetsBySession() throws {
        let store = SessionPrefixKVStore()
        try store.store(tokens: [1, 2, 3, 4], sessionKey: "s1", cache: Self.layers(tokenCount: 4))

        let extendHit = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 4, 5], sessionKey: "s1"))
        XCTAssertEqual(extendHit.matchedTokenCount, 4)
        store.release(extendHit)

        let trimHit = try XCTUnwrap(store.fetch(tokens: [1, 2, 9], sessionKey: "s1"))
        XCTAssertEqual(trimHit.matchedTokenCount, 2)
        let trimmed = try store.reconstructCache(from: trimHit)
        XCTAssertEqual(trimmed[0].state[0].dim(2), 2)
        store.release(trimHit)

        XCTAssertNil(store.fetch(tokens: [1, 2, 3, 4], sessionKey: "s2"))
        XCTAssertNil(store.fetch(tokens: [9, 9], sessionKey: "s1"))
    }

    func testAnonymousLaneUsesLongestPrefix() throws {
        let store = SessionPrefixKVStore()
        try store.store(tokens: [1, 2], sessionKey: nil, cache: Self.layers(tokenCount: 2))
        try store.store(tokens: [1, 2, 3, 4], sessionKey: nil, cache: Self.layers(tokenCount: 4))

        let hit = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 9], sessionKey: nil))
        XCTAssertEqual(hit.matchedTokenCount, 3)
        store.release(hit)
    }

    func testEvictsLeastRecentlyUsedUnleasedSlot() throws {
        let store = SessionPrefixKVStore(maxBytes: 40)
        try store.store(tokens: [1, 2], sessionKey: "old", cache: Self.layers(tokenCount: 2))
        try store.store(tokens: [3, 4], sessionKey: "new", cache: Self.layers(tokenCount: 2))

        XCTAssertNil(store.fetch(tokens: [1, 2, 5], sessionKey: "old"))
        let hit = try XCTUnwrap(store.fetch(tokens: [3, 4, 5], sessionKey: "new"))
        XCTAssertEqual(hit.matchedTokenCount, 2)
        store.release(hit)
        XCTAssertGreaterThanOrEqual(store.stats.evictionCount, 1)
    }

    func testLeasePreventsConcurrentTrimOrOverwriteUntilReleased() throws {
        let store = SessionPrefixKVStore()
        try store.store(tokens: [1, 2, 3], sessionKey: "s", cache: Self.layers(tokenCount: 3))

        let first = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 4], sessionKey: "s"))
        XCTAssertNil(store.fetch(tokens: [1, 2, 9], sessionKey: "s"))
        try store.store(tokens: [1, 2, 3, 4, 5], sessionKey: "s", cache: Self.layers(tokenCount: 5))

        store.release(first)
        let second = try XCTUnwrap(store.fetch(tokens: [1, 2, 9], sessionKey: "s"))
        XCTAssertEqual(second.matchedTokenCount, 2)
        store.release(second)

        let preserved = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 4, 5], sessionKey: "s"))
        XCTAssertEqual(preserved.matchedTokenCount, 3)
        store.release(preserved)
    }

    func testClearEntryRollsBackSlot() throws {
        let store = SessionPrefixKVStore()
        try store.store(tokens: [1, 2, 3], sessionKey: "s", cache: Self.layers(tokenCount: 3))

        let hit = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 4], sessionKey: "s"))
        store.clearEntry(hit)

        XCTAssertNil(store.fetch(tokens: [1, 2, 3, 4], sessionKey: "s"))
        XCTAssertEqual(store.stats.clearCount, 1)
    }

    private static func layers(tokenCount: Int) -> [SerializedKVLayer] {
        [
            SerializedKVLayer(
                state: [
                    MLXArray.zeros([1, 1, tokenCount, 2], dtype: .float32),
                    MLXArray.zeros([1, 1, tokenCount, 2], dtype: .float32),
                ],
                metaState: [],
                className: "KVCacheSimple"
            )
        ]
    }
}
