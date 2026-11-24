import MLX
import MLXLMCommon
@testable import MLXCat
import XCTest

final class SessionPrefixKVStoreTests: XCTestCase {
    func testNamedSessionPrefixRestoresFromDiskWithExactArrays() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data([0]).write(to: model.appendingPathComponent("model.safetensors"))
        let diskRoot = root.appendingPathComponent("prefix")
        let disk = try XCTUnwrap(SessionPrefixDiskStore(root: diskRoot, modelDirectory: model))
        let writer = SessionPrefixKVStore(maxBytes: 0, windowedLayerDecoder: nil, disk: disk)
        let tokens = Array(0..<256)
        let data = arange(256).asType(.bfloat16).reshaped(1, 1, 256, 1)
        let layer = SerializedKVLayer(state: [data, data], metaState: [], className: "KVCacheSimple")
        try writer.store(tokens: tokens, sessionKey: "session", cache: [layer])
        disk.flushWrites()
        let reopened = try XCTUnwrap(SessionPrefixDiskStore(root: diskRoot, modelDirectory: model))
        let reader = SessionPrefixKVStore(maxBytes: 0, windowedLayerDecoder: nil, disk: reopened)
        XCTAssertNil(reader.fetch(tokens: tokens + [256], sessionKey: "another-session"))
        let hit = try XCTUnwrap(reader.fetch(tokens: tokens + [256], sessionKey: "session"))
        XCTAssertEqual(hit.matchedTokenCount, 256)
        let restored = try reader.reconstructCache(from: hit)
        XCTAssertEqual(restored[0].state[0].view(dtype: .uint16).asArray(UInt16.self),
                       data.view(dtype: .uint16).asArray(UInt16.self))
        try Data([0, 1]).write(to: model.appendingPathComponent("model.safetensors"))
        let changedModel = try XCTUnwrap(SessionPrefixDiskStore(root: diskRoot, modelDirectory: model))
        XCTAssertNil(changedModel.read(session: "session"))
        reader.clearEntry(hit)
        reopened.flushWrites()
        XCTAssertNil(reopened.read(session: "session"))
    }

    func testRotatingPrefixRestoresChronologicalStateAfterWrap() throws {
        let cache = RotatingKVCache(maxSize: 4, keep: 1, step: 2)
        func append(_ value: Int, to cache: RotatingKVCache) -> [Float] {
            let row = MLXArray(Float(value)).reshaped(1, 1, 1, 1)
            return cache.update(keys: row, values: row).0.asArray(Float.self)
        }
        for token in 0..<7 { _ = append(token, to: cache) }
        let layer = SerializedKVLayer(state: cache.state, metaState: cache.metaState, className: "RotatingKVCache")
        let restored = try XCTUnwrap(BlockAwarePrefixKVStore.cache(from: layer) as? RotatingKVCache)
        XCTAssertEqual(restored.offset, cache.offset)
        XCTAssertEqual(append(7, to: restored), append(7, to: cache))
        XCTAssertEqual(append(8, to: restored), append(8, to: cache))
    }

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
        // A recurrent slot cannot rewind, so a shorter match is a miss at fetch
        // (it used to be handed out and fail at reconstruction, shadowing any
        // slot that could have served — HybridPrefixCheckpointTests).
        XCTAssertNil(store.fetch(tokens: [1, 2, 9], sessionKey: "mamba"))
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
