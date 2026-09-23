import MLX
import MLXLMCommon
@testable import MLXCat
import XCTest

/// Hybrid (attention + recurrent) prefix reuse: a follow-up turn diverges from the
/// stored prompt at its final token, so the recurrent layers — which cannot trim —
/// must resume from a checkpoint while attention KV trims to the same position.
final class HybridPrefixCheckpointTests: XCTestCase {
    private static func attention(tokens: Int) -> SerializedKVLayer {
        SerializedKVLayer(
            state: [
                MLXArray.zeros([1, 1, tokens, 2], dtype: .float32),
                MLXArray.zeros([1, 1, tokens, 2], dtype: .float32),
            ],
            metaState: [],
            className: "KVCacheSimple"
        )
    }

    /// A recurrent layer whose state carries `marker` so the test can tell which
    /// position's state came back.
    private static func recurrent(marker: Float) -> SerializedKVLayer {
        let cache = MambaCache()
        cache[0] = MLXArray.full([1, 2, 3], values: MLXArray(marker))
        cache[1] = MLXArray.full([1, 2, 3], values: MLXArray(marker))
        return SerializedKVLayer(state: cache.state, metaState: cache.metaState, className: "MambaCache")
    }

    private static func marker(_ layer: SerializedKVLayer) -> Float {
        layer.state[0].reshaped([-1])[0].item(Float.self)
    }

    func testFollowUpResumesAtTheCheckpointBelowTheDivergence() throws {
        let store = SessionPrefixKVStore()
        try store.store(
            tokens: [1, 2, 3, 4, 5],
            sessionKey: nil,
            cache: [Self.attention(tokens: 5), Self.recurrent(marker: 5)],
            checkpoints: [PrefixRecurrentCheckpoint(position: 4, layers: [nil, Self.recurrent(marker: 4)])]
        )

        // The follow-up re-renders the last prompt token differently (Qwen's
        // "<think>\n" vs "<think>\n\n</think>"), so the match stops at 4.
        let hit = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 4, 9, 10], sessionKey: nil))
        XCTAssertEqual(hit.matchedTokenCount, 4)
        let layers = try store.reconstructCache(from: hit)
        XCTAssertEqual(layers[0].state[0].dim(2), 4, "attention trims to the checkpoint")
        XCTAssertEqual(Self.marker(layers[1]), 4, "recurrent state comes from the checkpoint, not the end")
        store.release(hit)
    }

    func testExactExtensionUsesTheEndStateWithoutACheckpoint() throws {
        let store = SessionPrefixKVStore()
        try store.store(
            tokens: [1, 2, 3, 4, 5],
            sessionKey: nil,
            cache: [Self.attention(tokens: 5), Self.recurrent(marker: 5)],
            checkpoints: [PrefixRecurrentCheckpoint(position: 4, layers: [nil, Self.recurrent(marker: 4)])]
        )
        let hit = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 4, 5, 6], sessionKey: nil))
        XCTAssertEqual(hit.matchedTokenCount, 5)
        let layers = try store.reconstructCache(from: hit)
        XCTAssertEqual(Self.marker(layers[1]), 5)
        store.release(hit)
    }

    func testDivergenceBelowEveryCheckpointIsAMiss() throws {
        let store = SessionPrefixKVStore()
        try store.store(
            tokens: [1, 2, 3, 4, 5],
            sessionKey: nil,
            cache: [Self.attention(tokens: 5), Self.recurrent(marker: 5)],
            checkpoints: [PrefixRecurrentCheckpoint(position: 4, layers: [nil, Self.recurrent(marker: 4)])]
        )
        XCTAssertNil(store.fetch(tokens: [1, 2, 9, 9, 9], sessionKey: nil))
    }

    func testSessionContinuationKeepsThePromptCheckpointAndEndState() throws {
        let store = SessionPrefixKVStore()
        // Turn 1 admission: prompt [1..5], checkpoint at 4.
        try store.store(
            tokens: [1, 2, 3, 4, 5],
            sessionKey: "s",
            cache: [Self.attention(tokens: 5), Self.recurrent(marker: 5)],
            checkpoints: [PrefixRecurrentCheckpoint(position: 4, layers: [nil, Self.recurrent(marker: 4)])]
        )
        // Turn 1 finish: prompt + generated reply replaces the session slot.
        try store.store(
            tokens: [1, 2, 3, 4, 5, 6, 7],
            sessionKey: "s",
            cache: [Self.attention(tokens: 7), Self.recurrent(marker: 7)]
        )
        // Turn 2 diverges at the re-rendered reply.
        let hit = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 4, 8, 8, 8], sessionKey: "s"))
        XCTAssertEqual(hit.matchedTokenCount, 4)
        XCTAssertEqual(Self.marker(try store.reconstructCache(from: hit)[1]), 4)
        store.release(hit)
        let exact = try XCTUnwrap(store.fetch(tokens: [1, 2, 3, 4, 5, 9], sessionKey: "s"))
        XCTAssertEqual(exact.matchedTokenCount, 5, "the replaced slot's end state is a checkpoint too")
        XCTAssertEqual(Self.marker(try store.reconstructCache(from: exact)[1]), 5)
        store.release(exact)
    }

    func testFlagIsOffByDefault() {
        XCTAssertTrue(Scheduler.hybridPrefixReuseEnabled(environment: [:]))
        XCTAssertTrue(Scheduler.hybridPrefixReuseEnabled(environment: ["MLXCAT_HYBRID_PREFIX_REUSE": "1"]))
        XCTAssertFalse(Scheduler.hybridPrefixReuseEnabled(environment: ["MLXCAT_HYBRID_PREFIX_REUSE": "0"]))
        XCTAssertFalse(Scheduler.hybridPrefixReuseEnabled(environment: ["MLXCAT_HYBRID_PREFIX_REUSE": "off"]))
    }
}
