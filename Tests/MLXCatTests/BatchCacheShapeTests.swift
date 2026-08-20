import MLX
import MLXLMCommon
@testable import MLXCat
import XCTest

final class BatchCacheShapeTests: XCTestCase {
    func testSingletonMambaExtractionPreservesConcreteCacheType() throws {
        let cache = MambaCache()
        cache[0] = MLXArray.zeros([1, 2, 3])
        cache[1] = MLXArray.zeros([1, 2, 3])
        let layer = try BatchLayerCache.adoptSingle(cache)

        let extracted = layer.extract(0)

        XCTAssertTrue(extracted is MambaCache)
        XCTAssertEqual(extracted.metaState, cache.metaState)
        XCTAssertEqual(extracted.state.map(\.shape), cache.state.map(\.shape))
    }

    override func setUpWithError() throws {
        try super.setUpWithError()
        try MLXMetalRuntime.requireAvailable()
    }

    func testSequenceMergeUsesInferredAxisForRankThreeKV() throws {
        let short = KVCacheSimple()
        short.state = [
            MLXArray.zeros([1, 1, 8], dtype: .float32),
            MLXArray.zeros([1, 1, 8], dtype: .float32),
        ]

        let long = KVCacheSimple()
        long.state = [
            MLXArray.zeros([1, 3, 8], dtype: .float32),
            MLXArray.zeros([1, 3, 8], dtype: .float32),
        ]

        let batch = try BatchKVCache.merge([short, long])

        XCTAssertEqual(batch.state[0].shape, [2, 3, 8])
        XCTAssertEqual(batch.state[1].shape, [2, 3, 8])
        XCTAssertEqual(batch.extract(0).state[0].shape, [1, 1, 8])
        XCTAssertEqual(batch.extract(1).state[0].shape, [1, 3, 8])

        switch (batch as any KVCache).ropeOffset {
        case .batch(let offset):
            XCTAssertEqual(offset.shape, [2])
        default:
            XCTFail("BatchKVCache must dispatch per-row RoPE offsets")
        }
    }

    func testFixedStateMergePreservesConcreteCacheType() throws {
        let first = FixedStateCache(
            state: [
                MLXArray.zeros([1, 3, 16], dtype: .bfloat16),
                MLXArray.zeros([1, 4, 8, 8], dtype: .float32),
            ]
        )
        let second = FixedStateCache(
            state: [
                MLXArray.zeros([1, 3, 16], dtype: .bfloat16),
                MLXArray.zeros([1, 4, 8, 8], dtype: .float32),
            ]
        )

        let batch = try BatchLayerCache.merge([first, second])

        XCTAssertTrue(batch.kvCache is FixedStateCache)
        XCTAssertEqual(batch.kvCache.state[0].shape, [2, 3, 16])
        XCTAssertEqual(batch.kvCache.state[1].shape, [2, 4, 8, 8])

        batch.filter(keeping: [1])

        XCTAssertTrue(batch.kvCache is FixedStateCache)
        XCTAssertEqual(batch.kvCache.state[0].shape, [1, 3, 16])
        XCTAssertEqual(batch.kvCache.state[1].shape, [1, 4, 8, 8])

        let extracted = batch.extract(0)
        XCTAssertEqual(extracted.state[0].shape, [1, 3, 16])
        XCTAssertEqual(extracted.state[1].shape, [1, 4, 8, 8])
    }

    func testMergeAcceptsKnownRotatingSequenceCacheTypes() throws {
        let short = RotatingTestCache()
        short.offset = 6
        short.state = [
            MLXArray.zeros([1, 1, 2, 8], dtype: .float32),
            MLXArray.zeros([1, 1, 2, 8], dtype: .float32),
        ]

        let long = RotatingTestCache()
        long.state = [
            MLXArray.zeros([1, 1, 4, 8], dtype: .float32),
            MLXArray.zeros([1, 1, 4, 8], dtype: .float32),
        ]

        let batch = try BatchKVCache.merge([short, long])

        XCTAssertEqual(batch.state[0].shape, [2, 1, 4, 8])
        XCTAssertEqual(batch.state[1].shape, [2, 1, 4, 8])
        XCTAssertEqual(batch.extract(0).state[0].shape, [1, 1, 2, 8])
        XCTAssertEqual(batch.extract(1).state[0].shape, [1, 1, 4, 8])

        switch (batch as any KVCache).ropeOffset {
        case .batch(let offset):
            XCTAssertEqual(offset.asArray(Int.self), [6, 4])
        default:
            XCTFail("BatchKVCache must preserve per-row absolute RoPE offsets")
        }
    }

    func testMergeRestoresTemporalOrderForWrappedRotatingCacheState() throws {
        let rotating = RotatingTestCache()
        rotating.offset = 6
        rotating.metaState = ["0", "4", "256", "6", "2"]
        rotating.state = [
            MLXArray(Int32(0) ..< Int32(4)).reshaped([1, 1, 4, 1]),
            MLXArray(Int32(10) ..< Int32(14)).reshaped([1, 1, 4, 1]),
        ]

        let batch = try BatchKVCache.merge([rotating])

        XCTAssertEqual(batch.state[0].asArray(Int.self), [2, 3, 0, 1])
        XCTAssertEqual(batch.state[1].asArray(Int.self), [12, 13, 10, 11])
        switch (batch as any KVCache).ropeOffset {
        case .batch(let offset):
            XCTAssertEqual(offset.asArray(Int.self), [6])
        default:
            XCTFail("BatchKVCache must preserve absolute RoPE offsets after temporal restore")
        }
    }

    func testSlidingWindowMaskIgnoresLeftPaddingForMixedLengthRows() throws {
        let mask = try XCTUnwrap(CausalMask.create(
            n: 1,
            offset: 5,
            leftPadding: MLXArray([Int32(4), Int32(0)]),
            windowSize: 3
        ))

        XCTAssertEqual(mask.shape, [2, 1, 1, 6])
        XCTAssertEqual(mask.asArray(Bool.self), [
            false, false, false, false, true, true,
            false, false, false, true, true, true,
        ])
    }

    func testWindowedSingleTokenMaskDoesNotUseFullAttentionFastPath() throws {
        XCTAssertNil(CausalMask.create(n: 1, offset: 5))

        let mask = try XCTUnwrap(CausalMask.create(
            n: 1,
            offset: 5,
            windowSize: 3
        ))

        XCTAssertEqual(mask.shape, [1, 6])
        XCTAssertEqual(mask.asArray(Bool.self), [
            false, false, false, true, true, true,
        ])
    }

    func testTrimKeepsMixedLengthSequenceRowsConsistent() throws {
        let short = KVCacheSimple()
        short.state = [
            MLXArray.zeros([1, 1, 2, 8], dtype: .float32),
            MLXArray.zeros([1, 1, 2, 8], dtype: .float32),
        ]

        let long = KVCacheSimple()
        long.state = [
            MLXArray.zeros([1, 1, 5, 8], dtype: .float32),
            MLXArray.zeros([1, 1, 5, 8], dtype: .float32),
        ]

        let batch = try BatchKVCache.merge([short, long])

        XCTAssertEqual(batch.trim(4), 4)
        XCTAssertEqual(batch.extract(0).state[0].shape, [1, 1, 0, 8])
        XCTAssertEqual(batch.extract(1).state[0].shape, [1, 1, 1, 8])

        switch (batch as any KVCache).ropeOffset {
        case .batch(let offset):
            XCTAssertEqual(offset.asArray(Int.self), [0, 1])
        default:
            XCTFail("BatchKVCache must preserve per-row RoPE offsets after trim")
        }
    }
}

private final class FixedStateCache: KVCache {
    var state: [MLXArray]
    var metaState: [String]
    var offset: Int { 0 }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    init(state: [MLXArray], metaState: [String] = ["2", "0,1"]) {
        self.state = state
        self.metaState = metaState
    }

    func innerState() -> [MLXArray] {
        state
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        state = [newKeys, newValues]
        return (newKeys, newValues)
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        0
    }

    func copy() -> any KVCache {
        FixedStateCache(state: state.map { $0[.ellipsis] }, metaState: metaState)
    }
}

private final class RotatingTestCache: KVCache {
    var state: [MLXArray] = []
    var metaState: [String] {
        get {
            [String(keep), String(maxCacheSize), String(step), String(offset), String(rotatingIndex)]
        }
        set {
            guard newValue.count == 5 else { return }
            keep = Int(newValue[0]) ?? keep
            maxCacheSize = Int(newValue[1]) ?? maxCacheSize
            step = Int(newValue[2]) ?? step
            offset = Int(newValue[3]) ?? offset
            idx = Int(newValue[4]) ?? idx
        }
    }
    var offset = 0
    var maxSize: Int? { maxCacheSize }
    var isTrimmable: Bool { false }

    private var keep = 0
    private var maxCacheSize = 4
    private var step = 256
    private var idx = 0
    private var rotatingIndex: Int {
        idx == 0 ? state.first?.dim(2) ?? 0 : idx
    }

    func innerState() -> [MLXArray] {
        state
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        state = [newKeys, newValues]
        return (newKeys, newValues)
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        0
    }

    func copy() -> any KVCache {
        let copy = RotatingTestCache()
        copy.state = state.map { $0[.ellipsis] }
        copy.metaState = metaState
        return copy
    }
}
