import MLX
import MLXLMCommon
@testable import MLXCat
import XCTest

/// Stack formation (pkg 115 lever A): a prefill stack forms over COLD rows, so
/// the batched layer has to be built from caches that have never been written.
final class BatchLayerCacheEmptyBatchedTests: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try MLXMetalRuntime.requireAvailable()
    }

    func testFreshRowCacheReportsNoState() {
        // The premise the whole stack-formation path rests on. If this ever
        // changes upstream, emptyBatched's guard silently stops matching and
        // stacks quietly stop forming — so it is asserted, not assumed.
        XCTAssertTrue(KVCacheSimple().state.isEmpty)
    }

    func testEmptyBatchedFormsASequenceLayerOfTheRequestedWidth() throws {
        let layer = try XCTUnwrap(BatchLayerCache.emptyBatched(rows: 4, like: KVCacheSimple()))
        let batched = try XCTUnwrap(layer.kvCache as? BatchKVCache)

        XCTAssertEqual(batched.batchSize, 4)
        XCTAssertEqual(batched.offset, 0)
        eval(batched.leftPadding)
        XCTAssertEqual(batched.leftPadding.asArray(Int32.self), [0, 0, 0, 0])
    }

    func testEmptyBatchedLayerAcceptsABatchedWriteAndSplitsBackToRows() throws {
        let rows = 3
        let width = 2
        let layer = try XCTUnwrap(BatchLayerCache.emptyBatched(rows: rows, like: KVCacheSimple()))

        var values: [Float] = []
        for row in 0 ..< rows {
            for position in 0 ..< width {
                values.append(Float((row + 1) * 10 + position))
            }
        }
        let block = MLXArray(values, [rows, 1, width, 1])
        _ = layer.kvCache.update(keys: block, values: block)

        XCTAssertEqual(layer.batchSize, rows)

        // Each row must come back out carrying its own tokens — this is the
        // split that hands a finished prefill to the decode batch.
        for row in 0 ..< rows {
            let extracted = layer.extract(row)
            eval(extracted.state[0])
            let rowValues = extracted.state[0].reshaped([width]).asArray(Float.self)
            XCTAssertEqual(rowValues, (0 ..< width).map { Float((row + 1) * 10 + $0) })
        }
    }

    func testRightPaddingPassesThroughToTheUnderlyingBatchCache() throws {
        let layer = try XCTUnwrap(BatchLayerCache.emptyBatched(rows: 2, like: KVCacheSimple()))
        try layer.prepare(rightPadding: [0, 1])

        let block = MLXArray([1, 2, 3, 4].map(Float.init), [2, 1, 2, 1])
        _ = layer.kvCache.update(keys: block, values: block)
        layer.finalize()

        let batched = try XCTUnwrap(layer.kvCache as? BatchKVCache)
        eval(batched.leftPadding, batched.batchOffset)
        XCTAssertEqual(batched.leftPadding.asArray(Int32.self), [0, 1])
        XCTAssertEqual(batched.batchOffset.asArray(Int32.self), [2, 1])
    }

    func testAlreadyWrittenCacheIsRefused() {
        let warm = KVCacheSimple()
        let block = MLXArray([1, 2].map(Float.init), [1, 1, 2, 1])
        _ = warm.update(keys: block, values: block)

        // A warm row belongs on the existing merge path, which knows how to
        // left-pad it against its peers; emptyBatched would silently drop its
        // contents.
        XCTAssertNil(BatchLayerCache.emptyBatched(rows: 2, like: warm))
    }

    func testNonSequenceLayerIsRefusedSoTheCallerCanFallBack() {
        let mamba = MambaCache()
        XCTAssertNil(BatchLayerCache.emptyBatched(rows: 2, like: mamba))
    }

    func testZeroRowsIsRefused() {
        XCTAssertNil(BatchLayerCache.emptyBatched(rows: 0, like: KVCacheSimple()))
    }
}
