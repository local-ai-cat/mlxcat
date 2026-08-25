import MLX
import MLXLMCommon
@testable import MLXCat
import XCTest

/// Ragged multi-row prefill (pkg 115 lever A): rows write a common padded
/// width, then `finalize()` rolls each row's junk from the tail of the written
/// window to its head so the batch lands in the left-padded layout decode
/// expects.
final class BatchCacheRightPaddingTests: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try MLXMetalRuntime.requireAvailable()
    }

    /// Distinct per-row, per-position values so a misplaced roll is visible as
    /// a wrong number rather than a wrong shape.
    private func marker(row: Int, position: Int) -> Float {
        Float((row + 1) * 100 + position)
    }

    private func write(
        _ cache: BatchKVCache,
        rows: Int,
        width: Int,
        startPosition: Int
    ) {
        var values: [Float] = []
        for row in 0 ..< rows {
            for position in 0 ..< width {
                values.append(marker(row: row, position: startPosition + position))
            }
        }
        // [B, heads, width, headDim] with one head and one dim — the roll is
        // along the sequence axis and indifferent to the other extents.
        let block = MLXArray(values, [rows, 1, width, 1])
        _ = cache.update(keys: block, values: block)
    }

    func testFinalizeRollsEachRowByItsOwnPadding() throws {
        let cache = BatchKVCache(leftPadding: [0, 0, 0])
        // Real prompt lengths 4, 2, 3 written at a common width of 4.
        let lengths = [4, 2, 3]
        let width = 4
        try cache.prepare(rightPadding: lengths.map { width - $0 })
        write(cache, rows: 3, width: width, startPosition: 0)
        cache.finalize()

        eval(cache.state[0])
        let keys = cache.state[0]
        XCTAssertEqual(keys.shape, [3, 1, width, 1])

        for (row, length) in lengths.enumerated() {
            let padding = width - length
            let rowValues = keys[row].reshaped([width]).asArray(Float.self)
            for position in 0 ..< length {
                XCTAssertEqual(
                    rowValues[padding + position],
                    marker(row: row, position: position),
                    "row \(row) token \(position) landed on the wrong column"
                )
            }
        }
    }

    func testFinalizeMovesPaddingOutOfTheTipColumn() throws {
        let lengths = [4, 2, 3]
        let width = 4
        let cache = BatchKVCache(leftPadding: [0, 0, 0])
        try cache.prepare(rightPadding: lengths.map { width - $0 })
        write(cache, rows: 3, width: width, startPosition: 0)
        cache.finalize()

        eval(cache.state[0])
        let keys = cache.state[0]
        // Every row's LAST real token must sit on the final column — that is
        // the token the row decodes from, and the whole point of the roll.
        for (row, length) in lengths.enumerated() {
            let rowValues = keys[row].reshaped([width]).asArray(Float.self)
            XCTAssertEqual(
                rowValues[width - 1],
                marker(row: row, position: length - 1),
                "row \(row) does not have its tip on the last column"
            )
        }
    }

    func testFinalizeAdjustsPaddingAndPositionOffsets() throws {
        let lengths = [4, 2, 3]
        let width = 4
        let cache = BatchKVCache(leftPadding: [0, 0, 0])
        try cache.prepare(rightPadding: lengths.map { width - $0 })
        write(cache, rows: 3, width: width, startPosition: 0)
        cache.finalize()

        eval(cache.leftPadding, cache.batchOffset)
        XCTAssertEqual(cache.leftPadding.asArray(Int32.self), [0, 2, 1])
        // A row's next position is its real length, not the padded width.
        XCTAssertEqual(cache.batchOffset.asArray(Int32.self), lengths.map(Int32.init))
        // The written window keeps the padded width.
        XCTAssertEqual(cache.offset, width)
    }

    func testUnpaddedPrepareIsANoOp() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        try cache.prepare(rightPadding: [0, 0])
        XCTAssertFalse(cache.hasPendingRightPadding)

        write(cache, rows: 2, width: 3, startPosition: 0)
        cache.finalize()

        eval(cache.leftPadding, cache.batchOffset)
        XCTAssertEqual(cache.leftPadding.asArray(Int32.self), [0, 0])
        XCTAssertEqual(cache.batchOffset.asArray(Int32.self), [3, 3])
    }

    /// A row that finishes in an earlier chunk keeps its alignment while the
    /// other rows keep writing — the case that makes padding-per-tick safe.
    func testContinuingRowsAreUntouchedByAnotherRowsPadding() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        // Chunk 1: both rows write 2 real tokens, no padding.
        try cache.prepare(rightPadding: [0, 0])
        write(cache, rows: 2, width: 2, startPosition: 0)
        cache.finalize()

        // Chunk 2: row 0 writes 2 more, row 1 finishes with 1 real + 1 pad.
        try cache.prepare(rightPadding: [0, 1])
        write(cache, rows: 2, width: 2, startPosition: 2)
        cache.finalize()

        eval(cache.state[0], cache.leftPadding, cache.batchOffset)
        let keys = cache.state[0]
        let rowZero = keys[0].reshaped([4]).asArray(Float.self)
        XCTAssertEqual(rowZero, (0 ..< 4).map { marker(row: 0, position: $0) })

        let rowOne = keys[1].reshaped([4]).asArray(Float.self)
        // Row 1 holds 3 real tokens with one pad column rolled to the front.
        XCTAssertEqual(rowOne[1], marker(row: 1, position: 0))
        XCTAssertEqual(rowOne[2], marker(row: 1, position: 1))
        XCTAssertEqual(rowOne[3], marker(row: 1, position: 2))

        XCTAssertEqual(cache.leftPadding.asArray(Int32.self), [0, 1])
        XCTAssertEqual(cache.batchOffset.asArray(Int32.self), [4, 3])
    }

    func testPreparingTwiceWithoutFinalizeIsRejected() throws {
        let cache = BatchKVCache(leftPadding: [0, 0])
        try cache.prepare(rightPadding: [0, 1])
        XCTAssertThrowsError(try cache.prepare(rightPadding: [1, 0])) { error in
            XCTAssertEqual(error as? BatchKVCacheError, .rightPaddingAlreadyPrepared)
        }
    }

    func testPaddingMustNameEveryRow() throws {
        let cache = BatchKVCache(leftPadding: [0, 0, 0])
        XCTAssertThrowsError(try cache.prepare(rightPadding: [0, 1])) { error in
            XCTAssertEqual(error as? BatchKVCacheError, .rightPaddingWidthMismatch(expected: 3, actual: 2))
        }
    }
}
