import MLX
import MLXLMCommon
@testable import MLXCat
import XCTest

final class PrefillStackTests: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        try MLXMetalRuntime.requireAvailable()
    }

    private func coldRows(_ rows: Int, layers: Int = 2) -> [[any KVCache]] {
        (0 ..< rows).map { _ in (0 ..< layers).map { _ in KVCacheSimple() } }
    }

    // MARK: Formation

    func testFormsOverColdPlainCaches() throws {
        let stack = try XCTUnwrap(PrefillStack.form(rowCaches: coldRows(4)))
        XCTAssertEqual(stack.rowCount, 4)
        XCTAssertEqual(stack.caches.count, 2)
    }

    func testRefusesASingleRow() {
        // One row is not a stack; the caller keeps the existing solo path,
        // which is cheaper than a batched forward of width 1.
        XCTAssertNil(PrefillStack.form(rowCaches: coldRows(1)))
    }

    func testRefusesWhenAnyRowIsWarm() {
        var rows = coldRows(3)
        let warm = KVCacheSimple()
        let block = MLXArray([1, 2].map(Float.init), [1, 1, 2, 1])
        _ = warm.update(keys: block, values: block)
        rows[2][1] = warm

        // Row 2 already holds tokens. Forming here would drop them silently,
        // so the model falls back to solo prefill instead.
        XCTAssertNil(PrefillStack.form(rowCaches: rows))
    }

    func testRefusesAModelWithARecurrentLayer() {
        var rows = coldRows(2)
        rows[0][1] = MambaCache()
        rows[1][1] = MambaCache()
        XCTAssertNil(PrefillStack.form(rowCaches: rows))
    }

    func testRefusesRaggedLayerCounts() {
        var rows = coldRows(2)
        rows[1] = [KVCacheSimple()]
        XCTAssertNil(PrefillStack.form(rowCaches: rows))
    }

    // MARK: Token block

    func testTokenBlockRightPadsToTheWidestChunk() throws {
        let (block, padding) = try XCTUnwrap(
            PrefillStack.tokenBlock(chunks: [[1, 2, 3], [4], [5, 6]])
        )
        XCTAssertEqual(block.shape, [3, 3])
        XCTAssertEqual(padding, [0, 2, 1])

        eval(block)
        XCTAssertEqual(block.asArray(Int32.self), [1, 2, 3, 4, 0, 0, 5, 6, 0])
    }

    func testTokenBlockOfUniformChunksNeedsNoPadding() throws {
        let (block, padding) = try XCTUnwrap(PrefillStack.tokenBlock(chunks: [[1, 2], [3, 4]]))
        XCTAssertEqual(block.shape, [2, 2])
        XCTAssertEqual(padding, [0, 0])
    }

    func testTokenBlockRejectsAnAllEmptyTick() {
        // Nothing to write means the caller should have split every row out
        // before forwarding; a [k, 0] block would be a silent no-op forward.
        XCTAssertNil(PrefillStack.tokenBlock(chunks: [[], []]))
        XCTAssertNil(PrefillStack.tokenBlock(chunks: []))
    }

    // MARK: Row lifecycle

    private func write(_ stack: PrefillStack, chunks: [[Int]]) throws {
        let (block, padding) = try XCTUnwrap(PrefillStack.tokenBlock(chunks: chunks))
        try stack.prepare(rightPadding: padding)
        // Stand in for the model's per-layer KV write: one value per token so a
        // row's contents are identifiable after extraction.
        let width = block.dim(1)
        var values: [Float] = []
        for chunk in chunks {
            values.append(contentsOf: chunk.map(Float.init))
            values.append(contentsOf: Array(repeating: Float(0), count: width - chunk.count))
        }
        let kv = MLXArray(values, [chunks.count, 1, width, 1])
        for cache in stack.caches {
            _ = cache.update(keys: kv, values: kv)
        }
        stack.finalize()
    }

    func testExtractedRowCarriesOnlyItsOwnTokens() throws {
        let stack = try XCTUnwrap(PrefillStack.form(rowCaches: coldRows(3, layers: 1)))
        try write(stack, chunks: [[11, 12, 13], [21], [31, 32]])

        let row = stack.extractRow(0)
        XCTAssertEqual(row.count, 1)
        eval(row[0].state[0])
        XCTAssertEqual(row[0].state[0].reshaped([3]).asArray(Float.self), [11, 12, 13])
    }

    func testShorterRowsComeOutAtTheirRealLengthWithPaddingStripped() throws {
        let stack = try XCTUnwrap(PrefillStack.form(rowCaches: coldRows(3, layers: 1)))
        let chunks = [[11, 12, 13], [21], [31, 32]]
        try write(stack, chunks: chunks)

        // The rows shared a padded width of 3, but a row leaves the stack at
        // its OWN length: extract slices `leftPadding ..< idx`, so the padding
        // the roll moved to the head is exactly what gets cut here. A row that
        // came out padded would decode from a zero token.
        for (row, chunk) in chunks.enumerated() {
            let extracted = stack.extractRow(row)
            eval(extracted[0].state[0])
            let tokens = extracted[0].state[0]
            XCTAssertEqual(tokens.dim(2), chunk.count, "row \(row) came out at the wrong length")
            XCTAssertEqual(
                tokens.reshaped([chunk.count]).asArray(Float.self),
                chunk.map(Float.init),
                "row \(row) carries the wrong tokens"
            )
        }
    }

    func testRemovingRowsRenumbersTheRemainder() throws {
        let stack = try XCTUnwrap(PrefillStack.form(rowCaches: coldRows(3, layers: 1)))
        try write(stack, chunks: [[11, 12], [21, 22], [31, 32]])

        stack.removeRows([1])
        XCTAssertEqual(stack.rowCount, 2)

        // What was row 2 is now row 1 — the caller re-indexes to match.
        let promoted = stack.extractRow(1)
        eval(promoted[0].state[0])
        XCTAssertEqual(promoted[0].state[0].reshaped([2]).asArray(Float.self), [31, 32])
    }

    func testRemovingNoRowsIsANoOp() throws {
        let stack = try XCTUnwrap(PrefillStack.form(rowCaches: coldRows(2, layers: 1)))
        try write(stack, chunks: [[11], [21]])
        stack.removeRows([])
        XCTAssertEqual(stack.rowCount, 2)
    }
}
