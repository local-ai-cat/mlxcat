import MLX
import MLXNN
import XCTest

/// MLX's `rope_single` fast path drops the batch axis from its dispatch grid.
///
/// `RoPE::eval_gpu` takes a "single" branch when the input is row-contiguous,
/// the sequence length is 1, and the offset is a SCALAR
/// (`mlx/backend/metal/rope.cpp:95-96`). That branch dispatches
/// `MTL::Size(dims/2, N, 1)` where `N` multiplies only the head axes — the batch
/// size `B` appears nowhere (`rope.cpp:134-139`), while the general branch puts
/// it in the grid at `:149`. The kernel therefore covers exactly batch row 0,
/// and because the row-contiguous input is donated, rows 1..B-1 come out of the
/// shared buffer **unrotated**.
///
/// This is why batched gemma diverges and nothing else does: gemma's VLM
/// attention is the only code in our stack that reaches the RoPE primitive with
/// a scalar offset on a B>1 decode tensor (`Gemma4.swift:872,880,903`). Every
/// other family rotates through `applyRotaryPosition(rope, to:, offset:
/// cache?.ropeOffset)`, which for a `BatchKVCache` is a per-row ARRAY — and an
/// array offset forces `single = false`.
///
/// It also explains every observation without needing a second cause: width 1 is
/// exact because B=1 covers the whole tensor; identical rows disagree with each
/// other because row 0 is right and the rest are untouched; padding is
/// irrelevant because rows >= 1 never have their offset read at all; and the
/// pipelining toggle changes nothing because the defect lives inside one
/// primitive.
///
/// No model, no weights — this runs anywhere Metal does.
final class RoPEBatchGridProbeTests: XCTestCase {

    /// FIXED 2026-08-24, and this is now the regression guard.
    ///
    /// It was briefly pinned to the BROKEN behaviour, because asserting
    /// correctness while the bug was live would have left a permanently red test
    /// that everyone learns to ignore. That inversion did its job: it went red
    /// exactly once, the moment the vendored MLX gained the batch axis, and its
    /// message said what to do next.
    ///
    /// The fix is three lines backported from ml-explore/mlx `76a977ca` (#3498)
    /// onto `ce45c525`, carried on `atlas-open-sources/mlx` and pinned through
    /// `atlas-open-sources/mlx-swift`. If this test fails again, the pin has
    /// slipped back to a stock mlx-swift.
    func testScalarOffsetRoPERotatesEveryRow() throws {
        try MLXMetalRuntime.requireAvailable()

        let batch = 4, heads = 8, dims = 256, offset = 96
        MLXRandom.seed(0)
        // The decode shape gemma hands the primitive: [B, H, 1, D].
        let x = MLXRandom.normal([batch, heads, 1, dims]).asType(.bfloat16)
        let rope = RoPE(dimensions: dims, traditional: false, base: 10000)

        let batched = rope(x, offset: offset)
        eval(batched)

        // Row 0 is the control: whatever the kernel does, it does it correctly
        // for the first row, so a mismatch here would mean the probe is wrong
        // rather than the kernel.
        let solo = rope(x[0 ..< 1], offset: offset)
        eval(solo)
        let rowZeroError = (batched[0 ..< 1] - solo).abs().max().item(Float.self)
        XCTAssertEqual(rowZeroError, 0, accuracy: 1e-6,
            "row 0 does not match a width-1 rotation — this probe is measuring the wrong thing")

        // The signature: rows >= 1 identical to the INPUT means they were never
        // rotated at all.
        var unrotated: [Int] = []
        var maxRotation: [Float] = []
        for row in 1 ..< batch {
            let delta = (batched[row] - x[row]).abs().max().item(Float.self)
            maxRotation.append(delta)
            if delta == 0 { unrotated.append(row) }
        }
        print("ROPEGRID scalar-offset rows-unrotated=\(unrotated) "
            + "max|out-in| per row=\(maxRotation.map { String(format: "%.4f", $0) })")

        XCTAssertTrue(
            unrotated.isEmpty,
            """
            rows \(unrotated) came out of RoPE identical to the input. The \
            scalar-offset fast path is rotating row 0 and passing the rest \
            through, which means the mlx-swift pin has slipped back to a build \
            without the ml-explore/mlx 76a977ca backport.
            """)
    }

    func testArrayOffsetRoPERotatesEveryRow() throws {
        try MLXMetalRuntime.requireAvailable()

        // The path every other family takes, and the fix direction for gemma:
        // a per-row offset array forces the general branch, which has B in its
        // dispatch grid.
        let batch = 4, heads = 8, dims = 256, offset = 96
        MLXRandom.seed(0)
        let x = MLXRandom.normal([batch, heads, 1, dims]).asType(.bfloat16)
        let rope = RoPE(dimensions: dims, traditional: false, base: 10000)

        let perRow = rope(x, offset: MLXArray(Array(repeating: Int32(offset), count: batch)))
        eval(perRow)

        var unrotated: [Int] = []
        for row in 0 ..< batch where (perRow[row] - x[row]).abs().max().item(Float.self) == 0 {
            unrotated.append(row)
        }
        print("ROPEGRID array-offset rows-unrotated=\(unrotated)")
        XCTAssertTrue(unrotated.isEmpty, "rows \(unrotated) were not rotated on the array path")

        // Every row got the same offset, so every row must match row 0.
        for row in 1 ..< batch {
            let delta = (perRow[row] - rope(x[row ..< (row + 1)], offset: offset)).abs()
                .max().item(Float.self)
            XCTAssertEqual(delta, 0, accuracy: 1e-6, "row \(row) disagrees with its solo rotation")
        }
    }
}
