import MLX
import XCTest

struct Tolerance {
    let absolute: Float
    let relative: Float

    static let float32 = Tolerance(absolute: 1e-5, relative: 1e-5)
    static let float16 = Tolerance(absolute: 1e-3, relative: 1e-3)
    static let bfloat16 = Tolerance(absolute: 2e-2, relative: 2e-2)
}

func XCTAssertAllClose(
    _ lhs: MLXArray,
    _ rhs: MLXArray,
    tolerance: Tolerance,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertEqual(lhs.shape, rhs.shape, "shape mismatch", file: file, line: line)

    let lhs32 = lhs.asType(.float32)
    let rhs32 = rhs.asType(.float32)
    let difference = abs(lhs32 - rhs32)
    let threshold = MLXArray(tolerance.absolute) + MLXArray(tolerance.relative) * abs(rhs32)
    let close = all(difference .<= threshold).item(Bool.self)

    XCTAssertTrue(close, "arrays differ beyond tolerance", file: file, line: line)
}
