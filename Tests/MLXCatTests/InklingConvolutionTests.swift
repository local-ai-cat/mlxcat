import MLX
import XCTest

@testable import MLXCatNative

final class InklingConvolutionTests: XCTestCase {
    func testShortConvolutionRetainsThePreviousTokenAcrossChunks() throws {
        try MLXMetalRuntime.requireAvailable()
        let input = MLXArray([Float(2), 4, 8]).reshaped(1, 3, 1)
        let weights = MLXArray([Float(1), 0]).reshaped(1, 2, 1)
        var full = InklingConvolution(taps: 2)
        var chunked = InklingConvolution(taps: 2)
        let expected = full.apply(input, weight: weights, outputType: .float32)
        let prefix = chunked.apply(input[0..., ..<2, 0...], weight: weights, outputType: .float32)
        let suffix = chunked.apply(input[0..., 2..., 0...], weight: weights, outputType: .float32)
        XCTAssertEqual(concatenated([prefix, suffix], axis: 1).asArray(Float.self),
                       expected.asArray(Float.self))
        XCTAssertEqual(suffix.item(Float.self), 12 / Float(3).squareRoot(), accuracy: 1e-6)
    }
}
