import MLX
import XCTest

@testable import MLXCatNative

final class InklingAttentionTests: XCTestCase {
    func testRelativeMaskUsesBackwardDistanceAndAppendsTheSink() throws {
        try MLXMetalRuntime.requireAvailable()
        let relative = MLXArray.ones([1, 2, 1, 1])
        let projection = MLXArray([Float(3), 7]).reshaped(1, 2)
        let sink = MLXArray([Float(11), 13]).reshaped(1, 2, 1)
        let global = InklingAttention.mask(
            relative: relative, projection: projection, sink: sink, keyCount: 3, window: nil)
        XCTAssertEqual(global.asArray(Float.self), [7, 3, -1e30, 11, 0, 7, 3, 13])
        let local = InklingAttention.mask(
            relative: relative, projection: projection, sink: sink, keyCount: 3, window: 2)
        XCTAssertEqual(local.asArray(Float.self), [7, 3, -1e30, 11, -1e30, 7, 3, 13])
    }
}
