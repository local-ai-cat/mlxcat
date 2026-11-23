import MLX
import XCTest

@testable import MLXCatNative

final class InklingExpertsTests: XCTestCase {
    func testPackedGatherAppliesTheSelectedExpertScale() throws {
        try MLXMetalRuntime.requireAvailable()
        let weights = MLXArray.ones([2, 32, 64], dtype: .float32)
        let input = MLXArray.ones([1, 1, 1, 64], dtype: .float32)
        let indices = MLXArray([Int32(1), 0]).reshaped(1, 2)
        let formats: [(QuantizationMode, Int, Int, InklingExperts.Projection.Format, [Float])] = [
            (.nvfp4, 16, 4, .nvfp4(scale: MLXArray([Float(0.5), 2])), [128, 32]),
            (.mxfp8, 32, 8, .mxfp8, [64, 64]),
        ]
        for (mode, group, bits, format, expected) in formats {
            let packed = quantized(weights, groupSize: group, bits: bits, mode: mode)
            let projection = InklingExperts.Projection(
                weight: packed.wq, scales: packed.scales, format: format)
            let result = projection(input, indices: indices)
            XCTAssertEqual(result.shape, [1, 2, 1, 32])
            for (route, expectedValue) in expected.enumerated() {
                let outputs = result[0, route, 0, 0...].asArray(Float.self)
                XCTAssertEqual(outputs.count, 32)
                XCTAssertTrue(outputs.allSatisfy { abs($0 - expectedValue) <= 2 })
            }
        }
    }
}
