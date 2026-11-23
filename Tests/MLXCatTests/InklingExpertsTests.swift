import MLX
import XCTest

@testable import MLXCatNative

final class InklingExpertsTests: XCTestCase {
    func testPackedGatherMatchesIndividualPackedProjections() throws {
        try MLXMetalRuntime.requireAvailable()
        let ones = MLXArray.ones([1, 32, 64], dtype: .float32)
        let weights = concatenated([ones, ones * 2], axis: 0)
        let input = MLXArray.ones([1, 1, 1, 64], dtype: .bfloat16)
        let indices = MLXArray([Int32(1), 0]).reshaped(1, 2)
        let formats: [(QuantizationMode, Int, Int, InklingExperts.Projection.Format, [Float])] = [
            (.nvfp4, 16, 4, .nvfp4(scale: MLXArray([Float(0.5), 2])), [2, 0.5]),
            (.mxfp8, 32, 8, .mxfp8, [1, 1]),
        ]
        for (mode, group, bits, format, factors) in formats {
            let packed = quantized(weights, groupSize: group, bits: bits, mode: mode)
            let projection = InklingExperts.Projection(
                weight: packed.wq, scales: packed.scales, format: format)
            let result = projection(input, indices: indices)
            XCTAssertEqual(result.shape, [1, 2, 1, 32])
            for (route, factor) in factors.enumerated() {
                let expert = 1 - route
                let direct = quantizedMM(
                    input[0, 0, 0..., 0...].asType(mode == .nvfp4 ? .float32 : input.dtype),
                    packed.wq[expert, 0..., 0...], scales: packed.scales[expert, 0..., 0...],
                    biases: nil, groupSize: group, bits: bits, mode: mode)
                let expected = (direct * factor).asType(input.dtype)
                let actual = result[0, route, 0, 0...]
                let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
                XCTAssertLessThanOrEqual(error, 0.02, "\(mode) expert \(expert)")
            }
        }
    }
}
