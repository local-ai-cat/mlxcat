import MLX
import XCTest

@testable import MLXCatNative

final class InklingExpertsTests: XCTestCase {
    func testPairedNVFP4PreservesTopEightOutputAndPrefill() throws {
        try MLXMetalRuntime.requireAvailable()
        let count = 16
        let width = 32
        let weight = sin(arange(count * width * width).asType(.float32) * 0.03)
            .reshaped(count, width, width)
        func projection(_ offset: Float) -> InklingExperts.Projection {
            let packed = quantized(weight + offset, groupSize: 16, bits: 4, mode: .nvfp4)
            return .init(weight: packed.wq, scales: packed.scales,
                         format: .nvfp4(scale: (arange(count).asType(.float32) + 1) * 0.02))
        }
        let up = projection(0.2)
        let gate = projection(-0.2)
        let pair = try XCTUnwrap(InklingExperts.PairedNVFP4(up: up, gate: gate))
        let router = weight[0 ..< count, 0, 0...]
        let bias = MLXArray.zeros([count])
        let down = projection(0.1)
        let reference = InklingExperts(
            router: router, routingBias: bias, up: up, gate: gate, down: down,
            topK: 8, routeScale: 8, upCap: 25, gateCap: 4)
        let fused = InklingExperts(
            router: router, routingBias: bias,
            up: pair.projection(rows: 0..<width, format: up.format),
            gate: pair.projection(rows: width..<(width * 2), format: gate.format),
            down: down, topK: 8, routeScale: 8, upCap: 25, gateCap: 4, paired: pair)
        for length in [1, 32] {
            let input = cos(arange(length * width).asType(.float32) * 0.17)
                .reshaped(1, length, width).asType(.bfloat16)
            XCTAssertEqual(reference(input).asArray(Float.self), fused(input).asArray(Float.self))
        }
    }

    func testMXFP8VectorGatherMatchesTheAlignedMLXKernel() throws {
        try MLXMetalRuntime.requireAvailable()
        let width = 256
        let channels = 16
        let input = sin(arange(3 * width).asType(.float32))
            .reshaped(1, 3, 1, 1, width).asType(.bfloat16)
        let weights = cos(arange(2 * channels * width).asType(.float32))
            .reshaped(2, channels, width).asType(.bfloat16)
        let packed = quantized(weights, groupSize: 32, bits: 8, mode: .mxfp8)
        let projection = InklingExperts.Projection(
            weight: packed.wq, scales: packed.scales, format: .mxfp8)
        let indices = MLXArray([Int32(0), 1, 1, 0, 1, 1]).reshaped(1, 3, 2)

        // Explicitly prepare the zero columns expected by the pinned gather.
        let paddedInput = concatenated([input, MLXArray.zeros(like: input)], axis: -1)
        let paddedWeights = concatenated([packed.wq, MLXArray.zeros(like: packed.wq)], axis: -1)
        let identityScales = MLXArray.ones(like: packed.scales) * UInt8(127)
        let paddedScales = concatenated([packed.scales, identityScales], axis: -1)
        let expected = gatherQuantizedMM(
            paddedInput, paddedWeights, scales: paddedScales, biases: nil, rhsIndices: indices,
            groupSize: 32, bits: 8, mode: .mxfp8)

        let actual = projection(input, indices: indices)
        XCTAssertEqual(actual.shape, [1, 3, 2, 1, channels])
        XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                       expected.asType(.float32).asArray(Float.self))
    }

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

    func testAffineExpertGatherMatchesSelectedAffineMatrix() throws {
        try MLXMetalRuntime.requireAvailable()
        let rows = 32
        let width = 64
        let weight = sin(arange(2 * rows * width).asType(.float32))
            .reshaped(2, rows, width).asType(.bfloat16)
        let input = cos(arange(2 * width).asType(.float32))
            .reshaped(2, 1, width).asType(.bfloat16)
        let packed = quantized(weight, groupSize: 64, bits: 4, mode: .affine)
        let biases = try XCTUnwrap(packed.biases)
        let projection = InklingExperts.Projection(
            weight: packed.wq, scales: packed.scales,
            format: .affine(biases: biases, groupSize: 64, bits: 4))
        let actual = projection(input, indices: MLXArray([Int32(1), 0]))
        for (row, expert) in [1, 0].enumerated() {
            let expected = quantizedMM(
                input[row], packed.wq[expert], scales: packed.scales[expert],
                biases: biases[expert], groupSize: 64, bits: 4, mode: .affine)
            XCTAssertEqual(actual[row].asType(.float32).asArray(Float.self),
                           expected.asType(.float32).asArray(Float.self))
        }
    }

    func testPackedGroupedTileMatchesStagedGroupedMatmul() throws {
        try MLXMetalRuntime.requireAvailable()
        let width = 64
        let channels = 64
        let experts = 3
        let input = sin(arange(37 * width).asType(.float32))
            .reshaped(37, 1, width).asType(.bfloat16)
        let weight = cos(arange(experts * channels * width).asType(.float32))
            .reshaped(experts, channels, width)
        let packed = quantized(weight, groupSize: 16, bits: 4, mode: .nvfp4)
        let factors = MLXArray([Float(0.75), 1, 1.5])
        let routes = MLXArray(Array(repeating: Int32(0), count: 18)
            + Array(repeating: Int32(2), count: 19))
        let staged = InklingNVFP4Unpack.run(
            weights: packed.wq, scales: packed.scales, factor: factors, dtype: input.dtype)
        let expected = gatherMM(
            input, staged.transposed(0, 2, 1), rhsIndices: routes, sortedIndices: true)
        let actual = InklingNVFP4Grouped.run(
            input, weights: packed.wq, scales: packed.scales, factors: factors, indices: routes)
        XCTAssertEqual(actual.shape, expected.shape)
        XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                       expected.asType(.float32).asArray(Float.self))
    }

    func testGroupedNVFP4UsesTheSelectedExpertsTensorScale() throws {
        try MLXMetalRuntime.requireAvailable()
        let one = MLXArray.ones([1, 8, 64], dtype: .bfloat16)
        let packed = quantized(concatenated([one, one * 2], axis: 0),
                               groupSize: 16, bits: 4, mode: .nvfp4)
        let projection = InklingExperts.Projection(
            weight: packed.wq, scales: packed.scales,
            format: .nvfp4(scale: MLXArray([Float(0.5), 2])))
        let input = MLXArray.ones([16, 1, 64], dtype: .bfloat16)
        let selected = MLXArray(Array(repeating: Int32(0), count: 8)
                                + Array(repeating: Int32(1), count: 8))
        let result = projection(input, indices: selected, sorted: true)
        XCTAssertEqual(result.shape, [16, 1, 8])

        // Compare against separate projections of the two stable expert rows.
        let dense = dequantized(packed.wq, scales: packed.scales, biases: nil,
                                groupSize: 16, bits: 4, mode: .nvfp4, dtype: .float32)
        for (expert, scale) in [Float(0.5), 2].enumerated() {
            let expertWeight = (dense[expert] * scale).asType(.bfloat16)
            let expected = matmul(input[(expert * 8)..<((expert + 1) * 8)], expertWeight.transposed())
            let actual = result[(expert * 8)..<((expert + 1) * 8)]
            XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                           expected.asType(.float32).asArray(Float.self))
        }
    }
}
