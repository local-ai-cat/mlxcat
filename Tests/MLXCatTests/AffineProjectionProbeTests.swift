import Foundation
import MLX
import XCTest

final class AffineProjectionProbeTests: XCTestCase {
    func testCaptureAffineProjection() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let input = environment["MLXCAT_AFFINE_INPUT"],
              let output = environment["MLXCAT_AFFINE_OUTPUT"] else {
            throw XCTSkip("Set MLXCAT_AFFINE_INPUT and MLXCAT_AFFINE_OUTPUT for the standalone projection probe.")
        }
        try MLXMetalRuntime.requireAvailable()
        let tensors = try loadArrays(url: URL(fileURLWithPath: input))
        let x = try XCTUnwrap(tensors["x"])
        var outputs: [String: MLXArray] = [:]

        for name in ["in_proj_qkv", "in_proj_z"] {
            let weights = try XCTUnwrap(tensors["\(name)_weight"])
            let scales = try XCTUnwrap(tensors["\(name)_scales"])
            let biases = try XCTUnwrap(tensors["\(name)_biases"])
            for rows in [1, 4, 5, 8, 16, x.dim(1)] {
                for (dtype, suffix) in [(DType.bfloat16, "bf16"), (.float32, "f32")] {
                    outputs["\(name)_\(rows)_\(suffix)"] = quantizedMM(
                        x[0..., ..<rows].asType(dtype), weights,
                        scales: scales.asType(dtype), biases: biases.asType(dtype),
                        transpose: true, groupSize: 64, bits: 4)
                }
            }
        }
        try save(arrays: outputs, url: URL(fileURLWithPath: output))
    }
}
