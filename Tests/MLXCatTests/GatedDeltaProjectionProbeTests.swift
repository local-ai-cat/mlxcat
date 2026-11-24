import Foundation
import MLX
import MLXCatNative
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class GatedDeltaProjectionProbeTests: XCTestCase {
    func testReferenceArithmeticOnCapturedLayerZero() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let capture = environment["MLXCAT_GDN_INPUT"],
              let checkpoint = environment["MLXCAT_GDN_MODEL"] else {
            throw XCTSkip("Set the layer-zero capture and verified model paths.")
        }
        try MLXMetalRuntime.requireAvailable()
        let model = try await VLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: checkpoint), using: NativeTokenizerLoader())
        try await model.perform { context in
            let arrays = try loadArrays(url: URL(fileURLWithPath: capture))
            func read(_ name: String) throws -> MLXArray {
                try XCTUnwrap(arrays[name], name)
            }
            let (output, state) = try gatedDeltaUpdate(
                q: read("q"), k: read("k"), v: read("v"),
                a: read("a"), b: read("b"), aLog: read("aLog"), dtBias: read("dtBias"),
                referenceRounding: true)
            try Self.assertExact(output, read("y"), label: "recurrence output")
            try Self.assertExact(state, read("state"), label: "recurrent state")

            let qwen = try XCTUnwrap(context.model as? Qwen35)
            let layer = try XCTUnwrap(qwen.loraLayers.first as? Qwen35Language.DecoderLayer)
            let attention = try XCTUnwrap(layer.linearAttn)
            try Self.assertExact(attention(read("x")), read("attention"), label: "layer-zero attention output")
        }
    }

    private static func assertExact(_ actual: MLXArray, _ expected: MLXArray, label: String) {
        XCTAssertEqual(actual.shape, expected.shape, label)
        XCTAssertEqual(actual.dtype, expected.dtype, label)
        guard actual.shape == expected.shape else { return }
        let differences = (actual .!= expected).sum().item(Int.self)
        XCTAssertEqual(differences, 0, label)
        print("GDN \(label) \(actual.size - differences)/\(actual.size) exact")
    }
}
