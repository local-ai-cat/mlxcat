import Foundation
import MLX
import MLXCatNative
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class GatedDeltaProjectionProbeTests: XCTestCase {
    func testReferenceArithmeticOnCapturedLayer() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let capture = environment["MLXCAT_GDN_INPUT"],
              let checkpoint = environment["MLXCAT_GDN_MODEL"] else {
            throw XCTSkip("Set the gated-delta capture and verified model paths.")
        }
        let index = environment["MLXCAT_GDN_LAYER"].flatMap(Int.init) ?? 0
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
            let layers = qwen.loraLayers
            let layer = try XCTUnwrap(
                layers.indices.contains(index) ? layers[index] as? Qwen35Language.DecoderLayer : nil)
            let attention = try XCTUnwrap(layer.linearAttn)
            let x = try read("x")
            for (name, value) in [
                ("in_proj_qkv", attention.inProjQKV(x)),
                ("in_proj_z", attention.inProjZ(x)),
                ("in_proj_b", attention.inProjB(x)),
                ("in_proj_a", attention.inProjA(x)),
            ] {
                if let expected = arrays[name] {
                    Self.assertExact(value, expected, label: name)
                }
            }
            try Self.assertExact(attention(x), read("attention"), label: "layer \(index) attention output")
            if let postNorm = arrays["post_norm"], let mlp = arrays["mlp"] {
                let block = try XCTUnwrap(layer.mlp as? Qwen35Language.MLP)
                Self.assertExact(block(postNorm), mlp, label: "layer \(index) MLP")
            }
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
