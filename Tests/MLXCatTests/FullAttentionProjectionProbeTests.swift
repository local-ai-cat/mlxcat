import Foundation
import MLX
import MLXCatNative
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class FullAttentionProjectionProbeTests: XCTestCase {
    func testCapturedAttention() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelName = env["MLXCAT_ATTN_MODEL"], let reference = env["MLXCAT_ATTN_CAPTURE"] else {
            throw XCTSkip("Set the checkpoint and full-attention capture paths.")
        }
        let layerIndex = env["MLXCAT_ATTN_LAYER"].flatMap(Int.init) ?? 3
        if let metallib = env["MLXCAT_ATTN_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: metallib)
        }
        try MLXMetalRuntime.requireAvailable()
        let container = try await VLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelName), using: NativeTokenizerLoader())
        try await container.perform { context in
            let model = try XCTUnwrap(context.model as? Qwen35)
            let layers = model.loraLayers
            let layer = try XCTUnwrap(
                layers.indices.contains(layerIndex) ? layers[layerIndex] as? Qwen35Language.DecoderLayer : nil,
                "Expected a full-attention layer at index \(layerIndex)")
            let attention = try XCTUnwrap(layer.selfAttn)
            for file in try FileManager.default.contentsOfDirectory(atPath: reference).sorted()
                where file.hasSuffix(".safetensors") {
                print("CAPTURE \(file)")
                let path = URL(fileURLWithPath: reference).appending(path: file)
                let arrays = try loadArrays(url: path)
                try Self.replay(
                    attention, arrays: arrays, base: model.config.textConfiguration.ropeTheta,
                    checkShortRotation: layerIndex == 3)
                if let tokens = arrays["tokens"] {
                    try Self.traceBoundaries(model, tokens: tokens, expected: arrays, through: layerIndex)
                }
            }
        }
    }

    private static func traceBoundaries(
        _ model: Qwen35, tokens: MLXArray, expected: [String: MLXArray], through layerIndex: Int
    ) throws {
        let text = model.languageModel.model
        let layers = model.loraLayers.compactMap { $0 as? Qwen35Language.DecoderLayer }
        let sequence = [tokens] + (expected["prefix"]?.asArray(Int32.self) ?? []).map {
            MLXArray([$0]).reshaped(1, 1)
        }
        for explicitMultimodalPositions in [true, false] {
            let caches = try model.newCache(parameters: nil)
            var offset = 0
            for (step, input) in sequence.enumerated() {
                var hidden = text.embedTokens(input)
                let mask: MLXArray?
                if case .array(let causal) = createAttentionMask(h: hidden, cache: caches[text.faIdx], returnArray: true) {
                    mask = causal
                } else {
                    mask = nil
                }
                let length = input.dim(-1)
                let positions = broadcast(
                    arange(offset, offset + length).reshaped(1, 1, length), to: [3, 1, length])
                for (index, layer) in layers.prefix(layerIndex + 1).enumerated() {
                    hidden = layer(
                        hidden, attentionMask: mask, ssmMask: nil, cache: caches[index],
                        positionIds: explicitMultimodalPositions ? positions : nil)
                    if step + 1 == sequence.count {
                        let boundary = try XCTUnwrap(expected["layer_\(index)"])
                        let difference = abs(hidden.asType(.float32) - boundary.asType(.float32)).max().item(Float.self)
                        print("BOUNDARY \(explicitMultimodalPositions ? "mrope" : "flat") \(index) max \(difference)")
                    }
                }
                offset += length
            }
        }
    }

    private static func replay(
        _ attention: Qwen35Language.Attention, arrays: [String: MLXArray], base: Float,
        checkShortRotation: Bool
    ) throws {
        func read(_ name: String) throws -> MLXArray { try XCTUnwrap(arrays[name], name) }
        func check(
            _ name: String, _ actual: MLXArray, against expected: MLXArray, exact: Bool = false
        ) {
            XCTAssertEqual(actual.shape, expected.shape, name)
            XCTAssertEqual(actual.dtype, expected.dtype, name)
            guard actual.shape == expected.shape else { return }
            let diff = abs(actual.asType(.float32) - expected.asType(.float32))
            let changed = (diff .!= 0).sum().item(Int.self)
            if exact {
                XCTAssertEqual(changed, 0, name)
            }
            print("ATTN \(name) \(actual.dtype)/\(expected.dtype) max \(diff.max().item(Float.self)) "
                + "changed \(changed)/\(actual.size)")
        }
        func compare(_ name: String, _ x: MLXArray, exact: Bool = true) throws {
            try check(name, x, against: read(name), exact: exact)
        }

        let x = try read("x"), offset = try read("offset").item(Int.self)
        let b = x.dim(0), t = x.dim(1)
        let projected = attention.qProj(x), kp = attention.kProj(x), vp = attention.vProj(x)
        try compare("projected", projected)
        try compare("kp", kp)
        try compare("vp", vp)
        let split = try read("projected").reshaped(b, t, attention.numAttentionHeads, -1).split(parts: 2, axis: -1)
        // Preserve the sequence/head transpose. Safetensors materializes
        // the arrays and would select a different Metal RoPE specialization.
        let q = attention.qNorm(split[0]).transposed(0, 2, 1, 3)
        let k = attention.kNorm(kp.reshaped(b, t, attention.numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
        let v = vp.reshaped(b, t, attention.numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        try compare("q", q)
        try compare("k", k)
        try compare("v", v)
        let positions = broadcast(arange(offset, offset + t).reshaped(1, 1, t), to: [3, b, t])
        let (co, si) = attention.rotaryEmbedding(x: v, positionIds: positions)
        let (qr, kr) = Qwen35Language.applyMultimodalRotaryPosEmb(q: q, k: k, cos: co, sin: si)
        try compare("qr", qr, exact: false)
        try compare("kr", kr, exact: false)
        let rotate = { (x: MLXArray) in
            MLXFast.RoPE(x, dimensions: co.dim(-1), traditional: false, base: base, scale: 1, offset: offset)
        }
        try check("fast-qr", rotate(q), against: read("qr"))
        try check("fast-kr", rotate(k), against: read("kr"))
        let starts = MLXArray([Int32(offset)])
        for (name, input) in [("qr", q), ("kr", k)] {
            let expected = try read(name)
            let actual = MLXFast.RoPE(
                input, dimensions: co.dim(-1), traditional: false, base: base, scale: 1, offset: starts)
            let exact = checkShortRotation && t < 1024
            check("array-offset-\(name)", actual, against: expected, exact: exact)
            if exact {
                XCTAssertTrue(arrayEqual(rotate(input), expected).item(Bool.self), "Scalar RoPE \(name)")
            }
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode =
            arrays["mask"].map { .array($0) } ?? (arrays["causal"] == nil ? .none : .causal)
        let keys = try read("ck"), values = try read("cv")
        let cache = KVCacheSimple()
        if offset > 0 {
            _ = cache.update(keys: keys[0..., 0..., ..<offset, 0...], values: values[0..., 0..., ..<offset, 0...])
        }
        let (newK, newV) = try cache.update(keys: read("kr"), values: v)
        try compare("ck", newK)
        try compare("cv", newV)
        let attended = try MLXFast.scaledDotProductAttention(
            queries: read("qr"), keys: keys, values: values, scale: attention.scale, mask: mask)
        try compare("attended", attended, exact: false)
        if let contiguous = arrays["attended_contiguous"] {
            check("attended-contiguous", attended, against: contiguous)
        }
        if arrays["causal"] != nil {
            let explicit = createCausalMask(n: t, offset: offset)
            let materializedMask = try MLXFast.scaledDotProductAttention(
                queries: read("qr"), keys: keys, values: values, scale: attention.scale, mask: .array(explicit))
            try check("attended-explicit-mask", materializedMask, against: read("attended"))
        }
        let gated = try read("attended").transposed(0, 2, 1, 3).reshaped(b, t, -1) * sigmoid(read("gate"))
        try compare("gated", gated)
        try compare("y", attention.oProj(read("gated")))
    }
}
