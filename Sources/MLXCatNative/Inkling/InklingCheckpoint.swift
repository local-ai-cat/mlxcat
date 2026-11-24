import Foundation
import MLX

struct InklingCheckpoint {
    struct Index: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    enum LoadError: Error {
        case missingTensor(String)
        case missingShard(String)
        case wrongShard(String)
        case invalidQuantization(String)
    }

    let tensors: [String: MLXArray]
    private let quantization: [String: Any]

    init(directory: URL) throws {
        let indexURL = directory.appending(component: "model.safetensors.index.json")
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: indexURL))
        let shards = Dictionary(grouping: index.weightMap, by: \.value)
        var tensors: [String: MLXArray] = [:]
        for (file, entries) in shards {
            guard file == URL(fileURLWithPath: file).lastPathComponent,
                file.hasSuffix(".safetensors")
            else { throw LoadError.wrongShard(file) }
            let path = directory.appending(component: file)
            guard FileManager.default.fileExists(atPath: path.path) else {
                throw LoadError.missingShard(file)
            }
            let loaded = try loadArrays(url: path)
            for (key, _) in entries {
                guard let weight = loaded[key] else { throw LoadError.missingTensor(key) }
                tensors[key] = weight
            }
        }
        self.tensors = tensors
        let config = directory.appending(component: "config.json")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: config)) as? [String: Any]
        self.quantization = object?["quantization"] as? [String: Any] ?? [:]
    }

    var usesPackedVocabulary: Bool {
        let path = "language_model.lm_head"
        return quantizationMode(for: path) == "affine" || tensors["\(path).scales"] != nil
    }

    func required(_ name: String) throws -> MLXArray {
        guard let tensor = tensors[name] else { throw LoadError.missingTensor(name) }
        return tensor
    }

    private func quantizationMode(for path: String) -> String? {
        (quantization[path] as? [String: Any])?["mode"] as? String
    }

    private func affine(_ path: String) throws -> InklingProjection.Affine {
        let weight = try required("\(path).weight")
        let scales = try required("\(path).scales")
        let biases = try required("\(path).biases")
        guard let metadata = quantization[path] as? [String: Any],
            metadata["mode"] as? String == "affine",
            let groupSize = metadata["group_size"] as? Int, [32, 64, 128].contains(groupSize),
            let bits = metadata["bits"] as? Int, [2, 3, 4, 5, 6, 8].contains(bits),
            weight.dtype == .uint32, weight.ndim >= 2, scales.ndim == weight.ndim,
            scales.shape == biases.shape, scales.dtype == biases.dtype,
            weight.shape.dropLast() == scales.shape.dropLast(),
            weight.dim(-1) * 32 == scales.dim(-1) * groupSize * bits
        else { throw LoadError.invalidQuantization(path) }
        return .init(packed: weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }

    private func projection(_ path: String) throws -> InklingProjection {
        if quantizationMode(for: path) == "affine" || tensors["\(path).scales"] != nil {
            return .affine(try affine(path))
        }
        return .dense(try required("\(path).weight"))
    }

}

extension InklingCheckpoint {
    func textModel(config: InklingConfiguration.Text) throws -> InklingTextModel {
        let model = "language_model.model"
        let loop = try config.makeLoop()
        let layers = try (0..<config.layerCount).map { index in
            try layer(config: config, index: index, prefix: "\(model).layers.\(index)")
        }
        return try InklingTextModel(
            config: config, loop: loop, layers: layers,
            embeddings: projection("\(model).embed_tokens"),
            embeddingNorm: required("\(model).embed_norm.weight"),
            messageQueries: tensors["\(model).message_queries"]
                ?? required("\(model).message_input.query"),
            finalQuery: required("\(model).final_input.query"),
            finalNorm: required("\(model).norm.weight"),
            vocabulary: vocabulary())
    }

    private func vocabulary() throws -> InklingProjection {
        let path = "language_model.lm_head"
        if usesPackedVocabulary { return try .affine(affine(path)) }
        return try .vocabulary(required("\(path).weight"))
    }

    private func layer(
        config: InklingConfiguration.Text, index: Int, prefix: String
    ) throws -> InklingTextModel.Layer {
        let attention = "\(prefix).self_attn"
        let mlp = "\(prefix).mlp"
        let isLocal = config.localLayerIDs.contains(index)
        func convolution(_ name: String) throws -> InklingDecoderAttention.ShortWeights {
            let path = "\(attention).\(name)_sconv"
            if isLocal { return .fixed(try required("\(path).conv.weight")) }
            return try .dynamic(
                staticWeight: required("\(path).static_weight"),
                projection: required("\(path).weight_proj.weight"))
        }
        func expert(_ name: String) throws -> InklingExperts.Projection {
            let path = "\(mlp).switch_mlp.\(name)_proj"
            let format: InklingExperts.Projection.Format
            if let factor = tensors["\(path).weight_scale_2"] {
                format = .nvfp4(scale: factor)
            } else if quantizationMode(for: path) == "affine" || tensors["\(path).biases"] != nil {
                let weight = try affine(path)
                format = .affine(biases: weight.biases, groupSize: weight.groupSize, bits: weight.bits)
            } else {
                format = .mxfp8
            }
            return try InklingExperts.Projection(
                weight: required("\(path).weight"), scales: required("\(path).scales"), format: format)
        }
        let attentionModule = try InklingDecoderAttention(
            config: config, window: isLocal ? config.windowSize : nil,
            queryWeight: projection("\(attention).q_proj"),
            keyWeight: projection("\(attention).k_proj"),
            valueWeight: projection("\(attention).v_proj"),
            relativeWeight: projection("\(attention).r_proj"),
            outputWeight: projection("\(attention).o_proj"),
            gateWeight: projection("\(attention).gate_proj"),
            sinkWeight: projection("\(attention).sink_proj"),
            relativeProjection: required("\(attention).rel_proj"),
            queryScale: required("\(attention).q_scale"),
            relativeScale: required("\(attention).rel_scale"),
            keyConvolution: convolution("k"), valueConvolution: convolution("v"))
        let originalUp = try expert("up")
        let originalGate = try expert("gate")
        let pairExperts = ProcessInfo.processInfo.environment["MLXCAT_INKLING_PAIRED_EXPERTS"] == "1"
        let paired = pairExperts ? InklingExperts.PairedNVFP4(up: originalUp, gate: originalGate) : nil
        let middle = originalUp.weight.dim(1)
        if let paired {
            eval(paired.weight, paired.scales)
        }
        let experts = try InklingExperts(
            router: required("\(mlp).gate_weight"), routingBias: required("\(mlp).e_score_correction_bias"),
            up: paired?.projection(rows: 0..<middle, format: originalUp.format) ?? originalUp,
            gate: paired?.projection(rows: middle..<(2 * middle), format: originalGate.format) ?? originalGate,
            down: expert("down"),
            topK: config.routeCount, routeScale: config.routeScale,
            upCap: config.upCap, gateCap: config.gateCap, paired: paired)
        let normNames = isLocal ? ["input"] : ["q_input", "k_input", "v_input"]
        let norms = try normNames.map { try required("\(prefix).\($0)_layernorm.weight") }
        let controllers: [InklingTextModel.Controller]
        if (config.loop.start..<config.loop.end).contains(index) {
            controllers = try ["attn", "k", "v", "mlp"].map { name in
                let path = "\(prefix).controllers.\(name)"
                let legacy = "\(prefix).\(name == "attn" ? "q" : name)_message_gate"
                return try InklingTextModel.Controller(
                    down: tensors["\(path).down_weight"] ?? required("\(legacy).down_proj.weight"),
                    forget: tensors["\(path).forget_weight"] ?? required("\(legacy).forget_gate_proj.weight"),
                    gain: tensors["\(path).gain_weight"] ?? required("\(legacy).message_gate_proj.weight"))
            }
        } else {
            controllers = []
        }
        var layer = try InklingTextModel.Layer(
            attention: attentionModule, experts: experts, inputNorms: norms,
            postAttentionNorm: required("\(prefix).post_attention_layernorm.weight"),
            attentionQuery: tensors["\(prefix).attention_input.query"],
            mlpQuery: required("\(prefix).mlp_input.query"),
            attentionConvolution: required("\(prefix).attn_sconv.conv.weight"),
            mlpConvolution: required("\(prefix).mlp_sconv.conv.weight"),
            controllers: controllers)
        if ProcessInfo.processInfo.environment["MLXCAT_INKLING_COMPILED_EXPERTS"] != "0" {
            layer.compiledExperts = compile { (input: MLXArray) in experts(input) }
        }
        return layer
    }
}

extension InklingCheckpoint {
    func visionModel(config: InklingConfiguration.Vision) throws -> InklingVision {
        let prefix = "vision_tower"
        func linear(_ path: String) throws -> InklingVision.Linear {
            try .init(weight: projection(path), bias: tensors["\(path).bias"])
        }
        func norm(_ path: String) throws -> InklingVision.Norm {
            try .init(weight: required("\(path).weight"), bias: required("\(path).bias"),
                      epsilon: config.epsilon)
        }
        let layers = try (0..<config.layers).map { index in
            let layer = "\(prefix).layers.\(index)"
            let attention = "\(layer).self_attn"
            return try InklingVision.Layer(
                before: norm("\(layer).layernorm_before"), after: norm("\(layer).layernorm_after"),
                query: linear("\(attention).q_proj"), key: linear("\(attention).k_proj"),
                value: linear("\(attention).v_proj"), output: linear("\(attention).out_proj"),
                expand: linear("\(layer).mlp.fc1"), project: linear("\(layer).mlp.fc2"),
                attentionScale: required("\(layer).lambda_1"), mlpScale: required("\(layer).lambda_2"))
        }
        let embedding = "\(prefix).embeddings"
        return try InklingVision(
            config: config, patches: linear("\(embedding).patch_embedding"),
            position: required("\(embedding).position_embedding"),
            classEmbedding: required("\(embedding).class_embedding"),
            classPosition: required("\(embedding).cls_position_embedding"),
            preNorm: norm("\(prefix).pre_layernorm"), layers: layers,
            adapterIn: linear("\(prefix).proj_in"), adapterOut: linear("\(prefix).proj_out"))
    }
}
