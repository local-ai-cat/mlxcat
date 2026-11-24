import Foundation
import MLX

/// Checkpoint-backed DSpark head used by the speculative verifier probe.
struct InklingDraftProbe {
    struct Configuration: Decodable {
        let numHiddenLayers: Int
        let headDim: Int
        let rmsNormEps: Float
        let ropeTheta: Float
        let localLayerIds: [Int]
        let slidingWindowSize: Int
        let blockSize: Int
        let markovRank: Int
        let targetLayerIds: [Int]
    }

    struct Context {
        var keys: MLXArray?
        var values: MLXArray?
        var offset = 0
    }

    let config: Configuration
    private let weights: [String: MLXArray]
    private let blend: MLXArray
    private let vocabulary: InklingProjection
    private let vocabularySize: Int
    private var contexts: [Context]
    private var previousFeatures: MLXArray?

    init(directory: URL, model: InklingTextModel) throws {
        struct Checkpoint: Decodable { let mtpConfig: Configuration }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        config = try decoder.decode(
            Checkpoint.self, from: Data(contentsOf: directory.appending(component: "config.json"))
        ).mtpConfig
        let weights = try loadArrays(url: directory.appending(component: "mtp.safetensors"))
        guard let fuse = weights["fuse_weight"], config.numHiddenLayers > 0,
              config.headDim > 0, config.blockSize > 1, config.markovRank > 0,
              config.slidingWindowSize > 0, config.targetLayerIds.count == 4,
              Set(config.targetLayerIds).count == config.targetLayerIds.count,
              config.targetLayerIds.allSatisfy({ (0...model.loop.visits.count).contains($0) }),
              weights["mask_embed"]?.shape == [model.config.hiddenSize],
              fuse.shape == [config.numHiddenLayers + 1, config.targetLayerIds.count]
        else { throw DraftError.unsupportedHead }
        self.weights = weights
        blend = softmax(fuse.asType(.float32), axis: -1).asType(fuse.dtype)
        vocabulary = model.vocabulary
        vocabularySize = model.config.vocabularySize
        contexts = Array(repeating: Context(), count: config.numHiddenLayers)
        try validateWeights()
    }

    private func validateWeights() throws {
        let required = ["mask_embed", "norm.weight", "markov_head.dflare_norm.weight",
                        "markov_head.hidden_out.weight", "markov_head.hidden_out.bias",
                        "markov_head.markov_w1.weight", "markov_head.rnn_proj.weight",
                        "markov_head.rnn_proj.bias"]
        let block = ["dflare_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
                     "self_attn.k_proj_ctx.weight", "self_attn.v_proj_ctx.weight", "self_attn.q_proj.weight",
                     "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                     "self_attn.k_norm_ctx.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                     "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
        let layerWeights = (0..<config.numHiddenLayers).flatMap { layer in block.map { "layers.\(layer).\($0)" } }
        guard (required + layerWeights).allSatisfy({ weights[$0] != nil }) else {
            throw DraftError.unsupportedHead
        }
    }

    mutating func observe(_ features: [MLXArray]) {
        for layer in contexts.indices {
            let prefix = "layers.\(layer)."
            let mixed = mix(features, row: layer, norm: prefix + "dflare_norm.weight")
            let k = heads(mixed, prefix: prefix, projection: "k_proj_ctx")
            let key = rotate(normalize(k, prefix + "self_attn.k_norm_ctx.weight"), at: contexts[layer].offset)
            let value = heads(mixed, prefix: prefix, projection: "v_proj_ctx")
            let keys = contexts[layer].keys.map { concatenated([$0, key], axis: 2) } ?? key
            let values = contexts[layer].values.map { concatenated([$0, value], axis: 2) } ?? value
            let keep = config.localLayerIds.contains(layer) ? config.slidingWindowSize : keys.dim(2)
            let start = max(0, keys.dim(2) - keep)
            contexts[layer].keys = keys[0..., 0..., start...]
            contexts[layer].values = values[0..., 0..., start...]
            contexts[layer].offset += key.dim(2)
        }
        previousFeatures = mix(features.map { $0[0..., (-1)...] }, row: -1, norm: "markov_head.dflare_norm.weight")
        eval(contexts.flatMap { [$0.keys!, $0.values!] } + [previousFeatures!])
    }

    private func hiddenPanel(anchor: Int, model: InklingTextModel) -> MLXArray {
        let embedded = model.embed(MLXArray([anchor]).reshaped(1, 1))
        let dtype = embedded.dtype
        let masks = broadcast(weights["mask_embed"]!, to: [1, config.blockSize - 1, embedded.dim(-1)])
        var hidden = concatenated([embedded, masks], axis: 1)
        var residual: MLXArray?
        for (layer, context) in contexts.enumerated() {
            let prefix = "layers.\(layer)."
            let summed = residual.map { hidden.asType(.float32) + $0.asType(.float32) } ?? hidden
            let x = normalize(summed, prefix + "input_layernorm.weight").asType(dtype)
            let q = rotate(normalize(heads(x, prefix: prefix, projection: "q_proj"),
                                     prefix + "self_attn.q_norm.weight"), at: context.offset)
            let k = rotate(normalize(heads(x, prefix: prefix, projection: "k_proj"),
                                     prefix + "self_attn.k_norm.weight"), at: context.offset)
            let v = heads(x, prefix: prefix, projection: "v_proj")
            let attention = MLXFast.scaledDotProductAttention(
                queries: q, keys: concatenated([context.keys!, k], axis: 2),
                values: concatenated([context.values!, v], axis: 2),
                scale: 1 / Float(config.headDim).squareRoot(), mask: .none)
            let merged = attention.transposed(0, 2, 1, 3).reshaped(1, config.blockSize, -1)
            let nextResidual = summed.asType(dtype).asType(.float32)
                + linear(merged, prefix + "self_attn.o_proj.weight").asType(.float32)
            let normalized = normalize(nextResidual, prefix + "post_attention_layernorm.weight").asType(dtype)
            let gate = linear(normalized, prefix + "mlp.gate_proj.weight").asType(.float32)
            let up = linear(normalized, prefix + "mlp.up_proj.weight").asType(.float32)
            hidden = linear((gate * sigmoid(gate) * up).asType(dtype), prefix + "mlp.down_proj.weight")
            residual = nextResidual.asType(dtype)
        }
        return normalize(hidden.asType(.float32) + residual!.asType(.float32), "norm.weight").asType(dtype)
    }

    func propose(anchor: Int, count: Int, model: InklingTextModel) -> [Int] {
        guard let previousFeatures else { return [] }
        let panel = hiddenPanel(anchor: anchor, model: model)
        let dtype = panel.dtype
        let rank = config.markovRank
        let markov = weights["markov_head.rnn_proj.weight"]!.asType(.float32)
        var previous = MLXArray([anchor])
        var recurrent = MLXArray.zeros([1, rank], dtype: dtype)
        var guesses: [MLXArray] = []
        for row in 1...count {
            let state = panel[0..., row]
            let dynamic = concatenated([recurrent, weights["markov_head.markov_w1.weight"]![previous]], axis: -1)
            let conditioned = concatenated([state, previousFeatures.reshaped(1, -1)], axis: -1)
            let update = (matmul(concatenated([dynamic, conditioned], axis: -1).asType(.float32), markov.transposed())
                + weights["markov_head.rnn_proj.bias"]!.asType(.float32)).asType(dtype)
            let gate = sigmoid(update[0..., ..<rank].asType(.float32)).asType(dtype)
            let candidate = tanh(update[0..., rank..<(2 * rank)].asType(.float32)).asType(dtype)
            recurrent = gate * recurrent + (1 - gate) * candidate
            let signal = tanh(update[0..., (2 * rank)...].asType(.float32)).asType(dtype)
            let delta = linear(signal, "markov_head.hidden_out.weight") + weights["markov_head.hidden_out.bias"]!
            let scores = vocabulary((state + delta).asType(.float32))[0, ..<vocabularySize]
            previous = argMax(scores).reshaped(1)
            guesses.append(previous)
        }
        return concatenated(guesses).asArray(Int.self)
    }

    private func mix(_ features: [MLXArray], row: Int, norm: String) -> MLXArray {
        let terms = features.enumerated().map { index, feature in
            feature.asType(.float32) * blend[row, index].asType(.float32)
        }
        return normalize(terms.dropFirst().reduce(terms[0], +), norm).asType(features[0].dtype)
    }

    private func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        matmul(x, weights[key]!.transposed())
    }

    private func normalize(_ x: MLXArray, _ key: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weights[key]!, eps: config.rmsNormEps)
    }

    private func heads(_ x: MLXArray, prefix: String, projection: String) -> MLXArray {
        linear(x, prefix + "self_attn." + projection + ".weight")
            .reshaped(1, x.dim(1), -1, config.headDim).transposed(0, 2, 1, 3)
    }

    private func rotate(_ x: MLXArray, at offset: Int) -> MLXArray {
        MLXFast.RoPE(x, dimensions: config.headDim, traditional: false,
                     base: config.ropeTheta, scale: 1, offset: offset)
    }

    enum DraftError: Error { case unsupportedHead }
}
