import MLX

struct InklingTextModel {
    struct VisitState {
        var attention: InklingAttentionState
        var attentionOutput: InklingConvolution
        var mlpOutput: InklingConvolution

        init(taps: Int) {
            attention = InklingAttentionState(taps: taps)
            attentionOutput = InklingConvolution(taps: taps)
            mlpOutput = InklingConvolution(taps: taps)
        }
    }

    struct Controller {
        let down: MLXArray
        let forget: MLXArray
        let gain: MLXArray

        func callAsFunction(_ value: MLXArray, message: MLXArray, epsilon: Float) -> MLXArray {
            let features = concatenated(
                [InklingMath.normalized(value, epsilon: epsilon),
                 InklingMath.normalized(message, epsilon: epsilon)], axis: -1)
            let hidden = matmul(features.asType(down.dtype), down.transposed()).asType(.float32)
            return (1 - tanh(matmul(hidden, forget.transposed()))) * value
                + tanh(matmul(hidden, gain.transposed())) * message
        }
    }

    struct Layer {
        let attention: InklingDecoderAttention
        let experts: InklingExperts
        let inputNorms: [MLXArray]
        let postAttentionNorm: MLXArray
        let attentionQuery: MLXArray?
        let mlpQuery: MLXArray
        let attentionConvolution: MLXArray
        let mlpConvolution: MLXArray
        let controllers: [Controller]

        func callAsFunction(
            _ stream: inout InklingLoop.ResidualStream<MLXArray>,
            messages: MLXArray?, boundary: Bool, state: inout VisitState
        ) {
            let config = attention.config
            let epsilon = config.rmsEpsilon
            let input = stream.partial
            let inputs: MLXArray
            if let attentionQuery {
                inputs = InklingMath.attentionResidual(
                    states: stream.completed + [input], query: attentionQuery,
                    normalizeValues: config.normalizeResidual)
            } else {
                inputs = input.expandedDimensions(axis: -2)
            }
            let parts = (0..<3).map { index in
                var part = inputs[.ellipsis, min(index, inputs.dim(-2) - 1), 0...]
                if let messages {
                    part = controllers[index](
                        part, message: messages[.ellipsis, index, 0...], epsilon: epsilon)
                }
                return MLXFast.rmsNorm(
                    part, weight: inputNorms[min(index, inputNorms.count - 1)], eps: epsilon)
                    .asType(input.dtype)
            }
            let attended = attention(query: parts[0], key: parts[1], value: parts[2], state: &state.attention)
            let residual = state.attentionOutput.apply(
                attended, weight: attentionConvolution, residual: boundary ? nil : input,
                outputType: .float32)
            if boundary { stream.completed.append(input) }
            var hidden = InklingMath.attentionResidual(
                states: stream.completed + [residual], query: mlpQuery,
                normalizeValues: config.normalizeResidual).squeezed(axis: -2)
            if let messages {
                hidden = controllers[3](
                    hidden, message: messages[.ellipsis, 3, 0...], epsilon: epsilon)
            }
            let normalized = MLXFast.rmsNorm(hidden, weight: postAttentionNorm, eps: epsilon)
                .asType(input.dtype)
            stream.partial = state.mlpOutput.apply(
                experts(normalized), weight: mlpConvolution, residual: residual, outputType: input.dtype)
        }
    }

    let config: InklingConfiguration.Text
    let loop: InklingLoop
    let layers: [Layer]
    let embeddings: InklingProjection
    let embeddingNorm: MLXArray
    let messageQueries: MLXArray
    let finalQuery: MLXArray
    let finalNorm: MLXArray
    /// Dense checkpoints retain the FP32 conversion across decode steps.
    let vocabulary: InklingProjection

    func makeCache() -> [VisitState] {
        loop.visits.map { _ in VisitState(taps: config.convolutionSize) }
    }

    func embed(_ tokens: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(embeddings.rows(tokens), weight: embeddingNorm, eps: config.rmsEpsilon)
    }

    func forward(
        embeddings: MLXArray, cache: inout [VisitState], projectLogits: Bool = true
    ) -> MLXArray {
        precondition(cache.count == loop.visits.count)
        let residual = loop.forward(embedding: embeddings, makeBank: InklingMath.messageBank) {
            slot, visit, bank, stream in
            let messages = bank.map {
                InklingMath.messages(query: messageQueries[visit.layer - config.loop.start], bank: $0)
            }
            layers[visit.layer](
                &stream, messages: messages,
                boundary: visit.layer.isMultiple(of: config.residualBlockSize), state: &cache[slot])
        }
        guard projectLogits else { return residual.partial }
        let hidden = InklingMath.attentionResidual(
            states: residual.completed + [residual.partial], query: finalQuery,
            normalizeValues: config.normalizeResidual).squeezed(axis: -2)
        let normalized = MLXFast.rmsNorm(hidden, weight: finalNorm, eps: config.rmsEpsilon)
            .asType(embeddingNorm.dtype).asType(.float32)
        let logits = vocabulary(normalized)
        let capped = tanh(logits / config.logitCap) * config.logitCap
        return capped[.ellipsis, ..<config.vocabularySize]
    }
}
