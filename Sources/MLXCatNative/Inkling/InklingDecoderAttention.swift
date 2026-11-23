import MLX

struct InklingAttentionState {
    var key: InklingConvolution
    var value: InklingConvolution
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0

    init(taps: Int) {
        key = InklingConvolution(taps: taps)
        value = InklingConvolution(taps: taps)
    }

    mutating func append(
        keys incomingKeys: MLXArray, values incomingValues: MLXArray, window: Int?
    ) -> (MLXArray, MLXArray) {
        let updatedKeys = keys.map { concatenated([$0, incomingKeys], axis: 2) } ?? incomingKeys
        let updatedValues = values.map { concatenated([$0, incomingValues], axis: 2) } ?? incomingValues
        let start = window.map { max(0, updatedKeys.dim(2) - $0 + 1) } ?? 0
        keys = updatedKeys[0..., 0..., start..., 0...]
        values = updatedValues[0..., 0..., start..., 0...]
        offset += incomingKeys.dim(2)
        return (updatedKeys, updatedValues)
    }
}

struct InklingDecoderAttention {
    enum ShortWeights {
        case fixed(MLXArray)
        case dynamic(staticWeight: MLXArray, projection: MLXArray)

        func apply(
            _ input: MLXArray, features: MLXArray, heads: Int, state: inout InklingConvolution
        ) -> MLXArray {
            switch self {
            case .fixed(let weight):
                return state.apply(input, weight: weight, outputType: input.dtype)
            case .dynamic(let weight, let projection):
                return state.applyDynamic(
                    input, staticWeight: weight,
                    projectedWeight: matmul(features, projection.transposed()), heads: heads)
            }
        }
    }

    let config: InklingConfiguration.Text
    let window: Int?
    let queryWeight: MLXArray
    let keyWeight: MLXArray
    let valueWeight: MLXArray
    let relativeWeight: MLXArray
    let outputWeight: MLXArray
    let gateWeight: MLXArray
    let sinkWeight: MLXArray
    let relativeProjection: MLXArray
    let queryScale: MLXArray
    let relativeScale: MLXArray
    let keyConvolution: ShortWeights
    let valueConvolution: ShortWeights

    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray, state: inout InklingAttentionState
    ) -> MLXArray {
        let batch = query.dim(0)
        let length = query.dim(1)
        let headWidth = config.headWidth
        let heads = config.heads
        let kvHeads = config.keyValueHeads
        let keyInput = matmul(key, keyWeight.transposed())
        let valueInput = matmul(value, valueWeight.transposed())
        let k = keyConvolution.apply(keyInput, features: key, heads: kvHeads, state: &state.key)
        let v = valueConvolution.apply(valueInput, features: value, heads: kvHeads, state: &state.value)
        var queries = clipped(
            matmul(query, queryWeight.transposed()).reshaped(batch, length, heads, headWidth),
            scale: queryScale, threshold: config.queryClip)
        var relative = clipped(
            matmul(query, relativeWeight.transposed()).reshaped(batch, length, heads, config.relativeWidth),
            scale: relativeScale, threshold: config.relativeClip)
        if window == nil, let floor = config.logScaleFloor {
            let positions = arange(length).asType(.float32) + Float(state.offset + 1)
            let temperature = 1 + config.logScaleAlpha * log(maximum(positions / Float(floor), 1))
            let adjusted = temperature.reshaped(1, length, 1, 1)
            queries = (queries.asType(.float32) * adjusted).asType(query.dtype)
            relative = (relative.asType(.float32) * adjusted).asType(query.dtype)
        }
        let keys = clipped(k.reshaped(batch, length, kvHeads, headWidth), threshold: config.queryClip)
        let values = InklingMath.normalized(
            v.reshaped(batch, length, kvHeads, headWidth), epsilon: config.rmsEpsilon)
            .asType(query.dtype)
        let (cachedKeys, cachedValues) = state.append(
            keys: keys.transposed(0, 2, 1, 3), values: values.transposed(0, 2, 1, 3),
            window: window)
        let sink = matmul(query, sinkWeight.transposed())
        let mask = InklingAttention.mask(
            relative: relative, projection: relativeProjection.asType(query.dtype),
            sink: sink, keyCount: cachedKeys.dim(2), window: window)
        let zero = MLXArray.zeros([batch, kvHeads, 1, headWidth], dtype: query.dtype)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries.transposed(0, 2, 1, 3),
            keys: concatenated([cachedKeys, zero], axis: 2),
            values: concatenated([cachedValues, zero], axis: 2),
            scale: 1 / Float(headWidth).squareRoot(), mask: .array(mask))
        let merged = output.transposed(0, 2, 1, 3).reshaped(batch, length, heads * headWidth)
        let gate = sigmoid(matmul(query, gateWeight.transposed()).asType(.float32))
        return matmul((merged.asType(.float32) * gate).asType(query.dtype), outputWeight.transposed())
    }

    private func clipped(_ value: MLXArray, scale: MLXArray? = nil, threshold: Float) -> MLXArray {
        var floatValue = value.asType(.float32)
        if let scale {
            floatValue = floatValue * scale.asType(.float32)
        }
        let norm = sqrt((floatValue * floatValue).sum(axis: -1, keepDims: true))
        return (floatValue * minimum(threshold / (norm + Float(1e-6)), 1)).asType(value.dtype)
    }
}
