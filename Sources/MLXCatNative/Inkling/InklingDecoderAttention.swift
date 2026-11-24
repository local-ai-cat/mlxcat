import Foundation
import MLX

struct InklingAttentionState {
    var key: InklingConvolution
    var value: InklingConvolution
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0
    private var globalKeys: MLXArray?
    private var globalValues: MLXArray?

    init(taps: Int) {
        key = InklingConvolution(taps: taps)
        value = InklingConvolution(taps: taps)
    }

    mutating func append(
        keys incomingKeys: MLXArray, values incomingValues: MLXArray, window: Int?
    ) -> (MLXArray, MLXArray) {
        globalKeys = nil
        globalValues = nil
        let updatedKeys = keys.map { concatenated([$0, incomingKeys], axis: 2) } ?? incomingKeys
        let updatedValues = values.map { concatenated([$0, incomingValues], axis: 2) } ?? incomingValues
        let start = window.map { max(0, updatedKeys.dim(2) - $0 + 1) } ?? 0
        keys = updatedKeys[0..., 0..., start..., 0...]
        values = updatedValues[0..., 0..., start..., 0...]
        offset += incomingKeys.dim(2)
        return (updatedKeys, updatedValues)
    }

    mutating func restore(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
        globalKeys = nil
        globalValues = nil
    }

    mutating func appendGlobalDecode(
        keys incomingKeys: MLXArray, values incomingValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let position = offset
        func appendRow(_ row: MLXArray, active: MLXArray?, reserved: MLXArray?) -> MLXArray {
            let backing: MLXArray
            if let reserved, reserved.dim(2) >= position + 2 {
                // Index assignment replaces this wrapper's graph. Pending
                // snapshots may still hold the original array.
                backing = reserved.reshaped(reserved.shape)
            } else {
                let reserveIncrement = 256
                let capacity = (position + 2 + reserveIncrement - 1) / reserveIncrement * reserveIncrement
                let zeros = MLXArray.zeros(
                    [row.dim(0), row.dim(1), capacity - position, row.dim(3)], dtype: row.dtype)
                backing = active.map { concatenated([$0, zeros], axis: 2) } ?? zeros
            }
            backing[0..., 0..., position ..< (position + 1), 0...] = row
            return backing
        }

        let nextKeys = appendRow(incomingKeys, active: keys, reserved: globalKeys)
        let nextValues = appendRow(incomingValues, active: values, reserved: globalValues)
        globalKeys = nextKeys
        globalValues = nextValues
        offset += 1
        keys = nextKeys[0..., 0..., ..<offset, 0...]
        values = nextValues[0..., 0..., ..<offset, 0...]
        // The next untouched row is the zero attention sink.
        return (nextKeys[0..., 0..., ..<(offset + 1), 0...], nextValues[0..., 0..., ..<(offset + 1), 0...])
    }
}

struct InklingDecoderAttention {
    private static let reservesGlobalDecodeKV =
        ProcessInfo.processInfo.environment["MLXCAT_INKLING_RESERVED_GLOBAL_KV"] != "0"
    private static let reservedGlobalStart = 8192

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
    let queryWeight: InklingProjection
    let keyWeight: InklingProjection
    let valueWeight: InklingProjection
    let relativeWeight: InklingProjection
    let outputWeight: InklingProjection
    let gateWeight: InklingProjection
    let sinkWeight: InklingProjection
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
        let keyInput = keyWeight(key)
        let valueInput = valueWeight(value)
        let k = keyConvolution.apply(keyInput, features: key, heads: kvHeads, state: &state.key)
        let v = valueConvolution.apply(valueInput, features: value, heads: kvHeads, state: &state.value)
        var queries = clipped(
            queryWeight(query).reshaped(batch, length, heads, headWidth),
            scale: queryScale, threshold: config.queryClip)
        var relative = clipped(
            relativeWeight(query).reshaped(batch, length, heads, config.relativeWidth),
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
        let paddedKeys: MLXArray
        let paddedValues: MLXArray
        let incomingKeys = keys.transposed(0, 2, 1, 3)
        let incomingValues = values.transposed(0, 2, 1, 3)
        if Self.reservesGlobalDecodeKV, window == nil, length == 1,
            state.offset >= Self.reservedGlobalStart
        {
            (paddedKeys, paddedValues) = state.appendGlobalDecode(keys: incomingKeys, values: incomingValues)
        } else {
            let (cachedKeys, cachedValues) = state.append(
                keys: incomingKeys, values: incomingValues, window: window)
            let zero = MLXArray.zeros([batch, kvHeads, 1, headWidth], dtype: query.dtype)
            paddedKeys = concatenated([cachedKeys, zero], axis: 2)
            paddedValues = concatenated([cachedValues, zero], axis: 2)
        }
        let sink = sinkWeight(query)
        let mask = InklingAttention.mask(
            relative: relative, projection: relativeProjection.asType(query.dtype),
            sink: sink, keyCount: paddedKeys.dim(2) - 1, window: window)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries.transposed(0, 2, 1, 3),
            keys: paddedKeys, values: paddedValues,
            scale: 1 / Float(headWidth).squareRoot(), mask: .array(mask))
        let merged = output.transposed(0, 2, 1, 3).reshaped(batch, length, heads * headWidth)
        let gate = sigmoid(gateWeight(query).asType(.float32))
        return outputWeight((merged.asType(.float32) * gate).asType(query.dtype))
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
