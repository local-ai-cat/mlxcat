import MLX

/// FP32 history for an unpadded sequence. The four histories are indexed by
/// the execution visit, even when several visits share convolution weights.
struct InklingConvolution {
    var history: MLXArray?
    let taps: Int

    init(taps: Int) {
        precondition(taps > 1)
        self.taps = taps
    }

    // mlx-vlm language.py:174–225. Round the convolution skip to the input
    // dtype before adding the residual stream in FP32.
    mutating func apply(
        _ input: MLXArray, weight: MLXArray, residual: MLXArray? = nil, outputType: DType
    ) -> MLXArray {
        let value = input.asType(.float32)
        let context = append(value)
        let convolved = conv1d(context, weight.asType(.float32), groups: input.dim(-1))
        let skipped = ((convolved + value) / Float(taps + 1).squareRoot()).asType(input.dtype)
        guard let residual else { return skipped.asType(outputType) }
        return (skipped.asType(.float32) + residual.asType(.float32)).asType(outputType)
    }

    // mlx-vlm language.py:228–259. Tap zero reads the current token.
    mutating func applyDynamic(
        _ input: MLXArray, staticWeight: MLXArray, projectedWeight: MLXArray, heads: Int
    ) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        let channels = input.dim(2)
        let value = input.asType(.float32)
        let context = append(value)
        let weights = staticWeight.asType(.float32).transposed().reshaped(heads, -1, taps)
            + projectedWeight.asType(.float32).reshaped(batch, length, heads, 1, taps)
        var output = MLXArray.zeros([batch, length, heads, channels / heads], dtype: .float32)
        for lag in 0..<taps {
            let start = taps - 1 - lag
            let values = context[0..., start..<(start + length), 0...].reshaped(output.shape)
            output = output + values * weights[.ellipsis, lag]
        }
        return ((output.reshaped(input.shape) + value) / Float(2 * taps + 1).squareRoot())
            .asType(input.dtype)
    }

    private mutating func append(_ value: MLXArray) -> MLXArray {
        let width = taps - 1
        let state = history ?? MLXArray.zeros([value.dim(0), width, value.dim(2)], dtype: .float32)
        let context = concatenated([state, value], axis: 1)
        history = context[0..., value.dim(1)..., 0...]
        return context
    }
}
