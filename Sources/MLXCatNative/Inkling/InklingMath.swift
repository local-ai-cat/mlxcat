import MLX

/// Numerical primitives from mlx-vlm's Inkling 2 language.py at 466069f.
enum InklingMath {
    static func normalized(_ value: MLXArray, epsilon: Float) -> MLXArray {
        let floatValue = value.asType(.float32)
        let weight = MLXArray.ones([value.dim(-1)], dtype: .float32)
        return MLXFast.rmsNorm(floatValue, weight: weight, eps: epsilon)
    }

    static func messageBank(_ states: [MLXArray]) -> MLXArray {
        normalized(stacked(states, axis: -2), epsilon: 1e-6)
    }

    static func messages(query: MLXArray, bank: MLXArray) -> MLXArray {
        let scale = 1 / Float(bank.dim(-1)).squareRoot()
        let scores = matmul(query.asType(.float32), bank.swappedAxes(-1, -2)) * scale
        return matmul(softmax(scores, axis: -1), bank)
    }

    static func attentionResidual(
        states: [MLXArray], query: MLXArray, normalizeValues: Bool
    ) -> MLXArray {
        let values = stacked(states, axis: -2).asType(.float32)
        let keys = normalized(values, epsilon: 1e-6)
        let width = keys.dim(-1)
        let scale = 1 / Float(width).squareRoot()
        let scores = matmul(keys, query.asType(.float32).reshaped(-1, width).transposed()) * scale
        let weights = softmax(scores, axis: -2).swappedAxes(-1, -2)
        return matmul(weights, normalizeValues ? keys : values)
    }

    static func route(
        _ input: MLXArray, weight: MLXArray, bias: MLXArray, count: Int, scale: Float
    ) -> (indices: MLXArray, weights: MLXArray) {
        let scores = sigmoid(matmul(input.asType(.float32), weight.asType(.float32).transposed()))
        let partitioned = argPartition(-(scores + bias), kth: count - 1, axis: -1)
        let indices = partitioned[.ellipsis, ..<count]
        let selected = takeAlong(scores, indices, axis: -1)
        let weights = selected / (selected.sum(axis: -1, keepDims: true) + Float(1e-20)) * scale
        return (indices, weights)
    }

    static func gatedActivation(
        up: MLXArray, gate: MLXArray, upCap: Float, gateCap: Float
    ) -> MLXArray {
        let gate32 = gate.asType(.float32)
        let up32 = up.asType(.float32)
        return (tanh(gate32 / gateCap) * gateCap * sigmoid(gate32)
            * tanh(up32 / upCap) * upCap).asType(gate.dtype)
    }
}
