import MLX

struct InklingExperts {
    struct Projection {
        enum Format {
            case nvfp4(scale: MLXArray)
            case mxfp8
        }

        let weight: MLXArray
        let scales: MLXArray
        let format: Format

        func callAsFunction(_ input: MLXArray, indices: MLXArray) -> MLXArray {
            switch format {
            case .mxfp8:
                return gatherQuantizedMM(
                    input, weight, scales: scales, biases: nil, rhsIndices: indices,
                    groupSize: 32, bits: 8, mode: .mxfp8)
            case .nvfp4(let globalScale):
                // mlx-vlm quantization/nvfp4.py uses this FP32 fallback for
                // per-row scales. The pinned gather API has no globalScale.
                let result = gatherQuantizedMM(
                    input.asType(.float32), weight, scales: scales, biases: nil,
                    rhsIndices: indices, groupSize: 16, bits: 4, mode: .nvfp4)
                let factor: MLXArray
                if globalScale.size == 1 {
                    factor = globalScale.asType(.float32).reshaped([])
                } else {
                    let rows = globalScale.asType(.float32).reshaped(weight.dim(0), -1)
                    factor = rows[indices].expandedDimensions(axis: -2)
                }
                return (result * factor).asType(input.dtype)
            }
        }
    }

    let router: MLXArray
    let routingBias: MLXArray
    let up: Projection
    let gate: Projection
    let down: Projection
    let topK: Int
    let routeScale: Float
    let upCap: Float
    let gateCap: Float

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let routes = InklingMath.route(
            input, weight: router, bias: routingBias, count: topK, scale: routeScale)
        let expanded = input.expandedDimensions(axes: [-3, -2])
        let upValues = up(expanded, indices: routes.indices)
        let gateValues = gate(expanded, indices: routes.indices)
        let hidden = InklingMath.gatedActivation(
            up: upValues, gate: gateValues, upCap: upCap, gateCap: gateCap)
        let values = down(hidden, indices: routes.indices).squeezed(axis: -2)
        let weighted = values.asType(.float32) * routes.weights.expandedDimensions(axis: -1)
        return weighted.sum(axis: -2).asType(input.dtype)
    }
}
