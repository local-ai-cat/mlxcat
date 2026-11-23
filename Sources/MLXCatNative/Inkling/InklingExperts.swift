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
        private let inputPadding: Int

        init(weight: MLXArray, scales: MLXArray, format: Format) {
            self.format = format
            // MLX 0.32 chooses its fast 8-bit vector reduction at multiples
            // of 256. MLX 0.31.1 requires 512. Extend the packed columns once
            // so Inkling's 2,304-wide MXFP8 projections use the same reduction.
            let alignment = 512
            let valuesPerWord = 4
            let scaleGroup = 32
            let width = weight.dim(-1) * valuesPerWord
            if case .mxfp8 = format, width % alignment == alignment / 2,
                weight.dim(-2).isMultiple(of: 8)
            {
                let padding = alignment / 2
                inputPadding = padding
                self.weight = Self.extending(weight, columns: padding / valuesPerWord, fill: 0)
                // E8M0 exponent bias. The added weights are exactly zero.
                self.scales = Self.extending(scales, columns: padding / scaleGroup, fill: 127)
            } else {
                inputPadding = 0
                self.weight = weight
                self.scales = scales
            }
        }

        private static func extending(_ array: MLXArray, columns: Int, fill: Int) -> MLXArray {
            var shape = array.shape
            shape[shape.count - 1] = columns
            let extra = broadcast(MLXArray(fill, dtype: array.dtype), to: shape)
            return concatenated([array, extra], axis: -1)
        }

        func callAsFunction(
            _ input: MLXArray, indices: MLXArray, sorted: Bool = false
        ) -> MLXArray {
            switch format {
            case .mxfp8:
                let values = inputPadding == 0 ? input
                    : Self.extending(input, columns: inputPadding, fill: 0)
                return gatherQuantizedMM(
                    values, weight, scales: scales, biases: nil, rhsIndices: indices,
                    groupSize: 32, bits: 8, mode: .mxfp8, sortedIndices: sorted)
            case .nvfp4(let globalScale):
                // mlx-vlm quantization/nvfp4.py uses this FP32 fallback for
                // per-row scales. The pinned gather API has no globalScale.
                let result = gatherQuantizedMM(
                    input.asType(.float32), weight, scales: scales, biases: nil,
                    rhsIndices: indices, groupSize: 16, bits: 4, mode: .nvfp4, sortedIndices: sorted)
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
        let values: MLXArray
        if routes.indices.size >= 64 {
            // mlx-vlm models/switch_layers.py groups prefill rows by expert,
            // restoring the router order before the FP32 reduction.
            let flattened = routes.indices.reshaped(-1)
            let order = argSort(flattened)
            let source = floorDivide(order, topK)
            let sortedInputs = input.reshaped(-1, 1, input.dim(-1)).take(source, axis: 0)
            let sortedValues = project(sortedInputs, indices: flattened[order], sorted: true)
            values = sortedValues.take(argSort(order), axis: 0)
                .reshaped(routes.indices.shape + [sortedValues.dim(-1)])
        } else {
            let expanded = input.expandedDimensions(axes: [-3, -2])
            values = project(expanded, indices: routes.indices, sorted: false)
        }
        let weighted = values.asType(.float32) * routes.weights.expandedDimensions(axis: -1)
        return weighted.sum(axis: -2).asType(input.dtype)
    }

    private func project(_ input: MLXArray, indices: MLXArray, sorted: Bool) -> MLXArray {
        let upValues = up(input, indices: indices, sorted: sorted)
        let gateValues = gate(input, indices: indices, sorted: sorted)
        let hidden = InklingMath.gatedActivation(
            up: upValues, gate: gateValues, upCap: upCap, gateCap: gateCap)
        return down(hidden, indices: indices, sorted: sorted).squeezed(axis: -2)
    }
}
