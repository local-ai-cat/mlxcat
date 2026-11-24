import Foundation
import MLX

struct InklingExperts {
    struct Projection {
        enum Format {
            case nvfp4(scale: MLXArray)
            case mxfp8
            case affine(biases: MLXArray, groupSize: Int, bits: Int)
        }

        let weight: MLXArray
        let scales: MLXArray
        let format: Format
        private let inputPadding: Int
        private static let usesPackedGroupedPrefill =
            ProcessInfo.processInfo.environment["MLXCAT_INKLING_PACKED_PREFILL"] != "0"

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
            case .affine(let biases, let groupSize, let bits):
                return gatherQuantizedMM(
                    input, weight, scales: scales, biases: biases, rhsIndices: indices,
                    groupSize: groupSize, bits: bits, mode: .affine, sortedIndices: sorted)
            case .mxfp8:
                let values = inputPadding == 0 ? input
                    : Self.extending(input, columns: inputPadding, fill: 0)
                return gatherQuantizedMM(
                    values, weight, scales: scales, biases: nil, rhsIndices: indices,
                    groupSize: 32, bits: 8, mode: .mxfp8, sortedIndices: sorted)
            case .nvfp4(let globalScale):
                // MLX's grouped matrix kernel rounds the scaled tile to the
                // input dtype before GEMM. Reproduce that on the pinned core
                // with one transient dense projection. The vector kernel
                // applies the scale after FP32 accumulation.
                let grouped = sorted && indices.size >= 16 && indices.size / weight.dim(0) >= 4
                if grouped, globalScale.size == 1 || globalScale.size == weight.dim(0),
                    Device.defaultDevice().deviceType == .gpu
                {
                    let tensorScale = globalScale.asType(.float32)
                    let nvfp4Range: Float = 448 * 6
                    let factor = (tensorScale * nvfp4Range) / nvfp4Range
                    if Self.usesPackedGroupedPrefill, input.ndim == 3, input.dim(1) == 1,
                        input.dim(-1).isMultiple(of: 32), weight.dim(1).isMultiple(of: 32),
                        input.dtype == .bfloat16
                    {
                        return InklingNVFP4Grouped.run(
                            input, weights: weight, scales: scales, factors: factor, indices: indices)
                    }
                    let scaled = InklingNVFP4Unpack.run(
                        weights: weight, scales: scales, factor: factor,
                        dtype: input.dtype)
                    let projected = gatherMM(
                        input, scaled.transposed(0, 2, 1),
                        rhsIndices: indices, sortedIndices: true)
                    // Finish this projection before building the next one so
                    // the temporary weights never accumulate across visits.
                    eval(projected)
                    return projected
                }
                // mlx-vlm quantization/nvfp4.py uses this FP32 fallback for
                // per-row scales. The pinned gather API has no globalScale.
                let result = gatherQuantizedMM(
                    input.asType(.float32), weight, scales: scales, biases: nil,
                    rhsIndices: indices, groupSize: 16, bits: 4, mode: .nvfp4, sortedIndices: sorted)
                return applyNVFP4Scale(result, scale: globalScale, indices: indices, dtype: input.dtype)
            }
        }

        func applyNVFP4Scale(_ result: MLXArray, scale: MLXArray, indices: MLXArray, dtype: DType) -> MLXArray {
            let factor: MLXArray
            if scale.size == 1 {
                factor = scale.asType(.float32).reshaped([])
            } else {
                let rows = scale.asType(.float32).reshaped(weight.dim(0), -1)
                factor = rows[indices].expandedDimensions(axis: -2)
            }
            return (result * factor).asType(dtype)
        }
    }

    struct PairedNVFP4 {
        let weight: MLXArray
        let scales: MLXArray

        init?(up: Projection, gate: Projection) {
            guard case .nvfp4 = up.format, case .nvfp4 = gate.format,
                up.weight.shape == gate.weight.shape, up.scales.shape == gate.scales.shape,
                up.weight.dim(1).isMultiple(of: 8)
            else { return nil }
            weight = concatenated([up.weight, gate.weight], axis: 1)
            scales = concatenated([up.scales, gate.scales], axis: 1)
        }

        func projection(rows: Range<Int>, format: Projection.Format) -> Projection {
            Projection(
                weight: weight[0..., rows, 0...], scales: scales[0..., rows, 0...],
                format: format)
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
    var paired: PairedNVFP4? = nil

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
        let upValues: MLXArray
        let gateValues: MLXArray
        if let paired, !sorted, indices.size == topK, topK == 8,
            case .nvfp4(let upScale) = up.format, case .nvfp4(let gateScale) = gate.format
        {
            let both = gatherQuantizedMM(
                input.asType(.float32), paired.weight, scales: paired.scales, biases: nil,
                rhsIndices: indices, groupSize: 16, bits: 4, mode: .nvfp4)
            let width = up.weight.dim(1)
            upValues = up.applyNVFP4Scale(
                both[.ellipsis, ..<width], scale: upScale, indices: indices, dtype: input.dtype)
            gateValues = gate.applyNVFP4Scale(
                both[.ellipsis, width...], scale: gateScale, indices: indices, dtype: input.dtype)
        } else {
            upValues = up(input, indices: indices, sorted: sorted)
            gateValues = gate(input, indices: indices, sorted: sorted)
        }
        let hidden = InklingMath.gatedActivation(
            up: upValues, gate: gateValues, upCap: upCap, gateCap: gateCap)
        return down(hidden, indices: indices, sorted: sorted).squeezed(axis: -2)
    }
}
