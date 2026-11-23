import MLX

/// Apply the packed group's E4M3 scale and the expert's tensor scale before
/// rounding its E2M1 weights to the grouped GEMM input dtype.
enum InklingNVFP4Unpack {
    static func run(weights: MLXArray, scales: MLXArray, factor: MLXArray, dtype: DType) -> MLXArray {
        let experts = weights.dim(0)
        let rows = weights.dim(1)
        let valuesPerPackedWord = 8
        let columns = weights.dim(2) * valuesPerPackedWord
        return kernel(
            [weights, scales, broadcast(factor.reshaped(-1), to: [experts])],
            template: [("T", dtype)], grid: (weights.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[experts, rows, columns]], outputDTypes: [dtype])[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mlxcat_inkling_nvfp4_grouped_unpack",
        inputNames: ["packed", "scales", "factor"],
        outputNames: ["unpacked"],
        source: """
            const uint word = thread_position_in_grid.x;
            const uint wordsPerRow = packed_shape[2];
            const uint wordsPerExpert = wordsPerRow * packed_shape[1];
            if (word >= packed_shape[0] * wordsPerExpert) return;
            const uint bits = packed[word];
            const uchar scaleBits = scales[word / 2];
            const ushort e4m3 = ushort(((scaleBits & 127u) << 7) | ((scaleBits & 128u) << 8));
            const float blockScale = float(as_type<half>(e4m3) * half(256)) * factor[word / wordsPerExpert];
            #pragma unroll
            for (uint offset = 0; offset < 8; ++offset) {
                const uint value = (bits >> (offset * 4)) & 15u;
                const uint e = (value >> 1) & 3u;
                const float fraction = (value & 1u) ? 0.5f : 0.0f;
                const float magnitude = e == 0 ? fraction : (1.0f + fraction) * float(1u << (e - 1u));
                unpacked[word * 8 + offset] = T(blockScale * ((value & 8u) ? -magnitude : magnitude));
            }
            """)
}
