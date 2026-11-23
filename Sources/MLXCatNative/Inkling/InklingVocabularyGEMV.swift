import MLX

/// Keep the vocabulary table in 16-bit storage and use the pinned MLX GEMV's
/// FP32 lane order. MTPLX applies the same mixed storage/accumulator idea to
/// its Inkling vocabulary reader.
enum InklingVocabularyGEMV {
    static func supports(_ input: MLXArray, weight: MLXArray) -> Bool {
        let width = weight.dim(-1)
        let rows = weight.dim(0)
        return input.dtype == .float32 && input.size == width && width > 64 && width.isMultiple(of: 128)
            && rows >= 4096 && rows.isMultiple(of: 4)
            && (weight.dtype == .bfloat16 || weight.dtype == .float16)
            && Device.defaultDevice().deviceType == .gpu
    }

    static func run(_ input: MLXArray, weight: MLXArray) -> MLXArray {
        let rows = weight.dim(0)
        let width = weight.dim(1)
        return kernel(
            [input, weight],
            template: [("WIDTH", width), ("ROWS", rows)],
            grid: (rows * 8, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [Array(input.shape.dropLast()) + [rows]], outputDTypes: [.float32])[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mlxcat_inkling_vocabulary_gemv",
        inputNames: ["input", "weight"], outputNames: ["output"],
        source: """
            constexpr uint rowsPerSIMD = 4;
            constexpr uint inputsPerLane = 4;
            constexpr uint inputBlock = 32 * inputsPerLane;
            const uint row = thread_position_in_grid.x / 32 * rowsPerSIMD;
            const uint lane = thread_index_in_simdgroup;
            if (row >= ROWS) return;
            float sum[rowsPerSIMD] = {};
            for (uint block = 0; block < WIDTH; block += inputBlock) {
                const uint column = block + lane * inputsPerLane;
                float values[inputsPerLane];
                #pragma unroll
                for (uint k = 0; k < inputsPerLane; ++k) values[k] = input[column + k];
                #pragma unroll
                for (uint r = 0; r < rowsPerSIMD; ++r) {
                    #pragma unroll
                    for (uint k = 0; k < inputsPerLane; ++k) {
                        sum[r] += float(weight[size_t(row + r) * WIDTH + column + k]) * values[k];
                    }
                }
            }
            #pragma unroll
            for (uint r = 0; r < rowsPerSIMD; ++r) {
                #pragma unroll
                for (ushort offset = 16; offset >= 1; offset >>= 1) {
                    sum[r] += simd_shuffle_down(sum[r], offset);
                }
                if (lane == 0) output[row + r] = sum[r];
            }
            """)
}
