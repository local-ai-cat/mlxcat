import MLX

/// Sorted expert rows share a 16×32 output tile. Only the current K tile is
/// unpacked, so no dense expert bank is materialized in device memory.
enum InklingNVFP4Grouped {
    static func run(
        _ input: MLXArray, weights: MLXArray, scales: MLXArray, factors: MLXArray,
        indices: MLXArray
    ) -> MLXArray {
        let experts = weights.dim(0)
        let rows = indices.size
        let channels = weights.dim(1)
        let starts = offsets(
            [indices.asType(.int32)],
            grid: (experts, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[experts]], outputDTypes: [.int32])[0]
        let tileRows = 16
        let tileColumns = 32
        let tiles = (rows + tileRows - 1) / tileRows + experts - 1
        return kernel(
            [input, weights, scales, broadcast(factors.reshaped(-1), to: [experts]), starts],
            template: [("T", input.dtype)],
            grid: (channels / tileColumns * 64, tiles, 1), threadGroup: (64, 1, 1),
            outputShapes: [[rows, 1, channels]], outputDTypes: [input.dtype])[0]
    }

    private static let offsets = MLXFast.metalKernel(
        name: "mlxcat_inkling_expert_offsets", inputNames: ["indices"], outputNames: ["starts"],
        source: """
            const int expert = int(thread_position_in_grid.x);
            int lower = 0;
            int upper = int(indices_shape[0]);
            while (lower < upper) {
                const int middle = lower + (upper - lower) / 2;
                if (indices[middle] < expert) lower = middle + 1;
                else upper = middle;
            }
            starts[expert] = lower;
            """)

    private static let kernel = MLXFast.metalKernel(
        name: "mlxcat_inkling_nvfp4_grouped",
        inputNames: ["input", "packed", "scales", "factors", "starts"],
        outputNames: ["output"],
        source: """
            constexpr int rowTile = 16;
            constexpr int columnTile = 32;
            constexpr int kTile = 32;
            const int lanes = 32;
            const int threads = 64;
            const int inputRows = input_shape[0];
            const int channels = packed_shape[1];
            const int width = packed_shape[2] * 8;
            const int experts = packed_shape[0];
            const int lane = thread_index_in_simdgroup;
            const int local = thread_position_in_threadgroup.x;
            const int column = threadgroup_position_in_grid.x * columnTile;
            const int tile = threadgroup_position_in_grid.y;
            int precedingTiles = 0;
            int expert = -1;
            int row = 0;
            int rows = 0;

            // Assign packed row tiles using the same SIMD prefix scheme as
            // MLX's schedule_row_tile. This also skips unused experts.
            for (int base = 0; base < experts; base += lanes) {
                const int candidate = base + lane;
                const int start = candidate < experts ? starts[candidate] : inputRows;
                const int end = candidate + 1 < experts ? starts[candidate + 1] : inputRows;
                const int count = (end - start + rowTile - 1) / rowTile;
                const int through = precedingTiles + simd_prefix_inclusive_sum(count);
                const ushort owner = ushort(simd_sum(int(through <= tile)));
                if (owner < lanes) {
                    expert = base + owner;
                    row = simd_shuffle(start, owner) +
                        (tile - simd_shuffle(through - count, owner)) * rowTile;
                    rows = min(rowTile, simd_shuffle(end, owner) - row);
                    break;
                }
                precedingTiles = simd_shuffle(through, 31);
            }
            if (expert < 0) return;

            constexpr int kStride = kTile + 8;
            threadgroup T activations[rowTile * kStride];
            threadgroup T weights[columnTile * kStride];
            threadgroup float result[rowTile * columnTile];
            const int halfColumn = simdgroup_index_in_threadgroup * 16;
            const float factor = factors[expert];
            const size_t expertBase = size_t(expert) * channels + column;

            simdgroup_matrix<float, 8, 8> accumulator[2][2];
            for (int m = 0; m < 2; ++m) {
                for (int n = 0; n < 2; ++n) {
                    accumulator[m][n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            for (int k = 0; k < width; k += kTile) {
                for (int cell = local; cell < rowTile * kTile; cell += threads) {
                    const int m = cell / kTile;
                    const int q = cell % kTile;
                    activations[m * kStride + q] = m < rows
                        ? input[size_t(row + m) * width + k + q] : T(0);
                }
                for (int word = local; word < columnTile * (kTile / 8); word += threads) {
                    const int n = word / (kTile / 8);
                    const int q = word % (kTile / 8);
                    const size_t packedIndex = (expertBase + n) * (width / 8) + (k / 8) + q;
                    const uint bits = packed[packedIndex];
                    const uchar scaleBits = scales[packedIndex / 2];
                    const ushort e4m3 = ushort(((scaleBits & 127u) << 7) | ((scaleBits & 128u) << 8));
                    const float blockScale = float(as_type<half>(e4m3) * half(256)) * factor;
                    #pragma unroll
                    for (int valueIndex = 0; valueIndex < 8; ++valueIndex) {
                        const uint value = (bits >> (valueIndex * 4)) & 15u;
                        const uint exponent = (value >> 1) & 3u;
                        const float fraction = (value & 1u) ? 0.5f : 0.0f;
                        const float magnitude = exponent == 0
                            ? fraction : (1.0f + fraction) * float(1u << (exponent - 1u));
                        weights[n * kStride + q * 8 + valueIndex] =
                            T(blockScale * ((value & 8u) ? -magnitude : magnitude));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma unroll
                for (int q = 0; q < kTile; q += 8) {
                    simdgroup_matrix<T, 8, 8> left[2];
                    simdgroup_matrix<T, 8, 8> right[2];
                    #pragma unroll
                    for (int index = 0; index < 2; ++index) {
                        simdgroup_load(left[index], activations + index * 8 * kStride + q, kStride);
                        simdgroup_load(right[index], weights + (halfColumn + index * 8) * kStride + q,
                                       kStride, ulong2(0, 0), true);
                    }
                    #pragma unroll
                    for (int m = 0; m < 2; ++m) {
                        #pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            simdgroup_multiply_accumulate(accumulator[m][n], left[m], right[n], accumulator[m][n]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                #pragma unroll
                for (int n = 0; n < 2; ++n) {
                    simdgroup_store(accumulator[m][n], result + m * 8 * columnTile + halfColumn + n * 8,
                                    columnTile);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int cell = local; cell < rows * columnTile; cell += threads) {
                output[size_t(row + cell / columnTile) * channels + column + cell % columnTile] =
                    T(result[cell]);
            }
            """,
        header: "#include <metal_simdgroup_matrix>\n")
}
