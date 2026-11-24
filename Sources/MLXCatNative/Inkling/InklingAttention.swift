import MLX

enum InklingAttention {
    static func mask(
        relative: MLXArray, projection: MLXArray, sink: MLXArray, keyCount: Int, window: Int?
    ) -> MLXArray {
        if Device.defaultDevice().deviceType == .gpu {
            return metalMask(
                relative: relative, projection: projection, sink: sink, keyCount: keyCount, window: window)
        }
        return composedMask(
            relative: relative, projection: projection, sink: sink, keyCount: keyCount, window: window)
    }

    /// The mask is query-dependent. Its last column is the sink score,
    /// corresponding to a zero key and a zero value.
    static func composedMask(
        relative: MLXArray, projection: MLXArray, sink: MLXArray, keyCount: Int,
        window: Int?
    ) -> MLXArray {
        let length = relative.dim(1)
        let relativeScores = matmul(relative, projection).transposed(0, 2, 1, 3)
        let extent = projection.dim(-1)
        let queryPositions = arange(length) + (keyCount - length)
        let distances = queryPositions.expandedDimensions(axis: 1) - arange(keyCount)
        let distanceGrid = distances.reshaped(1, 1, length, keyCount)
        let indices = broadcast(
            clip(distanceGrid, min: 0, max: extent - 1),
            to: [relative.dim(0), relative.dim(2), length, keyCount])
        let band = takeAlong(relativeScores, indices, axis: -1)
        let scores = which(distanceGrid .>= extent, 0, band)
        var invalid = distanceGrid .< 0
        if let window {
            invalid = invalid .|| (distanceGrid .>= window)
        }
        let masked = which(invalid, Float(-1e30), scores).asType(relative.dtype)
        return concatenated([masked, sink.transposed(0, 2, 1).expandedDimensions(axis: -1)], axis: -1)
    }
}

extension InklingAttention {
    // mlx-vlm's Inkling banded Metal mask specializes by geometry and dtype,
    // sharing the same pipeline across prompt lengths. Write the sink column
    // in the final buffer to avoid materializing a second full mask.
    private static let bandedMask = MLXFast.metalKernel(
        name: "mlxcat_inkling_banded_mask",
        inputNames: ["relative", "projection", "sink", "keys"],
        outputNames: ["mask"],
        source: """
            const int L = relative_shape[1];
            const int H = relative_shape[2];
            const int S = keys_shape[0];
            const int key = thread_position_in_grid.x;
            const int query = thread_position_in_grid.y;
            const int bh = thread_position_in_grid.z;
            if (key > S || query >= L || bh >= relative_shape[0] * H) return;

            const int batch = bh / H;
            const int head = bh % H;
            const size_t output = ((size_t(bh) * L + query) * (S + 1)) + key;
            if (key == S) {
                mask[output] = sink[batch * sink_strides[0] + query * sink_strides[1]
                                    + head * sink_strides[2]];
                return;
            }

            const int distance = query + S - L - key;
            if (distance < 0 || (Window > 0 && distance >= Window)) {
                mask[output] = T(-1e30f);
                return;
            }
            float score = 0.0f;
            if (distance < Extent) {
                const size_t base = batch * relative_strides[0] + query * relative_strides[1]
                                  + head * relative_strides[2];
                for (int d = 0; d < Width; ++d) {
                    score += float(relative[base + d * relative_strides[3]])
                           * float(projection[d * projection_strides[0] + distance * projection_strides[1]]);
                }
            }
            mask[output] = T(score);
            """,
        ensureRowContiguous: false)

    static func metalMask(
        relative: MLXArray, projection: MLXArray, sink: MLXArray, keyCount: Int, window: Int?
    ) -> MLXArray {
        let batch = relative.dim(0)
        let length = relative.dim(1)
        let heads = relative.dim(2)
        let tile = 8
        let keyGrid = (keyCount + tile) / tile * tile
        let queryGrid = (length + tile - 1) / tile * tile
        let keys = MLXArray.zeros([keyCount], dtype: .bool)
        return bandedMask(
            [relative, projection, sink, keys],
            template: [
                ("T", relative.dtype), ("Width", relative.dim(-1)),
                ("Extent", projection.dim(-1)), ("Window", window ?? 0),
            ],
            grid: (keyGrid, queryGrid, batch * heads), threadGroup: (tile, tile, 1),
            outputShapes: [[batch, heads, length, keyCount + 1]],
            outputDTypes: [relative.dtype])[0]
    }
}
