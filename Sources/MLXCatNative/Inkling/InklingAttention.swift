import MLX

enum InklingAttention {
    /// The mask is query-dependent. Its last column is the sink score,
    /// corresponding to a zero key and a zero value.
    static func mask(
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
