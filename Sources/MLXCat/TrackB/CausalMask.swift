import MLX

public enum CausalMask {
    public static func create(
        n: Int,
        offset: Int,
        leftPadding: MLXArray? = nil,
        windowSize: Int? = nil
    ) -> MLXArray? {
        if n == 1, windowSize == nil, leftPaddingMax(leftPadding) == 0 {
            return nil
        }

        var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
        var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
        linds = linds[0..., .newAxis]
        rinds = rinds[.newAxis]

        var mask = linds .>= rinds
        if let windowSize {
            mask = mask & (linds .< rinds + windowSize)
        }
        if let leftPadding {
            mask = expandedDimensions(mask, axes: [0, 1])
            mask = mask & (expandedDimensions(leftPadding, axes: [1, 2, 3]) .<= rinds)
        }
        return mask
    }

    private static func leftPaddingMax(_ leftPadding: MLXArray?) -> Int {
        guard let leftPadding else { return 0 }
        return leftPadding.max().item(Int.self)
    }
}
