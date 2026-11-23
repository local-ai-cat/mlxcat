import CoreImage
import MLX
import MLXLMCommon

enum InklingImageProcessor {
    enum InputError: Error {
        case invalidImageExtent
        case imageCountMismatch
    }

    static func prepare(_ source: UserInput.Image, patch: Int) throws -> (MLXArray, THW) {
        let image = try source.asCIImage()
        let extent = image.extent.integral
        guard extent.width.isFinite, extent.height.isFinite,
            extent.width > 0, extent.height > 0
        else { throw InputError.invalidImageExtent }
        let width = Int(extent.width)
        let height = Int(extent.height)
        let rows = (height + patch - 1) / patch
        let columns = (width + patch - 1) / patch
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        let context = CIContext()
        rgba.withUnsafeMutableBytes {
            context.render(image, toBitmap: $0.baseAddress!, rowBytes: width * 4,
                           bounds: extent, format: .RGBA8, colorSpace: colorSpace)
        }

        let paddedWidth = columns * patch
        let scale: Float = 2 / 255
        var values = [Float](repeating: -1 - scale, count: rows * patch * paddedWidth * 3)
        for y in 0..<height {
            for x in 0..<width {
                let source = (y * width + x) * 4
                let target = (y * paddedWidth + x) * 3
                let background = (y / 8 + x / 8).isMultiple(of: 2) ? 204 : 153
                let alpha = Int(rgba[source + 3])
                for channel in 0..<3 {
                    let pixel = Int(rgba[source + channel])
                    let composited = min(255, pixel + (background * (255 - alpha) + 127) / 255)
                    values[target + channel] = Float(composited) * scale - 1
                }
            }
        }
        let tiles = MLXArray(values).reshaped(rows, patch, columns, patch, 3)
            .transposed(0, 2, 1, 3, 4).reshaped(rows * columns, patch, patch, 3)
        return (tiles, THW(1, rows, columns))
    }
}
