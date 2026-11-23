import Foundation
import MLX
import MLXNN

struct InklingVision {
    struct Linear {
        let weight: MLXArray
        let bias: MLXArray?

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let output = matmul(x, weight.transposed())
            return bias.map { output + $0 } ?? output
        }
    }

    struct Norm {
        let weight: MLXArray
        let bias: MLXArray
        let epsilon: Float

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.layerNorm(x, weight: weight, bias: bias, eps: epsilon)
        }
    }

    struct Layer {
        let before: Norm
        let after: Norm
        let query: Linear
        let key: Linear
        let value: Linear
        let output: Linear
        let expand: Linear
        let project: Linear
        let attentionScale: MLXArray
        let mlpScale: MLXArray

        func callAsFunction(_ x: MLXArray, heads: Int, cosine: MLXArray, sine: MLXArray) -> MLXArray {
            let normalized = before(x)
            let count = x.dim(0)
            let headWidth = x.dim(1) / heads
            func rotate(_ projected: MLXArray) -> MLXArray {
                let input = projected.reshaped(1, count, heads, headWidth).asType(.float32)
                let pairs = input.reshaped(1, count, heads, headWidth / 2, 2)
                let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1)
                    .reshaped(input.shape)
                return (input * cosine + rotated * sine).asType(projected.dtype)
                    .transposed(0, 2, 1, 3)
            }
            let attended = MLXFast.scaledDotProductAttention(
                queries: rotate(query(normalized)), keys: rotate(key(normalized)),
                values: value(normalized).reshaped(1, count, heads, headWidth).transposed(0, 2, 1, 3),
                scale: 1 / Float(headWidth).squareRoot(), mask: .none)
            let residual = x + attentionScale * output(attended.transposed(0, 2, 1, 3).reshaped(x.shape))
            return residual + mlpScale * project(gelu(expand(after(residual))))
        }
    }

    let config: InklingConfiguration.Vision
    let patches: Linear
    let position: MLXArray
    let classEmbedding: MLXArray
    let classPosition: MLXArray
    let preNorm: Norm
    let layers: [Layer]
    let adapterIn: Linear
    let adapterOut: Linear

    func callAsFunction(_ pixels: MLXArray, rows: Int, columns: Int) -> MLXArray {
        let merge = config.merge
        let patch = config.patchSize
        let count = rows * columns
        let patchCount = count * merge * merge
        let width = config.width
        let unpacked = pixels.asType(preNorm.weight.dtype)
            .reshaped(count, merge, patch, merge, patch, 3)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(patchCount, patch * patch * 3)
        let encoded = patches(unpacked)
        let positions = interpolatedPosition(rows: rows * merge, columns: columns * merge)
            .reshaped(rows, merge, columns, merge, width)
            .transposed(0, 2, 1, 3, 4).reshaped(patchCount, width)
        let cls = (classEmbedding + classPosition).asType(encoded.dtype).expandedDimensions(axis: 0)
        var state = preNorm(concatenated([cls, encoded + positions.asType(encoded.dtype)], axis: 0))
        let angles = spatialAngles(rows: rows, columns: columns)
        let cosine = cos(angles)
        let sine = sin(angles)
        for layer in layers {
            state = layer(state, heads: config.heads, cosine: cosine, sine: sine)
        }
        return adapterOut(gelu(adapterIn(state[1..., 0...].reshaped(count, merge * merge * width))))
    }

    private func interpolatedPosition(rows: Int, columns: Int) -> MLXArray {
        let grid = config.gridSize
        let values = position.asType(.float32).reshaped(grid, grid, config.width)
        func interpolate(_ input: MLXArray, count: Int, axis: Int) -> MLXArray {
            let coordinates = (arange(count).asType(.float32) + 0.5) * (Float(grid) / Float(count)) - 0.5
            let lower = floor(coordinates)
            let first = input.take(clip(lower, min: 0, max: grid - 1).asType(.int32), axis: axis)
            let second = input.take(clip(lower + 1, min: 0, max: grid - 1).asType(.int32), axis: axis)
            let shape = axis == 0 ? [count, 1, 1] : [1, count, 1]
            let fraction = (coordinates - lower).reshaped(shape)
            return first + (second - first) * fraction
        }
        return interpolate(interpolate(values, count: rows, axis: 0), count: columns, axis: 1)
    }

    private func spatialAngles(rows: Int, columns: Int) -> MLXArray {
        let merge = config.merge
        var coordinates: [Float] = [0, 0]
        for row in 0..<rows {
            for column in 0..<columns {
                for dy in 0..<merge {
                    for dx in 0..<merge {
                        coordinates.append(Float(column * merge + dx + 1))
                        coordinates.append(Float(row * merge + dy + 1))
                    }
                }
            }
        }
        let spatialWidth = config.width / config.heads / 2
        let inverse = (0..<(spatialWidth / 2)).map { offset in
            pow(Float(10_000), -Float(2 * offset) / Float(spatialWidth))
        }
        let products = MLXArray(coordinates).reshaped(-1, 2, 1) * MLXArray(inverse)
        let angles = products.reshaped(coordinates.count / 2, spatialWidth)
        return stacked([angles, angles], axis: -1)
            .reshaped(1, coordinates.count / 2, 1, spatialWidth * 2)
    }
}
