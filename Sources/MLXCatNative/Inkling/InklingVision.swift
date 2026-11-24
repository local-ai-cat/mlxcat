import Foundation
import MLX
import MLXNN

struct InklingVision {
    struct Linear {
        let weight: InklingProjection
        let bias: MLXArray?

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            weight(x, bias: bias)
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
        if rows == grid, columns == grid { return values }
        func taps(_ count: Int) -> (indices: MLXArray, weights: MLXArray) {
            let coordinates = maximum(
                (arange(count).asType(.float32) + 0.5) * (Float(grid) / Float(count)) - 0.5, 0)
            let lower = floor(coordinates)
            let fraction = coordinates - lower
            let indices = clip(stacked([lower, lower + 1], axis: -1), min: 0, max: grid - 1)
                .asType(.int32)
            return (indices, stacked([1 - fraction, fraction], axis: -1))
        }
        let vertical = taps(rows)
        let horizontal = taps(columns)
        if Device.defaultDevice().deviceType == .gpu {
            let rowWidth = columns * config.width
            return Self.positionResize(
                [values, vertical.indices, vertical.weights, horizontal.indices, horizontal.weights],
                grid: (rowWidth, rows, 1), threadGroup: (min(rowWidth, 256), 1, 1),
                outputShapes: [[rows, columns, config.width]], outputDTypes: [.float32])[0]
        }
        let alongColumns = (
            values.take(horizontal.indices, axis: 1)
                * horizontal.weights.reshaped(1, columns, 2, 1)
        ).sum(axis: 2)
        return (
            alongColumns.take(vertical.indices, axis: 0)
                * vertical.weights.reshaped(rows, 2, 1, 1)
        ).sum(axis: 1)
    }

    // Contract the horizontal taps inside the vertical loop, as in
    // mlx-vlm's separable_interpolate Metal kernel.
    private static let positionResize = MLXFast.metalKernel(
        name: "mlxcat_inkling_position_resize",
        inputNames: ["x", "iy", "wy", "ix", "wx"], outputNames: ["out"],
        source: """
            const uint gx = thread_position_in_grid.x;
            const uint y = thread_position_in_grid.y;
            const int inWidth = x_shape[1], channels = x_shape[2];
            const int outWidth = ix_shape[0], outHeight = iy_shape[0];
            if (gx >= uint(outWidth * channels) || y >= uint(outHeight)) return;
            const size_t column = (gx / channels) * 2;
            const size_t channel = gx % channels;
            float result = 0.0f;
            for (int dy = 0; dy < 2; ++dy) {
                const size_t row = size_t(iy[y * 2 + dy]) * inWidth * channels + channel;
                float horizontal = 0.0f;
                for (int dx = 0; dx < 2; ++dx) {
                    horizontal += x[row + size_t(ix[column + dx]) * channels] * wx[column + dx];
                }
                result += horizontal * wy[y * 2 + dy];
            }
            out[size_t(y) * outWidth * channels + gx] = result;
            """)

    // One ULP in the frequencies can move a rotated key across a BF16
    // rounding boundary. Use the precise Metal function to match mlx-vlm.
    private static let rotaryFrequencies = MLXFast.metalKernel(
        name: "mlxcat_inkling_rotary_frequencies",
        inputNames: ["exponents"], outputNames: ["frequencies"],
        source: """
            const uint i = thread_position_in_grid.x;
            frequencies[i] = metal::precise::pow(10000.0f, exponents[i]);
            """)

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
        let exponents = -arange(0, spatialWidth, step: 2).asType(.float32) / Float(spatialWidth)
        let inverse: MLXArray
        if Device.defaultDevice().deviceType == .gpu {
            inverse = Self.rotaryFrequencies(
                [exponents], grid: (exponents.size, 1, 1),
                threadGroup: (min(exponents.size, 64), 1, 1),
                outputShapes: [exponents.shape], outputDTypes: [.float32])[0]
        } else {
            inverse = pow(MLXArray(Float(10_000)), exponents)
        }
        let products = MLXArray(coordinates).reshaped(-1, 2, 1) * inverse
        let angles = products.reshaped(coordinates.count / 2, spatialWidth)
        return stacked([angles, angles], axis: -1)
            .reshaped(1, coordinates.count / 2, 1, spatialWidth * 2)
    }
}
