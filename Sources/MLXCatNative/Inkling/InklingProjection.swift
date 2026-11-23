import Foundation
import MLX

/// Keep a checkpoint's affine matrices packed for both projection and embedding lookup.
enum InklingProjection {
    struct Affine {
        let packed: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let groupSize: Int
        let bits: Int

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            quantizedMM(
                input, packed, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits, mode: .affine)
        }

        func rows(_ indices: MLXArray) -> MLXArray {
            dequantized(
                packed[indices], scales: scales[indices], biases: biases[indices],
                groupSize: groupSize, bits: bits, mode: .affine)
        }
    }

    case dense(MLXArray)
    case vocabulary16(MLXArray)
    private static let usesVocabularyGEMV =
        ProcessInfo.processInfo.environment["MLXCAT_INKLING_VOCAB_GEMV"] == "1"

    static func vocabulary(_ weight: MLXArray) -> Self {
        usesVocabularyGEMV ? .vocabulary16(weight) : .dense(weight.asType(.float32))
    }
    case affine(Affine)

    func callAsFunction(_ input: MLXArray, bias: MLXArray? = nil) -> MLXArray {
        switch self {
        case .dense(let weight):
            if let bias { return addMM(bias, input, weight.transposed()) }
            return matmul(input, weight.transposed())
        case .vocabulary16(let weight):
            if InklingVocabularyGEMV.supports(input, weight: weight), bias == nil {
                return InklingVocabularyGEMV.run(input, weight: weight)
            }
            return Self.dense(weight.asType(.float32))(input, bias: bias)
        case .affine(let weight):
            let projected = weight(input)
            if let bias { return projected + bias }
            return projected
        }
    }

    func rows(_ indices: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight), .vocabulary16(let weight): return weight[indices]
        case .affine(let weight): return weight.rows(indices)
        }
    }
}
