import MLX
@testable import MLXCat
import XCTest

/// A tiny `top_p` must still leave the most likely token. Ported from
/// ml-explore/mlx-lm#1912's `test_apply_top_p_keeps_top_token_at_tiny_thresholds`.
final class TopPSamplingTests: XCTestCase {
    private let tinyTopPs: [Float] = [1e-8, 1e-6, 1e-4, 1e-3]

    func testTopPKeepsOnlyTheMostLikelyTokenAtTinyThresholds() throws {
        try MLXMetalRuntime.requireAvailable()

        let logprobs = log(MLXArray([Float(0.9), 0, 0, 0.1])).reshaped([1, 4])
        for topP in tinyTopPs {
            let filtered = TokenSampler.applyTopP(logprobs, topP: topP)
            let probs = softmax(filtered, axis: -1).asArray(Float.self)
            XCTAssertEqual(probs, [1, 0, 0, 0], "top_p \(topP)")
        }
    }

    func testTopPNeverMasksEveryTokenInHalfPrecision() throws {
        try MLXMetalRuntime.requireAvailable()

        let vocabularySize = 4096
        let logitScale: Float = 3
        let key = MLXRandom.key(0)
        let logits = MLXRandom.normal([2, vocabularySize], key: key) * logitScale
        let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)

        for dtype in [DType.float32, .float16, .bfloat16] {
            let typed = logprobs.asType(dtype)
            let rowMaxima = typed.max(axis: -1).asType(.float32).asArray(Float.self)
            for topP in tinyTopPs {
                let filtered = TokenSampler.applyTopP(typed, topP: topP)
                let kept = (filtered .> MLXArray(-Float.infinity)).sum(axis: -1).asArray(Int32.self)
                XCTAssertTrue(kept.allSatisfy { $0 > 0 }, "\(dtype) top_p \(topP) kept \(kept)")
                XCTAssertEqual(
                    filtered.max(axis: -1).asType(.float32).asArray(Float.self),
                    rowMaxima,
                    "\(dtype) top_p \(topP)"
                )
            }
        }
    }

    func testTopPStillKeepsTheNucleusAtOrdinaryThresholds() throws {
        try MLXMetalRuntime.requireAvailable()

        let logprobs = log(MLXArray([Float(0.5), 0.3, 0.15, 0.05])).reshaped([1, 4])
        let filtered = TokenSampler.applyTopP(logprobs, topP: 0.7)
        let kept = (filtered .> MLXArray(-Float.infinity)).asArray(Bool.self)
        XCTAssertEqual(kept, [true, true, false, false])
    }
}
