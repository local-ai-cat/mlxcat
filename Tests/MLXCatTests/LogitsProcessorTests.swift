import MLX
@testable import MLXCat
import XCTest

final class LogitsProcessorTests: XCTestCase {
    func testMinTokensMasksEOSBelowThreshold() throws {
        try MLXMetalRuntime.requireAvailable()

        let logits = MLXArray([0.0, 1.0, 5.0].map(Float.init))
        let parameters = SamplingParameters(
            temperature: 0,
            minTokens: 2,
            eosTokenIds: [2]
        )

        let belowThreshold = TokenSampler.sample(
            logits: logits,
            parameters: parameters,
            generatedTokens: [7]
        )
        XCTAssertEqual(belowThreshold.item(Int.self), 1)

        let atThreshold = TokenSampler.sample(
            logits: logits,
            parameters: parameters,
            generatedTokens: [7, 8]
        )
        XCTAssertEqual(atThreshold.item(Int.self), 2)
    }

    func testLogitBiasShiftsGreedyToken() throws {
        try MLXMetalRuntime.requireAvailable()

        let logits = MLXArray([0.0, 1.0, 3.0].map(Float.init))
        let token = TokenSampler.sample(
            logits: logits,
            parameters: SamplingParameters(
                temperature: 0,
                logitBias: [1: 4]
            )
        )

        XCTAssertEqual(token.item(Int.self), 1)
    }
}
