import MLX
@testable import MLXCat
import XCTest

final class BatchSamplingTests: XCTestCase {
    func testBatchSamplingMatchesScalarGreedyWithPenaltiesAndLogitBias() throws {
        try MLXMetalRuntime.requireAvailable()

        let parameters = SamplingParameters(
            temperature: 0,
            repetitionPenalty: 1.2,
            presencePenalty: 0.4,
            frequencyPenalty: 0.3,
            logitBias: [2: 1.5],
            eosTokenIds: [5]
        )
        let logits = MLXArray([
            Float(0), 4, 5, 1, -1, 3,
            Float(2), 1, 0, 6, 4, 3,
            Float(1), 3, 2, 0, 5, 4,
            Float(5), 0, 1, 2, 3, 4,
        ]).reshaped([4, 6])
        let histories = [
            [2, 2, 5],
            [3, 4],
            [4, 4, 4],
            [0, 5],
        ]

        let batch = try XCTUnwrap(
            TokenSampler.sampleBatch(
                logits: logits,
                parameters: Array(repeating: parameters, count: histories.count),
                generatedTokenHistories: histories,
                randomStates: Array(repeating: nil, count: histories.count)
            )
        ).asArray(Int.self)
        let scalar = histories.indices.map { row in
            TokenSampler.sample(
                logits: logits[row, 0...],
                parameters: parameters,
                generatedTokens: histories[row]
            ).item(Int.self)
        }

        XCTAssertEqual(batch, scalar)
    }

    func testBatchSamplingAppliesTopKForTemperatureSampling() throws {
        try MLXMetalRuntime.requireAvailable()

        let parameters = SamplingParameters(temperature: 1, topK: 1)
        let logits = MLXArray([
            Float(0), 4, 5, 1, -1, 3,
            Float(2), 1, 0, 6, 4, 3,
            Float(1), 3, 2, 0, 5, 4,
            Float(5), 0, 1, 2, 3, 4,
        ]).reshaped([4, 6])

        let batch = try XCTUnwrap(
            TokenSampler.sampleBatch(
                logits: logits,
                parameters: Array(repeating: parameters, count: 4),
                generatedTokenHistories: Array(repeating: [], count: 4),
                randomStates: Array(repeating: nil, count: 4)
            )
        ).asArray(Int.self)

        XCTAssertEqual(batch, [2, 3, 4, 0])
    }

    func testBatchSamplingFallsBackForSeededTemperatureRows() {
        let sampled = TokenSampler.sampleBatch(
            logits: MLXArray.zeros([2, 4]),
            parameters: [
                SamplingParameters(temperature: 1, seed: 1),
                SamplingParameters(temperature: 1, seed: 2),
            ],
            generatedTokenHistories: [[], []],
            randomStates: [
                MLXRandom.RandomState(seed: 1),
                MLXRandom.RandomState(seed: 2),
            ]
        )

        XCTAssertNil(sampled)
    }
}
