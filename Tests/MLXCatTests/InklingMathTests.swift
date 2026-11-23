import MLX
import XCTest

@testable import MLXCatNative

final class InklingMathTests: XCTestCase {
    func testRouterBiasChangesSelectionWithoutChangingRouteWeights() throws {
        try MLXMetalRuntime.requireAvailable()
        let input = MLXArray([Float(1)]).reshaped(1, 1)
        let weight = MLXArray([Float(2), 0, -2]).reshaped(3, 1)
        let bias = MLXArray([Float(0), 0, 1])
        let result = InklingMath.route(input, weight: weight, bias: bias, count: 2, scale: 1)
        let indices = result.indices.asArray(Int.self)
        let weights = result.weights.asArray(Float.self)
        let byExpert = Dictionary(uniqueKeysWithValues: zip(indices, weights))
        XCTAssertEqual(Set(indices), [0, 2])
        XCTAssertEqual(byExpert[0] ?? 0, 0.880797, accuracy: 1e-5)
        XCTAssertEqual(byExpert[2] ?? 0, 0.119203, accuracy: 1e-5)
    }

    func testZeroMessageQueriesAverageNormalizedCoreBlocks() throws {
        try MLXMetalRuntime.requireAvailable()
        let left = MLXArray([Float(3), 3]).reshaped(1, 1, 2)
        let right = MLXArray([Float(6), -6]).reshaped(1, 1, 2)
        let bank = InklingMath.messageBank([left, right])
        let messages = InklingMath.messages(query: MLXArray.zeros([4, 2]), bank: bank)
        XCTAssertEqual(messages.shape, [1, 1, 4, 2])
        for (index, value) in messages.asArray(Float.self).enumerated() {
            XCTAssertEqual(value, index.isMultiple(of: 2) ? 1 : 0, accuracy: 1e-5)
        }
    }
}
