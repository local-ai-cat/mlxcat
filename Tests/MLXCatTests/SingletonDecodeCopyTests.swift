import MLX
import MLXCat
import MLXLMCommon
import MLXNN
import XCTest

final class SingletonDecodeCopyTests: XCTestCase {
    func testSnapshotExcludesTheLaunchedButUnemittedStep() throws {
        let model = CopyCacheModel()
        let generator = try makeGenerator(model: model)
        XCTAssertEqual(generator.next().map(\.token), [0])
        XCTAssertEqual(model.offsets, [1, 2])
        XCTAssertEqual(generator.extractCache(uid: "row")?.first?.offset, 1)

        XCTAssertEqual(generator.next().map(\.token), [1])
        XCTAssertEqual(model.offsets, [1, 2, 3])
        XCTAssertEqual(generator.extractCache(uid: "row")?.first?.offset, 2)

        XCTAssertEqual(generator.next().map(\.token), [2])
        XCTAssertEqual(generator.extractCache(uid: "row")?.first?.offset, 3)
    }

    func testDiscardedBuildDoesNotAdvanceTheCachePastEOS() throws {
        let model = CopyCacheModel()
        let generator = try makeGenerator(model: model, eosTokenIds: [1])
        XCTAssertEqual(generator.next().map(\.token), [0])
        XCTAssertEqual(generator.next().map(\.token), [1])
        XCTAssertEqual(model.offsets, [1, 2, 3])
        XCTAssertEqual(generator.extractCache(uid: "row")?.first?.offset, 2)
    }

    private func makeGenerator(
        model: CopyCacheModel, eosTokenIds: Set<Int> = []
    ) throws -> ContinuousBatchGenerator {
        let generator = ContinuousBatchGenerator(model: model, parameters: GenerateParameters())
        try generator.insert(
            uid: "row", cache: model.newCache(parameters: nil), lastToken: MLXArray(Int32(7)),
            sampling: SamplingParameters(temperature: 0, eosTokenIds: eosTokenIds))
        return generator
    }
}

private final class CopyCache: CopyPrebuildKVCache {
    var offset = 0
    let maxSize: Int? = nil
    var state: [MLXArray] = []
    var metaState: [String] = []
    let isTrimmable = false
    var supportsCopyPrebuild: Bool { true }

    func innerState() -> [MLXArray] { state }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        offset += keys.dim(2)
        return (keys, values)
    }

    func trim(_ n: Int) -> Int { 0 }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        let result = CopyCache()
        result.offset = offset
        return result
    }
}

private final class CopyCacheModel: Module, LanguageModel {
    private(set) var offsets: [Int] = []

    func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?) -> LMOutput {
        let layer = cache![0] as! CopyCache
        layer.offset += 1
        offsets.append(layer.offset)
        let scores = (0..<8).map { $0 == layer.offset - 1 ? Float(1) : Float(0) }
        return LMOutput(logits: MLXArray(scores).reshaped(1, 1, 8))
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        [CopyCache()]
    }
}
