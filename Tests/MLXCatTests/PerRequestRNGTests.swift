import MLX
import MLXLMCommon
import MLXNN
import MLXCat
import XCTest

final class PerRequestRNGTests: XCTestCase {
    func testMixedSeededBatchUsesIndependentPerRequestRNGState() throws {
        let steps = 4
        let seedA = SamplingParameters(temperature: 1, seed: 11)
        let seedB = SamplingParameters(temperature: 1, seed: 29)
        let unseeded = SamplingParameters(temperature: 1)

        let mixed = try runBatch(
            rows: [
                ("seed-a", seedA),
                ("seed-b", seedB),
                ("unseeded", unseeded),
            ],
            globalSeed: 7,
            steps: steps
        )
        let repeatedMixed = try runBatch(
            rows: [
                ("seed-a", seedA),
                ("seed-b", seedB),
                ("unseeded", unseeded),
            ],
            globalSeed: 101,
            steps: steps
        )
        let seedAOnly = try runBatch(rows: [("seed-a", seedA)], globalSeed: 7, steps: steps)
        let seedBOnly = try runBatch(rows: [("seed-b", seedB)], globalSeed: 7, steps: steps)
        let unseededOnly = try runBatch(rows: [("unseeded", unseeded)], globalSeed: 7, steps: steps)

        XCTAssertEqual(mixed["seed-a"], seedAOnly["seed-a"])
        XCTAssertEqual(mixed["seed-b"], seedBOnly["seed-b"])
        XCTAssertEqual(mixed["seed-a"], repeatedMixed["seed-a"])
        XCTAssertEqual(mixed["seed-b"], repeatedMixed["seed-b"])
        XCTAssertEqual(mixed["unseeded"], unseededOnly["unseeded"])
    }

    private func runBatch(
        rows: [(uid: String, sampling: SamplingParameters)],
        globalSeed: UInt64,
        steps: Int
    ) throws -> [String: [Int]] {
        MLXRandom.seed(globalSeed)

        let model = FixedLogitLanguageModel(vocabularySize: 257)
        let generator = ContinuousBatchGenerator(model: model, parameters: GenerateParameters())
        var tokensByUID = Dictionary(uniqueKeysWithValues: rows.map { ($0.uid, [Int]()) })

        for row in rows {
            try generator.insert(
                uid: row.uid,
                cache: [Self.makeCache()],
                lastToken: MLXArray(Int32(0)),
                sampling: row.sampling
            )
        }

        for _ in 0 ..< steps {
            for response in generator.next() {
                tokensByUID[response.uid, default: []].append(response.token)
            }
        }

        return tokensByUID
    }

    private static func makeCache() -> KVCacheSimple {
        let cache = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, 1, 1])
        let values = MLXArray.zeros([1, 1, 1, 1])
        cache.state = [keys, values]
        return cache
    }
}

private final class FixedLogitLanguageModel: Module, LanguageModel {
    private let vocabularySize: Int

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        LMOutput(logits: MLXArray.zeros([input.tokens.dim(0), 1, vocabularySize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }
}
