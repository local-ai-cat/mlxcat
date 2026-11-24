import Foundation
import MLX
import MLXLMCommon

/// Developer fixture adapter for the same model and processor used by the native engine.
package struct InklingReferenceRuntime {
    private let model: InklingLanguageModel
    private let processor: InklingInputProcessor
    package let endToken: Int

    package init(directory: URL, tokenizer: any MLXLMCommon.Tokenizer) throws {
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(InklingConfiguration.self, from: configData)
        let weights = try InklingCheckpoint(directory: directory)
        let built = try weights.makeModel(config: config, tokenizer: tokenizer)
        model = built.model
        processor = built.processor
        endToken = config.eosTokenID ?? 200006
    }

    package func prepare(_ input: UserInput) throws -> LMInput {
        try processor.prepare(input: input)
    }

    package func compare(
        _ expectedTokens: [Int], in input: LMInput, chunk: Int, topTokenIDs: [[Int]]
    ) throws -> (tokens: [Int], scores: [[Float]]) {
        let cache = try model.newCache(parameters: nil)
        let prepared = try model.prepare(
            input, cache: cache, state: nil, prefill: .init(stepSize: chunk))
        guard case .logits(let initial) = prepared else { throw ProbeError.missingLogits }

        var logits = initial.logits
        var tokens: [Int] = []
        var scores: [[Float]] = []
        for (step, expected) in expectedTokens.enumerated() {
            let row = logits[0, -1, 0...]
            if step < topTokenIDs.count {
                scores.append(topTokenIDs[step].map { row[$0].item(Float.self) })
            }
            let token = argMax(row).item(Int.self)
            tokens.append(token)
            if token != expected { break }
            logits = model(MLXArray([token]), cache: cache)
        }
        return (tokens, scores)
    }

    /// A speculative verifier must reproduce the logits of the ordinary
    /// one-token path for each row, including at a rotating cache boundary.
    package func compareVerificationPanel(
        _ tokens: [Int], in input: LMInput, chunk: Int
    ) throws -> (maximumLogitError: Float, differentTopTokens: Int) {
        let cache = try model.newCache(parameters: nil)
        _ = try model.prepare(input, cache: cache, state: nil, prefill: .init(stepSize: chunk))
        eval(cache)
        let panel = model(MLXArray(tokens), cache: cache.map { $0.copy() })
        var maximum: Float = 0
        var differentTokens = 0
        for (index, token) in tokens.enumerated() {
            let expected = model(MLXArray([token]), cache: cache)[0, 0]
            let actual = panel[0, index]
            maximum = max(maximum, abs(expected - actual).max().item(Float.self))
            if argMax(expected).item(Int.self) != argMax(actual).item(Int.self) {
                differentTokens += 1
            }
        }
        return (maximum, differentTokens)
    }

    package func profileDecode(
        in input: LMInput, chunk: Int, steps: Int
    ) throws -> (tokens: [Int], buildMs: Double, submitMs: Double, readMs: Double) {
        let cache = try model.newCache(parameters: nil)
        let prepared = try model.prepare(input, cache: cache, state: nil, prefill: .init(stepSize: chunk))
        guard case .logits(let initial) = prepared else { throw ProbeError.missingLogits }
        func sample(_ logits: MLXArray) -> MLXArray { argMax(logits[0, -1, 0...]) }
        var token = sample(initial.logits)
        for _ in 0..<16 {
            token = sample(model(token.reshaped(1), cache: cache))
            eval(token)
        }

        print("Inkling decode probe start \(DispatchTime.now().uptimeNanoseconds)")
        var tokens: [Int] = []
        var build: UInt64 = 0
        var submit: UInt64 = 0
        var read: UInt64 = 0
        let measureStages = ProcessInfo.processInfo.environment["MLXCAT_INKLING_PROFILE_STAGES"] == "1"
        var stageElapsed: [String: UInt64] = [:]
        for _ in 0..<steps {
            let start = DispatchTime.now().uptimeNanoseconds
            let logits = autoreleasepool {
                guard measureStages else { return model(token.reshaped(1), cache: cache) }
                var previous = start
                let result = model.forward(
                    model.textModel.embed(token.reshaped(1, 1)), cache: cache
                ) { stage, array in
                    eval(array)
                    let now = DispatchTime.now().uptimeNanoseconds
                    stageElapsed[stage, default: 0] += now - previous
                    previous = now
                }
                eval(result)
                stageElapsed["final residual and vocabulary", default: 0] +=
                    DispatchTime.now().uptimeNanoseconds - previous
                return result
            }
            let next = sample(logits)
            let built = DispatchTime.now().uptimeNanoseconds
            asyncEval(next, logits)
            let submitted = DispatchTime.now().uptimeNanoseconds
            tokens.append(next.item(Int.self))
            let finished = DispatchTime.now().uptimeNanoseconds
            build += built - start
            submit += submitted - built
            read += finished - submitted
            token = next
        }
        print("Inkling decode probe end \(DispatchTime.now().uptimeNanoseconds)")
        func milliseconds(_ ticks: UInt64) -> Double { Double(ticks) / Double(steps) / 1_000_000 }
        for stage in stageElapsed.keys.sorted() {
            print("Inkling serialized stage \(stage): \(milliseconds(stageElapsed[stage, default: 0])) ms/token")
        }
        return (tokens, milliseconds(build), milliseconds(submit), milliseconds(read))
    }

    private enum ProbeError: Error { case missingLogits }
}
