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

    private enum ProbeError: Error { case missingLogits }
}
