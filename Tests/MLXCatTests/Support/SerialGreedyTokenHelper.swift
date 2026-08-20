import MLX
import MLXLMCommon

enum SerialGreedyTokenHelper {
    static func tokens(
        model: any LanguageModel,
        input: LMInput,
        parameters: GenerateParameters,
        steps: Int
    ) throws -> [Int] {
        let cache = try model.newCache(parameters: parameters)
        var state: LMOutput.State?
        var currentLogits: MLXArray
        var currentToken: MLXArray

        switch try model.prepare(input, cache: cache, state: nil, prefill: parameters.prefill) {
        case .tokens(let tokens):
            let output = model(tokens[text: .newAxis], cache: cache, state: state)
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
        case .logits(let output):
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
        }
        eval(currentToken, currentLogits, cache)

        var tokens: [Int] = []
        for _ in 0 ..< steps {
            eval(currentToken, currentLogits)
            tokens.append(currentToken.item(Int.self))

            let output = model(
                LMInput.Text(tokens: currentToken[.newAxis, 0...]),
                cache: cache,
                state: state
            )
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
            asyncEval(currentToken, currentLogits)
        }

        eval(currentToken, currentLogits)
        return tokens
    }
}
