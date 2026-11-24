import Foundation
import MLX
import MLXLMCommon

/// Developer fixture adapter for the same model and processor used by the native engine.
package struct InklingReferenceRuntime {
    private let directory: URL
    private let model: InklingLanguageModel
    private let processor: InklingInputProcessor
    package let endToken: Int

    package init(directory: URL, tokenizer: any MLXLMCommon.Tokenizer) throws {
        self.directory = directory
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
    ) throws -> (maximumLogitError: Float, differentTopTokens: Int, differentLogitRows: Int, cacheExact: Bool) {
        let cache = try model.newCache(parameters: nil)
        _ = try model.prepare(input, cache: cache, state: nil, prefill: .init(stepSize: chunk))
        eval(cache)

        let panelCache = cache.map { $0.copy() }
        let panel = model(MLXArray(tokens), cache: panelCache)
        var maximum: Float = 0
        var differentTokens = 0
        var differentRows = 0
        for (index, token) in tokens.enumerated() {
            let expected = model(MLXArray([token]), cache: cache)[0, 0]
            let actual = panel[0, index]
            maximum = max(maximum, abs(expected - actual).max().item(Float.self))
            if !verificationBitsEqual(expected, actual).item(Bool.self) { differentRows += 1 }
            if argMax(expected).item(Int.self) != argMax(actual).item(Int.self) {
                differentTokens += 1
            }
        }
        var comparisons: [MLXArray] = []
        var matchingMetadata = true
        for (panelVisit, serialVisit) in zip(panelCache, cache) {
            guard panelVisit.metaState == serialVisit.metaState,
                  panelVisit.state.count == serialVisit.state.count
            else { matchingMetadata = false; break }
            comparisons += zip(panelVisit.state, serialVisit.state).map(verificationBitsEqual)
        }
        let matchingCache = matchingMetadata && stacked(comparisons).all().item(Bool.self)
        return (maximum, differentTokens, differentRows, matchingCache)
    }

    private func verificationBitsEqual(_ lhs: MLXArray, _ rhs: MLXArray) -> MLXArray {
        guard lhs.dtype == rhs.dtype, lhs.shape == rhs.shape else { return MLXArray(false) }
        let bits: DType = lhs.dtype == .float32 ? .uint32 : .uint16
        return arrayEqual(lhs.view(dtype: bits), rhs.view(dtype: bits))
    }

    /// Decode greedy tokens with the shipped target's ordinary singleton arithmetic.
    /// A positive draft depth probes DSpark without changing serving policy.
    package func greedyDecode(
        in input: LMInput, chunk: Int, steps: Int, draftDepth: Int
    ) throws -> (tokens: [Int], seconds: Double, accepted: Int, proposed: Int) {
        var cache = try model.newCache(parameters: nil)
        var drafter = draftDepth == 0 ? nil : try InklingDraftProbe(directory: directory, model: model.textModel)
        let captureIDs = drafter?.config.targetLayerIds ?? []
        if let drafter, !(1..<drafter.config.blockSize).contains(draftDepth) {
            throw InklingDraftProbe.DraftError.unsupportedHead
        }
        var prefixFeatures: [Int: [MLXArray]] = [:]
        let prepared = try model.prepare(input, cache: cache, prefill: .init(stepSize: chunk),
            capture: drafter == nil ? nil : { index, array in
                if captureIDs.contains(index) {
                    eval(array)
                    prefixFeatures[index, default: []].append(array)
                }
            })
        guard case .logits(let initial) = prepared else { throw ProbeError.missingLogits }
        var anchor = argMax(initial.logits[0, -1]).item(Int.self)
        if drafter != nil {
            drafter?.observe(captureIDs.map { concatenated(prefixFeatures[$0]!, axis: 1) })
            prefixFeatures.removeAll()
        }
        var result: [Int] = []
        var accepted = 0
        var proposed = 0
        let start = DispatchTime.now().uptimeNanoseconds
        while result.count < steps {
            result.append(anchor)
            if anchor == endToken || result.count == steps { break }
            guard let activeDraft = drafter else {
                anchor = argMax(model(MLXArray([anchor]), cache: cache)[0, 0]).item(Int.self)
                continue
            }

            let depth = min(draftDepth, steps - result.count)
            let guesses = activeDraft.propose(anchor: anchor, count: depth, model: model.textModel)
            proposed += guesses.count
            // Evaluate only known token inputs. Each snapshot is from exactly
            // one singleton advance of the prior row's attention and conv state.
            let fork = cache.map { $0.copy() }
            var states: [[any KVCache]] = []
            var featureRows: [[MLXArray]] = []
            var selectors: [MLXArray] = []
            for token in [anchor] + guesses {
                var captures: [Int: MLXArray] = [:]
                let embeddings = model.textModel.embed(MLXArray([token]).reshaped(1, 1))
                let logits = model.forward(embeddings, cache: fork, capture: { index, array in
                    if captureIDs.contains(index) { captures[index] = array }
                })
                selectors.append(argMax(logits[0, 0]))
                featureRows.append(captureIDs.map { captures[$0]! })
                states.append(fork.map { $0.copy() })
            }
            let targetTokens = stacked(selectors).asArray(Int.self)
            let decision = InklingDraftDecision(
                proposals: guesses, target: targetTokens, remaining: steps - result.count, endToken: endToken)
            result.append(contentsOf: decision.accepted)
            accepted += decision.accepted.count
            if decision.isFinished { break }
            cache = states[decision.rowsToKeep - 1]
            anchor = decision.next
            drafter?.observe(captureIDs.indices.map { slot in
                concatenated(featureRows.prefix(decision.rowsToKeep).map { $0[slot] }, axis: 1)
            })
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        return (result, seconds, accepted, proposed)
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
                    model.textModel.embed(token.reshaped(1, 1)), cache: cache, observe: { stage, array in
                    eval(array)
                    let now = DispatchTime.now().uptimeNanoseconds
                    stageElapsed[stage, default: 0] += now - previous
                    previous = now
                })
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
