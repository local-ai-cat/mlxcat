import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers
import XCTest

private struct PrefixSchedulerGateResult: Sendable {
    let fetchHits: Int
    let stores: Int
    let checkedTokens: Int
    let mismatches: Int
    let allocatedBaseline: Int
    let allocatedAfterBatch: Int
    let refBaseline: Int
    let refAfterBatch: Int
}

final class PrefixSchedulerIntegrationTests: XCTestCase {
    func testSchedulerUsesPrefixCacheWithMixedHitAndMissBatch() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M5 prefix scheduler gate.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M5 prefix scheduler fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let result = try await container.perform { context in
            try await Self.evaluateGate(context: context)
        }

        if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_PREFIX_GATE"] == "1" {
            print(
                "M5 prefix+scheduler: fetchHits=\(result.fetchHits), stores=\(result.stores), checkedTokens=\(result.checkedTokens), mismatches=\(result.mismatches), allocatedBaseline=\(result.allocatedBaseline), allocatedAfter=\(result.allocatedAfterBatch), refBaseline=\(result.refBaseline), refAfter=\(result.refAfterBatch)"
            )
        }

        XCTAssertGreaterThanOrEqual(result.fetchHits, 2)
        XCTAssertGreaterThanOrEqual(result.stores, 1)
        XCTAssertGreaterThan(result.checkedTokens, 0)
        XCTAssertEqual(result.mismatches, 0)
        XCTAssertEqual(result.allocatedAfterBatch, result.allocatedBaseline)
        XCTAssertEqual(result.refAfterBatch, result.refBaseline)
    }

    func testSessionPrefixCacheReusesExtendedPrompt() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M10a session prefix gate.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M10a session prefix gate is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let stats = try await container.perform { context in
            try await Self.evaluateSessionReuseGate(context: context)
        }

        XCTAssertEqual(stats.fetchHitCount, 1)
        // Publishes are throttled: a maxTokens=2 request finishes on its only
        // decode step, so the sole guaranteed store is the finish publish
        // (request 2's stores are lease-blocked by its own hit). The fetch hit
        // above is the actual reuse contract. This asserted >= 2 when every
        // decode step published, which was a ~14% single-request throughput
        // cost — see Scheduler.midGenerationPublishInterval.
        XCTAssertGreaterThanOrEqual(stats.storeCount, 1)
        XCTAssertEqual(stats.clearCount, 0)
    }

    /// Largest top-1/top-2 logit margin at which a cached and an uncached run may
    /// legitimately pick different tokens. The engine runs a prompt's last token
    /// through the single-token decode kernel, so a session slot's last prompt
    /// position is computed by a different (equally valid) kernel than a cold
    /// prefill computes it. Measured on Qwen3-0.6B-4bit, moving tokens between the
    /// prefill and decode kernels shifts single logits by up to 1.58; two logits
    /// moving in opposite directions can close a gap of twice that. Rounded up.
    private static let kernelPathMarginTolerance: Float = 4.0

    /// Session prefix reuse is exact for every position the warm request
    /// prefilled, and the cached run agrees with a cache-disabled run token for
    /// token until the first near-tie. It is not bit-identical end to end: see
    /// docs/KNOWN-FAILURES.md §4 (2026-10-07).
    func testSessionPrefixCacheMatchesCacheDisabledForExtendedPrompt() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M10a session correctness gate.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M10a session correctness gate is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let result = try await container.perform { context in
            try await Self.evaluateSessionCorrectnessGate(context: context)
        }

        // Reuse: the cached request was served from the warm request's slot.
        XCTAssertEqual(result.cachedFetchHits, 1)
        XCTAssertGreaterThanOrEqual(result.stores, 1)

        // The slot covers the whole warm prompt and its bytes are the prefill's
        // bytes everywhere the warm request prefilled (all but its last token).
        XCTAssertEqual(result.restoredTokenCount, result.warmPromptCount)
        XCTAssertEqual(
            result.prefilledPositionsDiffering, [],
            "restored KV differs from a cold prefill at positions the warm request prefilled")

        // Token agreement until a near-tie. At the first step where the cached,
        // fresh and serial runs disagree, the serial run must be at a near-tie.
        XCTAssertEqual(result.cachedTokens.count, result.freshTokens.count)
        let steps = min(result.cachedTokens.count, result.freshTokens.count, result.serial.count)
        let firstDisagreement = (0 ..< steps).first { step in
            let serialToken = result.serial[step].token
            return result.cachedTokens[step] != serialToken || result.freshTokens[step] != serialToken
        }
        if let step = firstDisagreement {
            let margin = result.serial[step].margin
            XCTAssertLessThan(
                margin, Self.kernelPathMarginTolerance,
                "cached \(result.cachedTokens) and fresh \(result.freshTokens) diverge at step \(step), "
                    + "where the serial run's top-1/top-2 margin is \(margin): not a near-tie")
        }
    }

    private static func evaluateGate(context: ModelContext) async throws -> PrefixSchedulerGateResult {
        let model = context.model
        let blockSize = 256
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        let prefixTokens = try await makePrefixTokens(context: context, blockSize: blockSize)
        let suffixes = try await [
            " The answer is",
            " Therefore",
            " In summary",
            " Paris",
            " Swift",
            " The next",
        ].mapAsync { text in
            try await tokenIDs(for: text, context: context, parameters: parameters)
        }
        let unrelatedTokens = try await tokenIDs(
            for: "Completely separate prompt about oceans and mountains.",
            context: context,
            parameters: parameters
        )

        let manager = PagedCacheManager(blockSize: blockSize)
        let prefixCache = BlockAwarePrefixCache(
            modelName: "Qwen3-0.6B-4bit",
            blockSize: blockSize,
            manager: manager
        )
        let prefixStore = BlockAwarePrefixKVStore(prefixCache: prefixCache)

        let engine = MLXCatEngine(
            model: model,
            parameters: parameters,
            maxConcurrentRequests: 4,
            prefixStore: prefixStore
        )

        let warmTokens = prefixTokens + suffixes[0]
        _ = try await engine.generate([
            Request(
                uid: "warm",
                input: LMInput(text: LMInput.Text(tokens: MLXArray(warmTokens.map(Int32.init)))),
                maxTokens: 1,
                sampling: SamplingParameters(temperature: 0)
            )
        ])

        let allocatedBaseline = manager.allocatedBlocks
        let refBaseline = manager.totalRefCount
        XCTAssertEqual(allocatedBaseline, 1)

        let batchPrompts = [
            prefixTokens + suffixes[1],
            prefixTokens + suffixes[2],
            unrelatedTokens,
        ]
        let serial = try batchPrompts.map {
            try serialTrace(model: model, tokens: $0, parameters: parameters, steps: parameters.maxTokens ?? 4)
        }

        let requests = batchPrompts.enumerated().map { index, tokens in
            Request(
                uid: "batch-\(index)",
                input: LMInput(text: LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)))),
                maxTokens: parameters.maxTokens ?? 4,
                sampling: SamplingParameters(temperature: 0)
            )
        }
        let outputs = try await engine.generate(requests)

        var checkedTokens = 0
        var mismatches = 0
        for row in batchPrompts.indices {
            let comparison = compareWideMarginTokens(
                generated: outputs["batch-\(row)", default: []],
                serial: serial[row]
            )
            checkedTokens += comparison.checked
            mismatches += comparison.mismatches
        }

        return PrefixSchedulerGateResult(
            fetchHits: prefixStore.fetchHitCount,
            stores: prefixStore.storeCount,
            checkedTokens: checkedTokens,
            mismatches: mismatches,
            allocatedBaseline: allocatedBaseline,
            allocatedAfterBatch: manager.allocatedBlocks,
            refBaseline: refBaseline,
            refAfterBatch: manager.totalRefCount
        )
    }

    private static func evaluateSessionReuseGate(context: ModelContext) async throws -> SessionPrefixKVStoreStats {
        let parameters = GenerateParameters(maxTokens: 2, temperature: 0)
        let prefixTokens = try await makePrefixTokens(context: context, blockSize: 256)
        let first = prefixTokens + (try await tokenIDs(
            for: " First request.",
            context: context,
            parameters: parameters
        ))
        let second = first + (try await tokenIDs(
            for: " Second request extends the first.",
            context: context,
            parameters: parameters
        ))
        let prefixStore = SessionPrefixKVStore()
        let engine = MLXCatEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: 1,
            prefixStore: prefixStore
        )

        _ = try await engine.generate([
            Request(
                uid: "session-1",
                input: LMInput(text: LMInput.Text(tokens: MLXArray(first.map(Int32.init)))),
                maxTokens: parameters.maxTokens ?? 2,
                sampling: SamplingParameters(temperature: 0),
                cacheSession: "session-a"
            )
        ])
        _ = try await engine.generate([
            Request(
                uid: "session-2",
                input: LMInput(text: LMInput.Text(tokens: MLXArray(second.map(Int32.init)))),
                maxTokens: parameters.maxTokens ?? 2,
                sampling: SamplingParameters(temperature: 0),
                cacheSession: "session-a"
            )
        ])
        return prefixStore.stats
    }

    private struct SessionCorrectnessResult {
        let cachedTokens: [Int]
        let freshTokens: [Int]
        let serial: [(token: Int, margin: Float)]
        let cachedFetchHits: Int
        let stores: Int
        let warmPromptCount: Int
        let restoredTokenCount: Int
        let prefilledPositionsDiffering: [Int]
    }

    private static func evaluateSessionCorrectnessGate(
        context: ModelContext
    ) async throws -> SessionCorrectnessResult {
        let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
        let prefixTokens = try await makePrefixTokens(context: context, blockSize: 128)
        let first = prefixTokens + (try await tokenIDs(
            for: " Cache warmup request. Continue with a short deterministic answer.",
            context: context,
            parameters: parameters
        ))
        let second = first + (try await tokenIDs(
            for: " Extended request suffix. Answer in one sentence about Paris.",
            context: context,
            parameters: parameters
        ))

        let prefixStore = SessionPrefixKVStore()
        let cachedEngine = MLXCatEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: 1,
            prefixStore: prefixStore
        )
        _ = try await cachedEngine.generate([
            Request(
                uid: "warm",
                input: LMInput(text: LMInput.Text(tokens: MLXArray(first.map(Int32.init)))),
                maxTokens: parameters.maxTokens ?? 8,
                sampling: SamplingParameters(temperature: 0),
                cacheSession: "session-correctness"
            )
        ])

        let restored = try restoredPrefix(
            store: prefixStore, sessionKey: "session-correctness", prompt: second)
        let prefilledPositionsDiffering = try positionsDiffering(
            restored: restored.layers,
            fromColdPrefillOf: Array(first.dropLast()),
            context: context,
            parameters: parameters
        )

        let hitsBeforeCachedRequest = prefixStore.stats.fetchHitCount
        let cached = try await cachedEngine.generate([
            Request(
                uid: "cached",
                input: LMInput(text: LMInput.Text(tokens: MLXArray(second.map(Int32.init)))),
                maxTokens: parameters.maxTokens ?? 8,
                sampling: SamplingParameters(temperature: 0),
                cacheSession: "session-correctness"
            )
        ])["cached", default: []]
        let cachedFetchHits = prefixStore.stats.fetchHitCount - hitsBeforeCachedRequest

        let freshEngine = MLXCatEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: 1
        )
        let fresh = try await freshEngine.generate([
            Request(
                uid: "fresh",
                input: LMInput(text: LMInput.Text(tokens: MLXArray(second.map(Int32.init)))),
                maxTokens: parameters.maxTokens ?? 8,
                sampling: SamplingParameters(temperature: 0)
            )
        ])["fresh", default: []]

        let serial = try serialTrace(
            model: context.model,
            tokens: second,
            parameters: parameters,
            steps: parameters.maxTokens ?? 8
        ).map { (token: $0.token, margin: topOneTopTwoMargin($0.logits)) }

        return SessionCorrectnessResult(
            cachedTokens: cached,
            freshTokens: fresh,
            serial: serial,
            cachedFetchHits: cachedFetchHits,
            stores: prefixStore.stats.storeCount,
            warmPromptCount: first.count,
            restoredTokenCount: restored.matchedTokenCount,
            prefilledPositionsDiffering: prefilledPositionsDiffering
        )
    }

    /// Fetches, reconstructs and releases the slot `prompt` would be served from.
    private static func restoredPrefix(
        store: SessionPrefixKVStore,
        sessionKey: String,
        prompt: [Int]
    ) throws -> (matchedTokenCount: Int, layers: [SerializedKVLayer]) {
        guard let hit = store.fetch(tokens: prompt, sessionKey: sessionKey) else {
            XCTFail("the warm request left no session slot for the extended prompt")
            return (0, [])
        }
        defer { store.release(hit) }
        return (hit.matchedTokenCount, try store.reconstructCache(from: hit))
    }

    /// Positions in `0 ..< tokens.count` where any layer's restored keys or values
    /// differ from a cold prefill of `tokens`.
    private static func positionsDiffering(
        restored: [SerializedKVLayer],
        fromColdPrefillOf tokens: [Int],
        context: ModelContext,
        parameters: GenerateParameters
    ) throws -> [Int] {
        let cold = try context.model.newCache(parameters: parameters)
        let input = LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)))
        _ = context.model(input[text: .newAxis], cache: cold, state: nil)
        eval(cold)

        var differing = Set<Int>()
        for (layer, coldLayer) in zip(restored, cold) {
            for (restoredArray, coldArray) in zip(layer.state, coldLayer.state) {
                let positions = 0 ..< tokens.count
                let difference = abs(
                    restoredArray[0..., 0..., positions, 0...].asType(.float32)
                        - coldArray[0..., 0..., positions, 0...].asType(.float32)
                )
                let perPosition = difference.max(axes: [0, 1, 3]).asArray(Float.self)
                for (position, value) in perPosition.enumerated() where value > 0 {
                    differing.insert(position)
                }
            }
        }
        return differing.sorted()
    }

    private static func serialTrace(
        model: any LanguageModel,
        tokens: [Int],
        parameters: GenerateParameters,
        steps: Int
    ) throws -> [(token: Int, logits: MLXArray)] {
        let cache = try model.newCache(parameters: parameters)
        var state: LMOutput.State?
        let input = LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)))
        let output = model(input[text: .newAxis], cache: cache, state: state)
        state = output.state
        var currentLogits = output.logits[0..., -1, 0...]
        var currentToken = argMax(currentLogits, axis: -1)
        eval(currentToken, currentLogits, cache)

        var trace: [(token: Int, logits: MLXArray)] = []
        for _ in 0 ..< steps {
            eval(currentToken, currentLogits)
            trace.append((currentToken.item(Int.self), currentLogits))

            let stepOutput = model(
                LMInput.Text(tokens: currentToken[.newAxis, 0...]),
                cache: cache,
                state: state
            )
            state = stepOutput.state
            currentLogits = stepOutput.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
            asyncEval(currentToken, currentLogits)
        }

        eval(currentToken, currentLogits)
        return trace
    }

    private static func compareWideMarginTokens(
        generated: [Int],
        serial: [(token: Int, logits: MLXArray)]
    ) -> (checked: Int, mismatches: Int) {
        var checked = 0
        var mismatches = 0
        for (step, token) in generated.enumerated() where step < serial.count {
            let serialStep = serial[step]
            let margin = topOneTopTwoMargin(serialStep.logits)
            if margin > 1.25 * 4 + 1e-3 {
                checked += 1
                if token != serialStep.token {
                    mismatches += 1
                }
            }
        }
        return (checked, mismatches)
    }

    private static func makePrefixTokens(
        context: ModelContext,
        blockSize: Int
    ) async throws -> [Int] {
        let seedText = String(
            repeating: "The capital of France is Paris. Swift concurrency protects shared state. GPU kernels execute matrix operations quickly. ",
            count: 128
        )
        let seedTokens = try await tokenIDs(
            for: seedText,
            context: context,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )
        var prefixTokens: [Int] = []
        while prefixTokens.count < blockSize {
            prefixTokens.append(contentsOf: seedTokens)
        }
        return Array(prefixTokens.prefix(blockSize))
    }

    private static func tokenIDs(
        for text: String,
        context: ModelContext,
        parameters: GenerateParameters
    ) async throws -> [Int] {
        let input = try await context.processor.prepare(input: UserInput(prompt: text))
        let cache = try context.model.newCache(parameters: parameters)
        switch try context.model.prepare(input, cache: cache, state: nil, prefill: parameters.prefill) {
        case .tokens(let tokens):
            return tokens.tokens.asArray(Int.self)
        case .logits:
            throw PrefixSchedulerTestError.unexpectedPreparedLogits
        }
    }

    private static func topOneTopTwoMargin(_ logits: MLXArray) -> Float {
        let topValues = top(logits.asType(.float32), k: 2, axis: -1).asArray(Float.self)
        let sorted = topValues.sorted(by: >)
        return sorted[0] - sorted[1]
    }
}

private enum PrefixSchedulerTestError: Error {
    case unexpectedPreparedLogits
}

private extension Array {
    func mapAsync<T>(_ transform: (Element) async throws -> T) async throws -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self {
            result.append(try await transform(element))
        }
        return result
    }
}
