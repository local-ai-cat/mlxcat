import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXCat
import XCTest

final class SchedulerChunkedPrefillTests: XCTestCase {
    func testLongPromptAdmissionYieldsBetweenPrefillChunksDuringActiveDecode() async throws {
        let parameters = GenerateParameters(
            maxTokens: 4,
            temperature: 0,
            prefill: PrefillParameters(stepSize: 2, chunking: .remainder)
        )
        let model = RecordingChunkedLanguageModel(vocabularySize: 16)
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: parameters,
            maxConcurrentRequests: 2
        )

        try await scheduler.submit(
            Request(
                uid: "active",
                input: tokenInput([1, 2]),
                maxTokens: 4,
                sampling: SamplingParameters(temperature: 0)
            )
        )
        var interleavedResponses = try await scheduler.step()
        XCTAssertEqual(interleavedResponses.generatedCount(for: "active"), 1)

        try await scheduler.submit(
            Request(
                uid: "long",
                input: tokenInput([10, 11, 12, 13, 14, 15]),
                maxTokens: 3,
                sampling: SamplingParameters(temperature: 0)
            )
        )

        _ = model.drainEvents()
        let stepTwo = try await scheduler.step()
        let stepTwoEvents = model.drainEvents()
        interleavedResponses.append(contentsOf: stepTwo)

        XCTAssertEqual(stepTwo.generatedCount(for: "active"), 1)
        XCTAssertEqual(stepTwo.generatedCount(for: "long"), 0)
        XCTAssertTrue(stepTwoEvents.containsOrderedShapes([[1, 2], [1, 1]]))
        XCTAssertLessThanOrEqual(stepTwoEvents.maxPrefillWidth, 2)

        let stepThree = try await scheduler.step()
        let stepThreeEvents = model.drainEvents()
        interleavedResponses.append(contentsOf: stepThree)

        XCTAssertEqual(stepThree.generatedCount(for: "active"), 1)
        XCTAssertEqual(stepThree.generatedCount(for: "long"), 0)
        XCTAssertTrue(stepThreeEvents.containsOrderedShapes([[1, 2], [1, 1]]))
        XCTAssertLessThanOrEqual(stepThreeEvents.maxPrefillWidth, 2)

        while await !scheduler.isIdle {
            interleavedResponses.append(contentsOf: try await scheduler.step())
        }

        let interleavedLongTokens = interleavedResponses.generatedTokens(for: "long")
        let soloLongTokens = try await runSoloLongPrompt(parameters: parameters)
        XCTAssertEqual(interleavedLongTokens, soloLongTokens)
    }

    func testSchedulerManagedTextPrefillCanBeDisabledForPreparedLogitsModels() async throws {
        let model = PreparedLogitsLanguageModel(vocabularySize: 16)
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: GenerateParameters(
                maxTokens: 1,
                temperature: 0,
                prefill: PrefillParameters(stepSize: 2, chunking: .remainder)
            ),
            maxConcurrentRequests: 1,
            schedulerManagedTextPrefill: false
        )

        try await scheduler.submit(
            Request(
                uid: "vlm-text",
                input: tokenInput([10, 11, 12, 13, 14, 15]),
                maxTokens: 1,
                sampling: SamplingParameters(temperature: 0)
            )
        )
        let responses = try await scheduler.step()

        XCTAssertEqual(responses.generatedCount(for: "vlm-text"), 1)
        XCTAssertEqual(model.prepareCallCount, 1)
        XCTAssertTrue(model.drainEvents().isEmpty)
    }

    func testExactPrefixHitReevaluatesCompletePrompt() async throws {
        let model = RecordingChunkedLanguageModel(vocabularySize: 16)
        let store = SessionPrefixKVStore()
        let engine = MLXCatEngine(
            model: model,
            parameters: GenerateParameters(
                maxTokens: 1,
                temperature: 0,
                prefill: PrefillParameters(stepSize: 2, chunking: .remainder)
            ),
            maxConcurrentRequests: 1,
            prefixStore: store
        )
        let input = tokenInput([10, 11, 12, 13])

        _ = try await engine.generate([
            Request(
                uid: "first",
                input: input,
                maxTokens: 1,
                sampling: SamplingParameters(temperature: 0),
                cacheSession: "exact-prefix"
            )
        ])
        _ = model.drainEvents()

        _ = try await engine.generate([
            Request(
                uid: "second",
                input: input,
                maxTokens: 1,
                sampling: SamplingParameters(temperature: 0),
                cacheSession: "exact-prefix"
            )
        ])

        XCTAssertEqual(model.drainEvents().map(\.shape), [[1, 4]])
        XCTAssertEqual(store.stats.fetchHitCount, 1)
        XCTAssertEqual(store.stats.releaseCount, 1)
    }

    private func runSoloLongPrompt(parameters: GenerateParameters) async throws -> [Int] {
        let model = RecordingChunkedLanguageModel(vocabularySize: 16)
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: parameters,
            maxConcurrentRequests: 1
        )
        try await scheduler.submit(
            Request(
                uid: "long",
                input: tokenInput([10, 11, 12, 13, 14, 15]),
                maxTokens: 3,
                sampling: SamplingParameters(temperature: 0)
            )
        )

        var responses: [Response] = []
        while await !scheduler.isIdle {
            responses.append(contentsOf: try await scheduler.step())
        }
        return responses.generatedTokens(for: "long")
    }

    private func tokenInput(_ tokens: [Int32]) -> LMInput {
        LMInput(tokens: MLXArray(tokens))
    }
}

private struct RecordedModelCall: Equatable {
    let shape: [Int]
}

private final class RecordingChunkedLanguageModel: Module, LanguageModel {
    private let vocabularySize: Int
    private let lock = NSLock()
    private var events: [RecordedModelCall] = []

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        record(shape: input.tokens.shape)
        let batch = input.tokens.dim(0)
        let sequence = input.tokens.shape.count > 1 ? input.tokens.dim(1) : 1
        cache?.forEach { layerCache in
            guard let simple = layerCache as? KVCacheSimple else { return }
            let cachedSequence = simple.state.first?.dim(2) ?? 0
            let totalSequence = cachedSequence + max(1, sequence)
            simple.state = [
                MLXArray.zeros([batch, 1, totalSequence, 1]),
                MLXArray.zeros([batch, 1, totalSequence, 1]),
            ]
        }
        return LMOutput(logits: MLXArray.zeros([batch, 1, vocabularySize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    func drainEvents() -> [RecordedModelCall] {
        lock.lock()
        defer { lock.unlock() }
        let result = events
        events.removeAll()
        return result
    }

    private func record(shape: [Int]) {
        lock.lock()
        events.append(RecordedModelCall(shape: shape))
        lock.unlock()
    }
}

private final class PreparedLogitsLanguageModel: Module, LanguageModel {
    private let vocabularySize: Int
    private let lock = NSLock()
    private var events: [RecordedModelCall] = []
    private var prepareCalls = 0

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    var prepareCallCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return prepareCalls
    }

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        lock.lock()
        prepareCalls += 1
        lock.unlock()
        return .logits(LMOutput(logits: MLXArray.zeros([1, 1, vocabularySize])))
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        lock.lock()
        events.append(RecordedModelCall(shape: input.tokens.shape))
        lock.unlock()
        return LMOutput(logits: MLXArray.zeros([input.tokens.dim(0), 1, vocabularySize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    func drainEvents() -> [RecordedModelCall] {
        lock.lock()
        defer { lock.unlock() }
        let result = events
        events.removeAll()
        return result
    }
}

private extension Array where Element == Response {
    func generatedCount(for uid: String) -> Int {
        generatedTokens(for: uid).count
    }

    func generatedTokens(for uid: String) -> [Int] {
        filter { $0.uid == uid && $0.token >= 0 }.map(\.token)
    }
}

private extension Array where Element == RecordedModelCall {
    var maxPrefillWidth: Int {
        map(\.shape)
            .filter { $0.count == 2 && $0[0] == 1 && $0[1] > 1 }
            .map { $0[1] }
            .max() ?? 0
    }

    func containsOrderedShapes(_ expectedShapes: [[Int]]) -> Bool {
        var nextExpectedIndex = expectedShapes.startIndex
        for event in self where nextExpectedIndex < expectedShapes.endIndex {
            if event.shape == expectedShapes[nextExpectedIndex] {
                nextExpectedIndex = expectedShapes.index(after: nextExpectedIndex)
            }
        }
        return nextExpectedIndex == expectedShapes.endIndex
    }
}
