import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXCat
import XCTest

final class SchedulerPreemptionTests: XCTestCase {
    func testPreemptedTextRequestResumesWithByteIdenticalTokens() async throws {
        let baselineTokens = try await runGeneration(blockSize: 2, preemptionTrigger: nil).tokens
        let trigger = CountingPreemptionTrigger(preemptOnRunningSnapshots: [2, 4])
        let result = try await runGeneration(blockSize: 2, preemptionTrigger: trigger)

        XCTAssertEqual(result.tokens, baselineTokens)
        XCTAssertEqual(result.tokens.count, 4)
        XCTAssertEqual(result.droppedStaleResponses, 0)
        XCTAssertGreaterThan(result.prefixStore.fetchHitCount, 0)
    }

    func testGeneratedTokensArePublishedAsPrefixBlocks() async throws {
        let trigger = CountingPreemptionTrigger(preemptOnRunningSnapshots: [2])
        let result = try await runGeneration(blockSize: 3, preemptionTrigger: trigger)
        let hitCountAfterResume = result.prefixStore.fetchHitCount

        let secondScheduler = Scheduler(
            modelBox: LanguageModelBox(LengthTrackingLanguageModel(vocabularySize: 8)),
            parameters: Self.parameters,
            maxConcurrentRequests: 1,
            prefixStore: result.prefixStore
        )
        try await secondScheduler.submit(
            Request(
                uid: "second",
                input: tokenInput([1, 2, 0, 7]),
                maxTokens: 1,
                sampling: SamplingParameters(temperature: 0)
            )
        )
        while await !secondScheduler.isIdle {
            _ = try await secondScheduler.step()
        }

        XCTAssertGreaterThan(result.prefixStore.fetchHitCount, hitCountAfterResume)
    }

    func testPreemptionAfterToolTriggerResumesWithBodyGrammarArmed() async throws {
        let trigger = CountingPreemptionTrigger(preemptOnRunningSnapshots: [1])
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(LengthTrackingLanguageModel(vocabularySize: 6)),
            parameters: Self.parameters,
            maxConcurrentRequests: 1,
            pressurePolicy: Scheduler.PressurePolicy { snapshot in
                trigger.shouldPreempt(snapshot)
            }
        )
        let vocabulary = JSONGrammarVocabulary(tokens: [
            JSONGrammarToken(id: 0, text: "invalid"),
            JSONGrammarToken(id: 1, text: "<function=bash>"),
            JSONGrammarToken(id: 2, text: "</function>"),
            JSONGrammarToken(id: 3, text: "</tool_call>"),
            JSONGrammarToken(id: 4, text: " "),
            JSONGrammarToken(id: 5, text: "", isEOS: true),
        ])
        let toolGrammar = QwenXMLToolGrammarConfiguration(
            vocabulary: vocabulary,
            toolNames: ["bash"],
            toolCallStartTokenID: 0,
            toolCallEndTokenID: 3,
            postToolAllowedTokenIDs: [0, 4, 5]
        )

        try await scheduler.submit(
            Request(
                uid: "preempted",
                input: tokenInput([4, 4]),
                maxTokens: 4,
                sampling: SamplingParameters(temperature: 0, toolGrammar: toolGrammar)
            )
        )

        var responses: [Response] = []
        while await !scheduler.isIdle {
            responses.append(contentsOf: try await scheduler.step())
        }

        XCTAssertEqual(responses.generatedTokens(for: "preempted"), [0, 1, 2, 3])
    }

    private static let parameters = GenerateParameters(
        maxTokens: 4,
        temperature: 0,
        prefill: PrefillParameters(stepSize: 2, chunking: .remainder)
    )

    private func runGeneration(
        blockSize: Int,
        preemptionTrigger: CountingPreemptionTrigger?
    ) async throws -> (
        tokens: [Int],
        droppedStaleResponses: Int,
        prefixStore: BlockAwarePrefixKVStore
    ) {
        let prefixCache = BlockAwarePrefixCache(modelName: "synthetic", blockSize: blockSize)
        let prefixStore = BlockAwarePrefixKVStore(prefixCache: prefixCache)
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(LengthTrackingLanguageModel(vocabularySize: 8)),
            parameters: Self.parameters,
            maxConcurrentRequests: 1,
            prefixStore: prefixStore,
            pressurePolicy: Scheduler.PressurePolicy { snapshot in
                preemptionTrigger?.shouldPreempt(snapshot) ?? false
            }
        )

        try await scheduler.submit(
            Request(
                uid: "preempted",
                input: tokenInput([1, 2]),
                maxTokens: 4,
                sampling: SamplingParameters(temperature: 0)
            )
        )

        var responses: [Response] = []
        while await !scheduler.isIdle {
            responses.append(contentsOf: try await scheduler.step())
        }

        return (
            responses.generatedTokens(for: "preempted"),
            await scheduler.droppedStaleResponses,
            prefixStore
        )
    }
}

private final class CountingPreemptionTrigger: @unchecked Sendable {
    private let preemptOnRunningSnapshots: Set<Int>
    private let lock = NSLock()
    private var runningSnapshotCount = 0

    init(preemptOnRunningSnapshots: Set<Int>) {
        self.preemptOnRunningSnapshots = preemptOnRunningSnapshots
    }

    func shouldPreempt(_ snapshot: Scheduler.PressureSnapshot) -> Bool {
        guard snapshot.runningUIDs.contains("preempted") else {
            return false
        }

        lock.lock()
        defer { lock.unlock() }
        runningSnapshotCount += 1
        return preemptOnRunningSnapshots.contains(runningSnapshotCount)
    }
}

private final class LengthTrackingLanguageModel: Module, LanguageModel {
    private let vocabularySize: Int

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        let batch = input.tokens.dim(0)
        let sequence = input.tokens.shape.count > 1 ? input.tokens.dim(1) : 1
        cache?.forEach { layerCache in
            guard let simple = layerCache as? KVCacheSimple else { return }
            let previousLength = simple.state.first?.dim(1) ?? 0
            simple.state = [
                MLXArray.zeros([batch, previousLength + sequence, 1, 1]),
                MLXArray.zeros([batch, previousLength + sequence, 1, 1]),
            ]
        }
        return LMOutput(logits: MLXArray.zeros([batch, 1, vocabularySize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }
}

private func tokenInput(_ tokens: [Int32]) -> LMInput {
    LMInput(tokens: MLXArray(tokens))
}

private extension Array where Element == Response {
    func generatedTokens(for uid: String) -> [Int] {
        filter { $0.uid == uid && $0.token >= 0 }.map(\.token)
    }
}
