import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXServe
import XCTest

final class SchedulerSpeculativeDecodingTests: XCTestCase {
    func testNgramSpeculationProducesIdenticalTokensWithFewerDecodeCalls() async throws {
        let prompt: [Int32] = [1, 2, 3, 1, 2, 3, 1, 2]
        let baselineModel = RepeatingTokenLanguageModel(vocabularySize: 8)
        let baseline = try await generate(
            prompt: prompt,
            model: baselineModel,
            speculativeDecoding: .disabled
        )

        let speculativeModel = RepeatingTokenLanguageModel(vocabularySize: 8)
        let speculative = try await generate(
            prompt: prompt,
            model: speculativeModel,
            speculativeDecoding: SpeculativeDecodingConfiguration(
                enabled: true,
                maxProposalTokens: 4,
                minContextTokens: 8
            )
        )

        XCTAssertEqual(speculative.tokens, baseline.tokens)
        XCTAssertEqual(speculative.tokens.count, Self.maxTokens)
        XCTAssertGreaterThan(speculative.stats.acceptedTokenCount, 0)
        XCTAssertGreaterThan(speculative.stats.proposalBatchCount, 0)
        XCTAssertLessThan(speculativeModel.decodeCallCount, baselineModel.decodeCallCount)
        XCTAssertGreaterThan(
            speculativeModel.generatedTokensPerDecodeCall,
            baselineModel.generatedTokensPerDecodeCall
        )
    }

    func testRejectedNgramProposalFallsBackWithoutChangingOutput() async throws {
        let prompt: [Int32] = [1, 2, 3, 1, 2, 3, 1, 2]
        let baselineModel = MismatchingSuffixLanguageModel(vocabularySize: 8)
        let baseline = try await generate(
            prompt: prompt,
            model: baselineModel,
            speculativeDecoding: .disabled
        )

        let speculativeModel = MismatchingSuffixLanguageModel(vocabularySize: 8)
        let speculative = try await generate(
            prompt: prompt,
            model: speculativeModel,
            speculativeDecoding: SpeculativeDecodingConfiguration(
                enabled: true,
                maxProposalTokens: 4,
                minContextTokens: 8
            )
        )

        XCTAssertEqual(speculative.tokens, baseline.tokens)
        XCTAssertGreaterThan(speculative.stats.rejectedBatchCount, 0)
    }

    private static let maxTokens = 10

    private func generate(
        prompt: [Int32],
        model: SpeculativeTestLanguageModel,
        speculativeDecoding: SpeculativeDecodingConfiguration
    ) async throws -> (tokens: [Int], stats: SpeculativeDecodingStats) {
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: GenerateParameters(
                maxTokens: Self.maxTokens,
                temperature: 0,
                prefill: PrefillParameters(stepSize: 32, chunking: .remainder)
            ),
            maxConcurrentRequests: 1,
            speculativeDecoding: speculativeDecoding
        )

        try await scheduler.submit(
            Request(
                uid: "repeat",
                input: LMInput(tokens: MLXArray(prompt)),
                maxTokens: Self.maxTokens,
                sampling: SamplingParameters(temperature: 0)
            )
        )

        var responses: [Response] = []
        while await !scheduler.isIdle {
            responses.append(contentsOf: try await scheduler.step())
        }
        return (
            responses.generatedTokens(for: "repeat"),
            await scheduler.speculativeDecodingStats
        )
    }
}

private typealias SpeculativeTestLanguageModel = Module & LanguageModel & DecodeCallRecording

private protocol DecodeCallRecording: AnyObject {
    var decodeCallCount: Int { get }
    var generatedTokensPerDecodeCall: Double { get }
}

private class RepeatingTokenLanguageModel: Module, LanguageModel, DecodeCallRecording {
    private let vocabularySize: Int
    private let lock = NSLock()
    private var decodeInputWidths: [Int] = []

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    var decodeCallCount: Int {
        lock.withLock { decodeInputWidths.count }
    }

    var generatedTokensPerDecodeCall: Double {
        lock.withLock {
            guard !decodeInputWidths.isEmpty else { return 0 }
            let generated = decodeInputWidths.reduce(0, +)
            return Double(generated) / Double(decodeInputWidths.count)
        }
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        let batch = input.tokens.dim(0)
        let sequence = input.tokens.shape.count > 1 ? input.tokens.dim(1) : 1
        let previousLength = cache?.first.flatMap { ($0 as? KVCacheSimple)?.state.first?.dim(1) } ?? 0
        if previousLength > 0 {
            recordDecode(width: sequence)
        }
        cache?.forEach { layerCache in
            guard let simple = layerCache as? KVCacheSimple else { return }
            let layerPreviousLength = simple.state.first?.dim(1) ?? 0
            simple.state = [
                MLXArray.zeros([batch, layerPreviousLength + sequence, 1, 1]),
                MLXArray.zeros([batch, layerPreviousLength + sequence, 1, 1]),
            ]
        }

        return LMOutput(logits: logits(for: input.tokens, batch: batch, sequence: sequence))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    func nextToken(after token: Int) -> Int {
        switch token {
        case 1:
            return 2
        case 2:
            return 3
        default:
            return 1
        }
    }

    private func logits(for tokens: MLXArray, batch: Int, sequence: Int) -> MLXArray {
        let inputTokens = tokens.asArray(Int.self)
        var values = Array(repeating: Float(-20), count: batch * sequence * vocabularySize)
        for batchIndex in 0 ..< batch {
            for position in 0 ..< sequence {
                let token = inputTokens[batchIndex * sequence + position]
                let next = nextToken(after: token)
                let offset = (batchIndex * sequence + position) * vocabularySize + next
                values[offset] = 20
            }
        }
        return MLXArray(values).reshaped([batch, sequence, vocabularySize])
    }

    private func recordDecode(width: Int) {
        lock.withLock {
            decodeInputWidths.append(width)
        }
    }
}

private final class MismatchingSuffixLanguageModel: RepeatingTokenLanguageModel {
    override func nextToken(after token: Int) -> Int {
        token == 3 ? 4 : super.nextToken(after: token)
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}

private extension Array where Element == Response {
    func generatedTokens(for uid: String) -> [Int] {
        filter { $0.uid == uid && $0.token >= 0 }.map(\.token)
    }
}
