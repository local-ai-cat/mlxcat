import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXCat
import XCTest

/// What admission actually does with N waiting requests, made observable.
///
/// Our TTFT is roughly linear in concurrency where mlx-serve's is sublinear, and
/// nothing in the suite could see the shape of admission to say why — the
/// evidence lived only in hand-run benchmarks on a loaded Mac. These tests drive
/// the scheduler with a recording model, so the admission pattern is a unit test
/// with no weights and no Metal.
///
/// The reference behaviour they are written against
/// (`docs/PARITY-LEDGER.md`, open item 1): mlx-serve admits up to 16 slots per
/// tick and runs each prefill to COMPLETION, injecting a decode tick at each
/// chunk boundary (`guest/mlx-serve/src/scheduler.zig:3561`, `:3622`;
/// `guest/mlx-serve/src/generate.zig:1937`). Its own comment names the failure
/// mode: "without this every slot's first token waits for the LAST slot's
/// prefill (the TTFT staircase collapse)" (`scheduler.zig:3617`).
final class SchedulerAdmissionShapeTests: XCTestCase {
    private func tokenInput(_ count: Int) -> LMInput {
        LMInput(tokens: MLXArray((0 ..< count).map { Int32($0 % 97) }))
    }

    private func request(_ uid: String, promptTokens: Int, maxTokens: Int = 2) -> Request {
        Request(
            uid: uid,
            input: tokenInput(promptTokens),
            maxTokens: maxTokens,
            sampling: SamplingParameters(temperature: 0)
        )
    }

    /// An idle scheduler handed N requests at once must admit them all before it
    /// starts charging anyone's first token to the queue behind them. This pins
    /// that admission is not one-per-step on the cold path.
    func testIdleSchedulerAdmitsEveryWaitingRequestInOneStep() async throws {
        let model = AdmissionRecordingModel(vocabularySize: 16)
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: GenerateParameters(maxTokens: 2, temperature: 0),
            maxConcurrentRequests: 8
        )
        for index in 0 ..< 8 {
            try await scheduler.submit(request("r\(index)", promptTokens: 12))
        }

        // Each admitted request emits its first token in the step that admits
        // it, so the responses ARE the admission count — no test-only accessor
        // on the scheduler needed.
        let responses = try await scheduler.step()
        let admitted = Set(responses.filter { $0.token >= 0 }.map(\.uid)).count

        XCTAssertEqual(
            admitted, 8,
            "one step admitted \(admitted) of 8; every request after the first is paying "
                + "the whole queue's prefill before its own first token"
        )
    }

    /// The width the busy path prefills per chunk. 512 was out of line with every
    /// reference (mlx-lm and omlx 2048, mlx-serve 8192) and every boundary costs
    /// an actor round trip plus a decode tick for the running streams, so the
    /// count of forwards for a given prompt is the scheduling tax. Pinned so a
    /// change to it is deliberate and shows up in review.
    func testPrefillChunkWidthMatchesTheReferenceRange() async throws {
        let model = AdmissionRecordingModel(vocabularySize: 16)
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: GenerateParameters(maxTokens: 1, temperature: 0),
            maxConcurrentRequests: 4
        )
        // Something must be decoding for the BUSY width to apply — that is the
        // whole distinction. Admit a short request first, then the long one.
        try await scheduler.submit(request("keepalive", promptTokens: 4, maxTokens: 200))
        _ = try await scheduler.step()
        _ = model.drainEvents()

        try await scheduler.submit(request("long", promptTokens: 5000, maxTokens: 1))
        var prefillWidths: [Int] = []
        for _ in 0 ..< 8 {
            _ = try await scheduler.step()
            prefillWidths += model.drainEvents().map { $0.shape.count > 1 ? $0.shape[1] : 1 }
        }

        let widest = prefillWidths.max() ?? 0
        XCTAssertEqual(
            widest, 2048,
            "busy prefill chunk is \(widest); the reference range is 2048 (mlx-lm, omlx) "
                + "to 8192 (mlx-serve), and every boundary costs a decode tick plus an "
                + "actor round trip. Widths: \(prefillWidths)"
        )
    }

    /// A long prompt arriving while others decode must not stop their tokens.
    /// This is the property mlx-serve gets by injecting a decode tick at each
    /// chunk boundary; we get it by yielding between chunks. Either way the
    /// running stream must keep producing while the newcomer prefills.
    func testALongPrefillDoesNotStarveAlreadyRunningStreams() async throws {
        let model = AdmissionRecordingModel(vocabularySize: 16)
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: GenerateParameters(maxTokens: 40, temperature: 0),
            maxConcurrentRequests: 4
        )
        try await scheduler.submit(request("short", promptTokens: 8, maxTokens: 40))
        _ = try await scheduler.step()   // admits and emits its first token

        try await scheduler.submit(request("long", promptTokens: 9000, maxTokens: 2))

        var shortTokens = 0
        for _ in 0 ..< 12 {
            let responses = try await scheduler.step()
            shortTokens += responses.filter { $0.uid == "short" && $0.token >= 0 }.count
        }

        XCTAssertGreaterThan(
            shortTokens, 0,
            "the running stream produced no tokens across 12 steps while a 9000-token "
                + "prefill was admitted — this is the starvation the chunk boundary exists to prevent"
        )
    }
}

private struct AdmissionRecordedCall: Equatable {
    let shape: [Int]
}

/// Minimal `LanguageModel` that records the shape of every forward pass and
/// grows a fake KV cache, so admission and prefill chunking are observable with
/// no weights and no Metal.
private final class AdmissionRecordingModel: Module, LanguageModel {
    private let vocabularySize: Int
    private let lock = NSLock()
    private var events: [AdmissionRecordedCall] = []

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        lock.lock()
        events.append(AdmissionRecordedCall(shape: input.tokens.shape))
        lock.unlock()
        let batch = input.tokens.dim(0)
        let sequence = input.tokens.shape.count > 1 ? input.tokens.dim(1) : 1
        cache?.forEach { layerCache in
            guard let simple = layerCache as? KVCacheSimple else { return }
            let cached = simple.state.first?.dim(2) ?? 0
            let total = cached + max(1, sequence)
            simple.state = [
                MLXArray.zeros([batch, 1, total, 1]),
                MLXArray.zeros([batch, 1, total, 1]),
            ]
        }
        return LMOutput(logits: MLXArray.zeros([batch, 1, vocabularySize]))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    func drainEvents() -> [AdmissionRecordedCall] {
        lock.lock()
        defer { lock.unlock() }
        let result = events
        events.removeAll()
        return result
    }
}
