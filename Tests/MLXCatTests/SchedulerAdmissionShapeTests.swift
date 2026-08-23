import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXCat
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
    /// starts charging anyone's first token to the queue behind them.
    ///
    /// This pins the admission LOOP. It is deliberately paired with
    /// `testASerializingPolicyQueuesBehindWholeGenerations` below, because on its
    /// own it gives false comfort: it runs under the default policy, and for a
    /// serializing family the guard one line above the loop
    /// (`refusesToJoin`) means requests 2..N never reach this mechanism at all.
    /// I read a pass here as "admission count is not the problem" and it was not
    /// entitled to answer that question — for the families on the leaderboard the
    /// answer was the opposite.
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

    /// The measured TTFT collapse, as a unit test.
    ///
    /// mlxcat's TTFT was linear in N — and the reason was not that N prefills run
    /// serially. Under `.always` each waiting request queues behind the
    /// predecessor's ENTIRE GENERATION, so the per-unit cost is prefill PLUS all
    /// its decode steps. The bench fit is within 1.3% at every width on two
    /// models using G = prefill + tokens/decode-rate, and mlx-serve queues behind
    /// PREFILLS only — that ratio is the whole gap, not any prefill-speed
    /// difference.
    ///
    /// So the property worth pinning is not "how many admissions per step" but
    /// "does a second request get in while the first is still generating".
    func testASerializingPolicyQueuesBehindWholeGenerations() async throws {
        let model = AdmissionRecordingModel(vocabularySize: 16)
        let scheduler = Scheduler(
            modelBox: LanguageModelBox(model),
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            maxConcurrentRequests: 8,
            serializationPolicy: .always
        )
        for index in 0 ..< 4 {
            try await scheduler.submit(request("s\(index)", promptTokens: 8, maxTokens: 8))
        }

        let first = try await scheduler.step()
        let admittedFirst = Set(first.filter { $0.token >= 0 }.map(\.uid)).count
        XCTAssertEqual(
            admittedFirst, 1,
            "`.always` must admit exactly one; admitting \(admittedFirst) would mean the "
                + "solitude guard is not holding")

        // While that one generates, nobody else may start. This is the drain
        // that made TTFT linear in generations rather than in prefills.
        var seen: Set<String> = Set(first.filter { $0.token >= 0 }.map(\.uid))
        for _ in 0 ..< 4 {
            let responses = try await scheduler.step()
            seen.formUnion(responses.filter { $0.token >= 0 }.map(\.uid))
        }
        XCTAssertEqual(
            seen.count, 1,
            "a second request started while the first was still generating under `.always`; "
                + "either the policy changed or this test no longer describes it — \(seen)")

        // And a non-serializing policy must NOT behave that way, or the contrast
        // this test exists to draw is vacuous.
        let openModel = AdmissionRecordingModel(vocabularySize: 16)
        let open = Scheduler(
            modelBox: LanguageModelBox(openModel),
            parameters: GenerateParameters(maxTokens: 8, temperature: 0),
            maxConcurrentRequests: 8,
            serializationPolicy: .never
        )
        for index in 0 ..< 4 {
            try await open.submit(request("o\(index)", promptTokens: 8, maxTokens: 8))
        }
        let openFirst = try await open.step()
        XCTAssertEqual(
            Set(openFirst.filter { $0.token >= 0 }.map(\.uid)).count, 4,
            "`.never` should admit the whole burst in one step")
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

/// The last-token-alone prefill optimisation and the one cache family it breaks.
///
/// It is safe for mlx-lm because its single remaining token goes through
/// `_step()` — the same decode path every later token uses
/// (`guest/mlx-lm/mlx_lm/generate.py:580-587`). Ours goes through the PREFILL
/// call site, and a rotating (sliding-window) cache has separate multi-token and
/// single-token update paths, so the extra `S == 1` prefill update desyncs the
/// ring. Measured on gpt-oss-20b (window 128): 60 of 160 tokens diverged from
/// serial with a sustained run of 37, and bisecting this change alone restored
/// 3/3.
final class PrefillLastTokenAlonePolicyTests: XCTestCase {
    func testWindowedCachesKeepTheOldPathAndEverythingElseTakesIt() {
        let none: [String: String] = [:]
        XCTAssertTrue(
            Scheduler.prefillsLastTokenAlone(usesWindowedKVCache: false, environment: none),
            "a non-windowed cache should take the smaller logits tensor")
        XCTAssertFalse(
            Scheduler.prefillsLastTokenAlone(usesWindowedKVCache: true, environment: none),
            "a rotating cache desyncs on the extra prefill-path single-token update")
    }

    func testTheOverrideForcesEitherWayForMeasurement() {
        // `always` on a windowed model is how the divergence is reproduced —
        // the toggle exists so the trade-off is measurable, not so it is hidden.
        XCTAssertTrue(
            Scheduler.prefillsLastTokenAlone(
                usesWindowedKVCache: true,
                environment: ["MLXCAT_PREFILL_LAST_TOKEN_ALONE": "always"]))
        XCTAssertFalse(
            Scheduler.prefillsLastTokenAlone(
                usesWindowedKVCache: false,
                environment: ["MLXCAT_PREFILL_LAST_TOKEN_ALONE": "never"]))
        // An unrecognised value falls back to the capability rule rather than
        // silently picking one.
        XCTAssertTrue(
            Scheduler.prefillsLastTokenAlone(
                usesWindowedKVCache: false,
                environment: ["MLXCAT_PREFILL_LAST_TOKEN_ALONE": "banana"]))
    }
}
