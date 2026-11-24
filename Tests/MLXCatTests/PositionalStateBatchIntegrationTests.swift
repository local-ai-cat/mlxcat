import MLX
import MLXHuggingFace
import MLXLMCommon
@testable import MLXVLM
@testable import MLXCat
import Tokenizers
import XCTest

/// Batched decode for a family that anchors its RoPE in `LMOutput.State`.
///
/// `docs/KNOWN-FAILURES.md` §1d said the batched `qwen3_5` failure lived in the
/// SERVER, because `HybridBatchScaleTests` cleared the engine at 8 rows x 512
/// tokens. It did not. The crash report the benchmark left behind
/// (`mlxcat-http-2026-08-23-022441.ips`, `EXC_BREAKPOINT`) names the frame:
///
///     Swift runtime failure: precondition failure
///     specialized Qwen35.callAsFunction(_:cache:state:)      Qwen35.swift:1298
///     ContinuousBatchGenerator.computeNextTokens(...)        BatchGenerator.swift:567
///
/// Qwen35 refuses to continue a warm cache without `qwen35.ropeDeltas`, and the
/// batch generator dropped that state the moment a second row joined. Width 4
/// losing its `usage` frame and width 8 killing the process were the same trap
/// at two timings. `HybridBatchScaleTests` could not see it because it exercises
/// a hybrid CACHE, not this MODEL — the defect is the model's positional
/// contract, not the cache shape.
///
/// Two things are asserted here that a same-length batch would not catch:
///
///   * rows of DIFFERENT prompt lengths, so `leftPadding` is non-zero and the
///     `offset - leftPadding[row]` correction in ``BatchPositionalState`` is
///     actually under test. Feed the raw prefill anchor instead and the short
///     rows decode from the padded batch length — wrong positions, plausible
///     text, no crash.
///   * loading through `VLMModelFactory`, which is the path
///     `NativeModelLoader` takes in production (`qwen3_5` is in
///     `vlmModelTypes`). `BatchInvarianceTests` loads Qwen3.5 through
///     `LLMModelFactory`, whose `Qwen35Model` is a DIFFERENT type with no such
///     precondition — which is why its clean sweep across widths 1/2/4/8 never
///     saw this.
///
///     MLXCAT_POSITIONAL_TEST_MODEL=~/Library/Caches/models/mlx-community/Qwen3.5-4B-MLX-4bit \
///       swift test --filter PositionalStateBatchIntegrationTests
final class PositionalStateBatchIntegrationTests: XCTestCase {

    /// Deliberately ragged: the token counts differ enough that every row but
    /// the longest carries left padding.
    static let prompts = [
        "Name one primary colour:",
        "In one short sentence, say what a KV cache stores during autoregressive decoding:",
        "List three consecutive integers starting at seven, separated by commas:",
        "Explain, in a single sentence and without examples, why batching requests of different prompt lengths requires left padding before the attention mask is applied:",
    ]

    private static func modelURL() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MLXCAT_POSITIONAL_TEST_MODEL"] else {
            throw XCTSkip(
                "Set MLXCAT_POSITIONAL_TEST_MODEL to a Qwen3.5 checkpoint to run this gate.")
        }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    func testBatchedDecodeMatchesSerialGreedyTokensAcrossRaggedRows() async throws {
        try MLXMetalRuntime.requireAvailable()
        let url = try Self.modelURL()

        let container = try await VLMModelFactory.shared.loadContainer(
            from: url, using: #huggingFaceTokenizerLoader())

        try await container.perform { context in
            let tokenCount = 8
            let parameters = GenerateParameters(maxTokens: tokenCount, temperature: 0)

            var inputs: [LMInput] = []
            for prompt in Self.prompts {
                inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }

            // Prove the rows really are ragged before trusting the comparison.
            let lengths = inputs.map { $0.text.tokens.dim(-1) }
            XCTAssertGreaterThan(
                Set(lengths).count, 1,
                "every prompt tokenized to the same length — left padding is never exercised")

            let serial = try inputs.map {
                try SerialGreedyTokenHelper.tokens(
                    model: context.model, input: $0, parameters: parameters, steps: tokenCount)
            }

            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: inputs.count,
                serializationPolicy: .never
            )
            let batched = try await engine.generate(
                inputs.enumerated().map { index, input in
                    Request(
                        uid: "positional-\(index)",
                        input: input,
                        maxTokens: tokenCount,
                        sampling: SamplingParameters(temperature: 0)
                    )
                }
            )

            for index in inputs.indices {
                XCTAssertEqual(
                    batched["positional-\(index)"], serial[index],
                    "row \(index) (prompt length \(lengths[index])) diverged from serial")
            }
        }
    }

    private static func traceFirstDecodeProjection(_ model: Qwen35, tokens: [[Int]]) throws {
        let ids = try tokens.map { Int32(try XCTUnwrap($0.first)) }
        let layer = try XCTUnwrap(model.loraLayers.first as? Qwen35Language.DecoderLayer)
        let attention = try XCTUnwrap(layer.linearAttn)
        let embed = model.languageModel.model.embedTokens
        let batch = embed(MLXArray(ids).reshaped(ids.count, 1))
        let normalized = layer.inputLayerNorm(batch)
        let projected = attention.projectInputs(normalized, batch: ids.count, sequence: 1)

        func report(_ name: String, _ batch: MLXArray, _ singletons: [MLXArray]) {
            let reference = concatenated(singletons, axis: 0)
            let delta = abs(batch.asType(.float32) - reference.asType(.float32))
            print("WIDTH8 \(name) max \(delta.max().item(Float.self))"
                + " changed \((delta .!= 0).sum().item(Int.self))/\(delta.size)")
        }

        let singles = ids.map { embed(MLXArray([$0]).reshaped(1, 1)) }
        let norms = singles.map { layer.inputLayerNorm($0) }
        let projections = norms.map { attention.projectInputs($0, batch: 1, sequence: 1) }
        report("embedding", batch, singles)
        report("input layernorm", normalized, norms)
        report("gated delta QKV", projected.qkv, projections.map(\.qkv))
        report("gated delta Z", projected.z, projections.map(\.z))
        report("gated delta B", projected.b, projections.map(\.b))
        report("gated delta A", projected.a, projections.map(\.a))
    }

    /// The width the benchmark died at, driven straight through the engine —
    /// and held to token equality, not merely to surviving. Eight rows is where
    /// the server process died, and it is the width the concurrency claim rests
    /// on, so "it generated something" is not the bar.
    func testEightRowsAreTokenExactAgainstSerial() async throws {
        if let library = ProcessInfo.processInfo.environment["MLXCAT_WIDTH8_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: library)
        }
        try MLXMetalRuntime.requireAvailable()
        let url = try Self.modelURL()

        let container = try await VLMModelFactory.shared.loadContainer(
            from: url, using: #huggingFaceTokenizerLoader())

        try await container.perform { context in
            let tokenCount = 24
            let parameters = GenerateParameters(maxTokens: tokenCount, temperature: 0)
            let prompts = (0 ..< 8).map { Self.prompts[$0 % Self.prompts.count] + " (\($0))" }
            var inputs: [LMInput] = []
            for prompt in prompts {
                inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }

            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 8,
                serializationPolicy: .never
            )
            let serial = try inputs.map {
                try SerialGreedyTokenHelper.tokens(
                    model: context.model, input: $0, parameters: parameters, steps: tokenCount)
            }
            if ProcessInfo.processInfo.environment["MLXCAT_WIDTH8_OP_TRACE"] == "1",
                let qwen = context.model as? Qwen35
            {
                try Self.traceFirstDecodeProjection(qwen, tokens: serial)
                try Self.traceHeldPrefix(qwen, inputs: inputs, tokens: serial, parameters: parameters)
                let input = try XCTUnwrap(inputs.first)
                let control = try await BatchRowDivergenceProbe.identicalRowSplit(
                    model: context.model, input: input, width: 8, generated: tokenCount)
                print("WIDTH8 identical rows vs singleton \(control.vsSerial)/8"
                    + " vs each other \(control.vsEachOther)/8")
                XCTAssertEqual(control.vsEachOther, 0)
            }
            let batched = try await engine.generate(
                inputs.enumerated().map { index, input in
                    Request(
                        uid: "wide-\(index)",
                        input: input,
                        maxTokens: tokenCount,
                        sampling: SamplingParameters(temperature: 0)
                    )
                }
            )

            for index in inputs.indices {
                XCTAssertGreaterThan(
                    batched["wide-\(index)"]?.count ?? 0, 1,
                    "row \(index) produced no tokens at width 8")
                XCTAssertEqual(
                    batched["wide-\(index)"], serial[index],
                    "row \(index) diverged from serial at width 8")
            }
        }
    }

    private static func traceHeldPrefix(
        _ model: Qwen35, inputs: [LMInput], tokens: [[Int]], parameters: GenerateParameters
    ) throws {
        let rowCaches = try inputs.map { _ in try model.newCache(parameters: parameters) }
        var serialStates = try zip(inputs, rowCaches).map { input, cache in
            guard case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: parameters.prefill)
            else { throw XCTSkip("The width probe requires Qwen prefill logits.") }
            return output.state
        }
        let anchors = try serialStates.map { try XCTUnwrap(BatchPositionalState.delta(in: $0)) }
        let layers = try (0..<rowCaches[0].count).map { index in
            try BatchLayerCache.merge(rowCaches.map { $0[index] }).kvCache
        }
        let attentionCache = try XCTUnwrap(layers.compactMap { $0 as? BatchKVCache }.first)
        let batchState = BatchPositionalState.stepState(
            key: anchors[0].key,
            prefillDeltas: anchors.map(\.delta),
            leftPadding: attentionCache.leftPadding.asArray(Int32.self))

        let steps = tokens.map(\.count).min() ?? 0
        var maximum: Float = 0
        var flips = 0
        var wideFlips = 0
        for step in 0..<steps {
            let ids = tokens.map { Int32($0[step]) }
            let batch = model(
                LMInput.Text(tokens: MLXArray(ids).reshaped(ids.count, 1)),
                cache: layers, state: batchState).logits
            for row in ids.indices {
                let serial = model(
                    LMInput.Text(tokens: MLXArray([ids[row]]).reshaped(1, 1)),
                    cache: rowCaches[row], state: serialStates[row])
                serialStates[row] = serial.state
                let reference = serial.logits[0, -1].asType(.float32)
                let comparison = batch[row, -1].asType(.float32)
                let error = abs(reference - comparison).max().item(Float.self)
                maximum = max(maximum, error)
                if argMax(reference).item(Int.self) != argMax(comparison).item(Int.self) {
                    let topTwo = top(reference, k: 2).asArray(Float.self)
                    let margin = abs(topTwo[0] - topTwo[1])
                    flips += 1
                    if margin > 4 * error + 0.001 {
                        wideFlips += 1
                    }
                    print("WIDTH8 flip row \(row) step \(step + 1) margin \(margin) max \(error)")
                }
            }
        }
        print("WIDTH8 held prefix max logit error \(maximum) flips \(flips) wide-margin \(wideFlips)")
    }

    /// The second place the same trap could fire: a prefix-cache HIT resumes on
    /// a WARM cache, and the scheduler used to prefill the remainder with
    /// `state: nil`. `prefixCacheEnabled` is
    /// `prefixStore != nil && !usesWindowedKVCache` and Qwen3.5 declares no
    /// sliding window, so it reads as live — but it is not, and the reason is
    /// worth pinning: Qwen3.5's `prepare` runs the whole prompt inside the model
    /// and returns `.logits`, so the scheduler never sees prompt tokens for it,
    /// never stores a prefix, and never resumes one. Chunked prefill and the
    /// prefix cache do not apply to this family at all.
    ///
    /// So this test asserts two things: that a repeated-prefix pair generates,
    /// and that the store stayed empty. The second is the tripwire — the day
    /// `prepare` stops swallowing the prefill, hits become possible and the warm
    /// resume has to already be seeded (it is: `prepareHitRow`).
    func testPrefixCacheStaysOutOfTheWayForAModelThatPrefillsItself() async throws {
        try MLXMetalRuntime.requireAvailable()
        let url = try Self.modelURL()

        let container = try await VLMModelFactory.shared.loadContainer(
            from: url, using: #huggingFaceTokenizerLoader())

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            let blockSize = 16
            let manager = PagedCacheManager(blockSize: blockSize)
            let prefixStore = BlockAwarePrefixKVStore(
                prefixCache: BlockAwarePrefixCache(
                    modelName: url.lastPathComponent, blockSize: blockSize, manager: manager))

            let shared = String(
                repeating: "Background: the KV cache stores per-layer keys and values. ", count: 8)
            let first = try await context.processor.prepare(
                input: UserInput(prompt: shared + "Question one: what is stored?"))
            let second = try await context.processor.prepare(
                input: UserInput(prompt: shared + "Question two: what is reused?"))

            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 1,
                prefixStore: prefixStore,
                serializationPolicy: .never
            )

            _ = try await engine.generate([
                Request(
                    uid: "warm", input: first, maxTokens: 4,
                    sampling: SamplingParameters(temperature: 0))
            ])
            let secondOutput = try await engine.generate([
                Request(
                    uid: "hit", input: second, maxTokens: 4,
                    sampling: SamplingParameters(temperature: 0))
            ])

            XCTAssertGreaterThan(
                secondOutput["hit"]?.count ?? 0, 0,
                "the second request produced nothing")
            XCTAssertEqual(
                prefixStore.storeCount, 0,
                """
                a prefix was stored for a model whose `prepare` returns `.logits`. \
                That means the scheduler now owns this family's prefill, prefix \
                hits are reachable, and the warm-resume anchor in `prepareHitRow` \
                needs a real test rather than this tripwire.
                """)
        }
    }
}

/// The delta arithmetic on its own — no weights, no Metal, runs everywhere.
final class BatchPositionalStateTests: XCTestCase {

    private let key = LMOutput.Key<MLXArray>("qwen35.ropeDeltas")

    private func deltas(_ state: LMOutput.State) -> [Int32] {
        (state[key] ?? MLXArray([Int32]())).asArray(Int32.self)
    }

    func testLeftPaddingIsSubtractedPerRow() {
        // The model computes `cache.offset + delta[row] + j`, and `cache.offset`
        // is the PADDED batch length. A row padded by 5 must therefore be pulled
        // back by 5 to land on its own position.
        let state = BatchPositionalState.stepState(
            key: key, prefillDeltas: [0, 0, 0], leftPadding: [0, 5, 2])
        XCTAssertEqual(deltas(state), [0, -5, -2])
    }

    func testPrefillAnchorAndPaddingCompose() {
        // An image-bearing prompt pushes its text past the vision tokens; that
        // anchor and the padding correction are independent and add.
        let state = BatchPositionalState.stepState(
            key: key, prefillDeltas: [17, -3], leftPadding: [4, 0])
        XCTAssertEqual(deltas(state), [13, -3])
    }

    func testWidthOneWithoutPaddingIsTheScalarThatShippedBefore() {
        // The degenerate case has to be unchanged, or this refactor moves the
        // single-stream path it was not supposed to touch.
        let state = BatchPositionalState.stepState(
            key: key, prefillDeltas: [11], leftPadding: [0])
        XCTAssertEqual(deltas(state), [11])
    }

    func testMissingPaddingEntriesAreTreatedAsZero() {
        // A singleton cache reports no padding at all; rows must still get a
        // delta rather than crash on an index.
        let state = BatchPositionalState.stepState(
            key: key, prefillDeltas: [7, 9], leftPadding: [])
        XCTAssertEqual(deltas(state), [7, 9])
    }

    func testOnlyTheVerifiedKeyIsRecognized() {
        // Sibling families carry a key of the same shape and consume it
        // differently. Recognizing one by accident would batch a family whose
        // decode branch cannot index per row.
        XCTAssertEqual(BatchPositionalState.recognizedKeys.map(\.id), ["qwen35.ropeDeltas"])

        var foreign = LMOutput.State()
        foreign[LMOutput.Key<MLXArray>("qwen25vl.ropeDeltas")] = MLXArray([Int32(4)])
        XCTAssertNil(BatchPositionalState.delta(in: foreign))
        XCTAssertNil(BatchPositionalState.delta(in: nil))
    }

    func testAnchorIsReadFromEitherShape() {
        var oneD = LMOutput.State()
        oneD[key] = MLXArray([Int32(6)])
        XCTAssertEqual(BatchPositionalState.delta(in: oneD)?.delta, 6)

        var scalar = LMOutput.State()
        scalar[key] = MLXArray(Int32(6))
        XCTAssertEqual(BatchPositionalState.delta(in: scalar)?.delta, 6)
    }
}
