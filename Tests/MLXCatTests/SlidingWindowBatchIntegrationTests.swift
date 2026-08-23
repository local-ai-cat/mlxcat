import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers
import XCTest

final class SlidingWindowBatchIntegrationTests: XCTestCase {

    /// The whole point of this gate is a model with a rotating KV cache, so the
    /// engine has to be TOLD that — and it was not.
    ///
    /// `MLXCatEngine`'s default `cacheCapabilities` declares
    /// `usesWindowedKVCache: false`, so this gate was exercising a configuration
    /// that never ships: in production `NativeModelLoader` derives the real
    /// capability from the checkpoint and the scheduler makes cache-shape
    /// decisions from it (prefix-cache eligibility, and whether the last prompt
    /// token may be prefilled alone). A sliding-window gate that tells the engine
    /// it is not a sliding-window model can only test the wrong thing.
    ///
    /// Found 2026-08-23 when the last-token-alone prefill optimisation was
    /// correctly capability-gated OFF for windowed caches and this gate kept
    /// failing anyway — because it never declared itself windowed.
    static let windowedCapabilities = ModelCacheCapabilities(usesWindowedKVCache: true)

    /// Fails loudly when the configured model's sliding window is wider than the
    /// context this gate generates — the gate would otherwise pass or fail for
    /// reasons that have nothing to do with a sliding window.
    static func requireWindowIsActuallyCrossed(modelPath: String, steps: Int) throws {
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw XCTSkip("could not read config.json at \(configURL.path)")
        }
        let text = (root["text_config"] as? [String: Any]) ?? root
        guard let window = (text["sliding_window"] ?? root["sliding_window"]) as? Int else {
            throw XCTSkip("\(URL(fileURLWithPath: modelPath).lastPathComponent) declares no sliding_window")
        }
        XCTAssertLessThan(
            window, steps,
            """
            \(URL(fileURLWithPath: modelPath).lastPathComponent) has sliding_window=\(window) but             this gate only generates \(steps) tokens, so the window is never crossed and the gate             does not test what it is named for. Point MLXSERVE_SLIDING_TEST_MODEL at a model with a             smaller window (gpt-oss-20b is 128) or raise the step count above \(window).
            """
        )
    }

    func testSlidingWindowBatchMatchesSerialGreedyTokens() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_SLIDING_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_SLIDING_TEST_MODEL to a sliding-window model to run this gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let tokenCount = 4
            let parameters = GenerateParameters(maxTokens: tokenCount, temperature: 0)
            let prompts = [
                "In one sentence, define sliding-window attention:",
                "Write a concise Swift comment explaining why mixed-length prompt batches need left padding before attention masking:",
            ]
            var inputs: [LMInput] = []
            for prompt in prompts {
                inputs.append(
                    try await context.processor.prepare(input: UserInput(prompt: prompt))
                )
            }
            let serial = try inputs.map {
                try SerialGreedyTokenHelper.tokens(
                    model: context.model,
                    input: $0,
                    parameters: parameters,
                    steps: tokenCount
                )
            }
            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 2,
                cacheCapabilities: Self.windowedCapabilities
            )
            let batched = try await engine.generate(
                inputs.enumerated().map { index, input in
                    Request(
                        uid: "sliding-batch-\(index)",
                        input: input,
                        maxTokens: tokenCount,
                        sampling: SamplingParameters(temperature: 0)
                    )
                }
            )

            for index in inputs.indices {
                XCTAssertEqual(batched["sliding-batch-\(index)"], serial[index])
            }
        }
    }

    func testSlidingWindowInsertRemoveExtractMidBatch() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_SLIDING_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_SLIDING_TEST_MODEL to a sliding-window model to run this gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 2, temperature: 0)
            let generator = ContinuousBatchGenerator(
                model: context.model,
                parameters: parameters
            )
            let prompts = [
                "The kernel cache stores",
                "Explain batching in a short sentence:",
                "Name two Apple silicon GPU facts:",
            ]
            var inputs: [LMInput] = []
            for prompt in prompts {
                inputs.append(
                    try await context.processor.prepare(input: UserInput(prompt: prompt))
                )
            }

            try generator.insert(uid: "sliding-0", input: inputs[0], sampling: SamplingParameters(temperature: 0))
            try generator.insert(uid: "sliding-1", input: inputs[1], sampling: SamplingParameters(temperature: 0))
            XCTAssertEqual(generator.next().count, 2)
            try generator.insert(uid: "sliding-2", input: inputs[2], sampling: SamplingParameters(temperature: 0))
            generator.remove(uid: "sliding-1")

            XCTAssertEqual(generator.uids, ["sliding-0", "sliding-2"])
            XCTAssertNotNil(generator.extractCache(uid: "sliding-0"))
            XCTAssertNotNil(generator.extractCache(uid: "sliding-2"))
            // `next()` returns responses for whatever advanced THIS call, not
            // one per active row: a row inserted while a launched-ahead step is
            // outstanding joins the NEXT step (see `pendingEmission` in
            // BatchGenerator), so the first call after the insert carries only
            // the pre-existing row. The gate asserts what mid-batch mutation
            // must preserve — both remaining rows stay live and keep producing,
            // and the removed row emits nothing — not per-call cadence.
            var responsesByUID: [String: Int] = [:]
            for _ in 0 ..< 2 {
                for response in generator.next() {
                    responsesByUID[response.uid, default: 0] += 1
                }
            }
            XCTAssertEqual(Set(responsesByUID.keys), ["sliding-0", "sliding-2"])
        }
    }

    func testSlidingWindowBatchMatchesSerialBeyondWindow() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_SLIDING_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_SLIDING_TEST_MODEL to a sliding-window model to run this gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )

        // A check that cannot fail is not a check. This gate is named "beyond
        // window" and was written for gpt-oss-20b, whose window is 128 — but
        // scripts/nightly-models.sh wires MLXSERVE_SLIDING_TEST_MODEL to
        // gemma-4-E2B, whose window is 512. At ~175 total tokens the window never
        // engaged, so for its entire life this gate has been an ordinary
        // batched-vs-serial comparison wearing a sliding-window test's name.
        // Measured 2026-08-22: gpt-oss-20b PASSES it while genuinely crossing its
        // window, and gemma-4-E2B fails it without ever reaching one.
        try Self.requireWindowIsActuallyCrossed(modelPath: modelPath, steps: 160)

        try await container.perform { context in
            let tokenCount = 160
            let parameters = GenerateParameters(maxTokens: tokenCount, temperature: 0)
            let prompts = [
                "Continue this technical note about sliding-window attention in concise prose:",
                "Write a compact implementation diary for a Swift batch inference cache:",
            ]
            var inputs: [LMInput] = []
            for prompt in prompts {
                inputs.append(
                    try await context.processor.prepare(input: UserInput(prompt: prompt))
                )
            }
            let serial = try inputs.map {
                try SerialGreedyTokenHelper.tokens(
                    model: context.model,
                    input: $0,
                    parameters: parameters,
                    steps: tokenCount
                )
            }
            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 2,
                cacheCapabilities: Self.windowedCapabilities
            )
            let batched = try await engine.generate(
                inputs.enumerated().map { index, input in
                    Request(
                        uid: "sliding-long-\(index)",
                        input: input,
                        maxTokens: tokenCount,
                        sampling: SamplingParameters(temperature: 0)
                    )
                }
            )

            for index in inputs.indices {
                let actual = try XCTUnwrap(batched["sliding-long-\(index)"])
                // Long greedy generations can have rare isolated token flips from
                // accumulated fp non-determinism around close logits. The important
                // sliding-window failure mode is a sustained divergence after the
                // window engages, so this gate bounds total drift and rejects cascades.
                Self.assertTokenSequencesMatchWithinSlidingWindowMargin(
                    actual: actual,
                    expected: serial[index],
                    label: "sliding-long-\(index)"
                )
            }
        }
    }

    private static func assertTokenSequencesMatchWithinSlidingWindowMargin(
        actual: [Int],
        expected: [Int],
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "\(label) token count mismatch", file: file, line: line)
        let total = min(actual.count, expected.count)
        let mismatchIndices = (0 ..< total).filter { actual[$0] != expected[$0] }
        let allowedMismatches = max(2, total / 40)
        let firstMismatches = mismatchIndices.prefix(10).map(String.init).joined(separator: ",")

        XCTAssertLessThanOrEqual(
            mismatchIndices.count,
            allowedMismatches,
            "\(label) mismatch count \(mismatchIndices.count)/\(total) exceeds margin \(allowedMismatches); first mismatches: [\(firstMismatches)]",
            file: file,
            line: line
        )

        var longestRun = 0
        var currentRun = 0
        var previousIndex: Int?
        for index in mismatchIndices {
            if let previousIndex, index == previousIndex + 1 {
                currentRun += 1
            } else {
                currentRun = 1
            }
            longestRun = max(longestRun, currentRun)
            previousIndex = index
        }

        XCTAssertLessThan(
            longestRun,
            5,
            "\(label) has sustained mismatch run of \(longestRun); first mismatches: [\(firstMismatches)]",
            file: file,
            line: line
        )
    }
}
