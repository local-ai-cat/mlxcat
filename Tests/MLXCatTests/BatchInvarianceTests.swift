import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers
import XCTest

private struct DecodeTraceStep {
    let tokenId: Int
    let logits: MLXArray
}

private struct BatchGateResult: Sendable {
    let batchSize: Int
    let maxLogitError: Float
    let maxErrorStep: Int
    let maxErrorRow: Int
    let maxErrorPrompt: String
    let maxErrorMargin: Float
    let maxErrorSerialToken: Int
    let maxErrorBatchToken: Int
    let checkedTokenCount: Int
    let mismatchedCheckedTokens: Int
}

private struct DynamicBatchGateResult: Sendable {
    let checkedTokenCount: Int
    let mismatchedCheckedTokens: Int
    let responseCount: Int
    let insertionCount: Int
    let removalCount: Int
}

final class BatchInvarianceTests: XCTestCase {
    private static let prompts = [
        "The capital of France is",
        "Write a haiku about GPU kernels.",
        "In Swift, an actor protects",
        "List three colors:",
        "What is 2 + 2?",
        "Translate hello to Spanish:",
        "The largest planet is",
        "Complete: once upon a",
    ]

    func testStaticBatchedDecodeMatchesSerialWithinMarginGate() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M1 batch-invariance.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M1 batch-invariance fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let prompts = Self.prompts
        let results = try await container.perform { context in
            try await Self.evaluateBatchGate(context: context, prompts: prompts)
        }

        for result in results {
            if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_BATCH_GATE"] == "1" {
                print(
                    "M1 batch \(result.batchSize): maxLogitError=\(result.maxLogitError), checkedTokens=\(result.checkedTokenCount), mismatches=\(result.mismatchedCheckedTokens), maxErrorStep=\(result.maxErrorStep), maxErrorRow=\(result.maxErrorRow)"
                )
            }
            XCTAssertLessThan(
                result.maxLogitError,
                1.25,
                "batch size \(result.batchSize) exceeded logit tolerance at step \(result.maxErrorStep), row \(result.maxErrorRow), prompt '\(result.maxErrorPrompt)', margin \(result.maxErrorMargin), serial token \(result.maxErrorSerialToken), batch token \(result.maxErrorBatchToken)"
            )
            XCTAssertGreaterThan(
                result.checkedTokenCount,
                0,
                "batch size \(result.batchSize) had no wide-margin token checks"
            )
            XCTAssertEqual(
                result.mismatchedCheckedTokens,
                0,
                "batch size \(result.batchSize) mismatched wide-margin tokens"
            )
        }
    }

    /// The same logit-level check, across model FAMILIES.
    ///
    /// `testStaticBatchedDecodeMatchesSerialWithinMarginGate` is pinned to
    /// Qwen3-0.6B-4bit — and `NativeModelEngine.usesSerializedDecode` disables
    /// batched decode for `qwen3_5`, `qwen3_vl`, `gemma4`, `gemma4_unified`,
    /// `mistral3`, `qwen2`, `qwen3` and `qwen3_moe`. The two sets do not
    /// intersect: the gate that decides whether batching is numerically sound has
    /// never been run against a single family that batching is switched off for.
    /// Those exclusions have therefore never been re-earned, and on the 2026-08-22
    /// leaderboard they cost five of six benchmark models their concurrency —
    /// mlxcat's TTFT scales 15-86x from c1 to c8 where every competitor scales
    /// 4-9x (`docs/COMPETITIVE.md`).
    ///
    /// Nothing in the fixture is model-specific: the prompts are generic and the
    /// evaluation compares logits and only asserts token equality where the
    /// top-1/top-2 margin is wide. So the pin is lifted here as an opt-in, leaving
    /// the pinned gate's meaning untouched.
    ///
    ///     MLXCAT_BATCH_INVARIANCE_MODELS=$R/gemma-4-E2B-it-qat-4bit,$R/Qwen3.5-4B-MLX-4bit     ///       swift test --filter testBatchInvarianceAcrossModelFamilies
    func testBatchInvarianceAcrossModelFamilies() async throws {
        try MLXMetalRuntime.requireAvailable()

        let raw = ProcessInfo.processInfo.environment["MLXCAT_BATCH_INVARIANCE_MODELS"] ?? ""
        let paths = raw.split(separator: ",").map(String.init).filter { !$0.isEmpty }
        guard !paths.isEmpty else {
            throw XCTSkip(
                "Set MLXCAT_BATCH_INVARIANCE_MODELS to a comma-separated list of model directories."
            )
        }

        var failures: [String] = []
        for path in paths {
            let url = URL(fileURLWithPath: path)
            let name = url.lastPathComponent
            let container = try await LLMModelFactory.shared.loadContainer(
                from: url, using: #huggingFaceTokenizerLoader())
            let prompts = Self.prompts
            let results = try await container.perform { context in
                try await Self.evaluateBatchGate(context: context, prompts: prompts)
            }
            for result in results {
                let verdict =
                    result.maxLogitError < 1.25 && result.checkedTokenCount > 0
                    && result.mismatchedCheckedTokens == 0 ? "OK  " : "FAIL"
                // Printed unconditionally: the point of this test is the table.
                print(
                    "FAMILY \(verdict) \(name) batch=\(result.batchSize) "
                        + "maxLogitError=\(result.maxLogitError) "
                        + "checked=\(result.checkedTokenCount) "
                        + "mismatched=\(result.mismatchedCheckedTokens) "
                        + "step=\(result.maxErrorStep) margin=\(result.maxErrorMargin)"
                )
                if verdict == "FAIL" {
                    failures.append(
                        "\(name) batch=\(result.batchSize): logitError=\(result.maxLogitError), "
                            + "checked=\(result.checkedTokenCount), "
                            + "mismatched=\(result.mismatchedCheckedTokens)"
                    )
                }
            }
        }
        XCTAssertTrue(failures.isEmpty, "families failing batch invariance:\n" + failures.joined(separator: "\n"))
    }

    func testContinuousBatchingSupportsInsertAndRemoveMidBatch() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M1.5 dynamic batch-invariance.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M1.5 dynamic batch-invariance fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let prompts = Array(Self.prompts.prefix(4))
        let result = try await container.perform { context in
            try await Self.evaluateDynamicBatchGate(context: context, prompts: prompts)
        }

        if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_BATCH_GATE"] == "1" {
            print(
                "M1.5 dynamic: responses=\(result.responseCount), inserts=\(result.insertionCount), removals=\(result.removalCount), checkedTokens=\(result.checkedTokenCount), mismatches=\(result.mismatchedCheckedTokens)"
            )
        }
        XCTAssertEqual(result.insertionCount, 4)
        XCTAssertEqual(result.removalCount, 2)
        XCTAssertGreaterThan(result.responseCount, 0)
        XCTAssertGreaterThan(result.checkedTokenCount, 0)
        XCTAssertEqual(result.mismatchedCheckedTokens, 0)
    }

    private static func evaluateBatchGate(
        context: ModelContext,
        prompts: [String]
    ) async throws -> [BatchGateResult] {
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        var results: [BatchGateResult] = []

        // Width 1 is the CONTROL, not a batch size anyone ships. One row through
        // StaticBatchGenerator has nothing to be contaminated by, so whatever
        // error it shows is the batched code path's own numerics against the
        // serial reference — not batching. Without it a family whose two paths
        // simply differ is indistinguishable from a family whose rows corrupt
        // each other, and the gate reads the first as the second.
        for batchSize in [1, 2, 4, 8] {
            var inputs: [LMInput] = []
            for prompt in prompts.prefix(batchSize) {
                inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }

            let serial = try inputs.map {
                try Self.serialTrace(
                    model: context.model,
                    input: $0,
                    parameters: parameters,
                    steps: parameters.maxTokens ?? 4
                )
            }
            let batchGenerator = try StaticBatchGenerator(
                model: context.model,
                inputs: inputs,
                parameters: parameters
            )

            var maxLogitError: Float = 0
            var maxErrorStep = -1
            var maxErrorRow = -1
            var maxErrorPrompt = ""
            var maxErrorMargin: Float = 0
            var maxErrorSerialToken = -1
            var maxErrorBatchToken = -1
            var checkedTokenCount = 0
            var mismatchedCheckedTokens = 0

            for stepIndex in 0 ..< (parameters.maxTokens ?? 4) {
                let batchStep = batchGenerator.next()
                if stepIndex == 0 {
                    continue
                }
                for row in 0 ..< batchSize {
                    let serialStep = serial[row][stepIndex]
                    let batchLogits = batchStep.logits[row, 0...]
                    let logitError = Self.maxAbsoluteDifference(batchLogits, serialStep.logits)
                    let margin = Self.topOneTopTwoMargin(serialStep.logits)
                    if logitError > maxLogitError {
                        maxLogitError = logitError
                        maxErrorStep = stepIndex
                        maxErrorRow = row
                        maxErrorPrompt = prompts[row]
                        maxErrorMargin = margin
                        maxErrorSerialToken = serialStep.tokenId
                        maxErrorBatchToken = batchStep.tokenIds[row]
                    }

                    if margin > logitError * 4 + 1e-3 {
                        checkedTokenCount += 1
                        if batchStep.tokenIds[row] != serialStep.tokenId {
                            mismatchedCheckedTokens += 1
                        }
                    }
                }
            }

            results.append(
                BatchGateResult(
                    batchSize: batchSize,
                    maxLogitError: maxLogitError,
                    maxErrorStep: maxErrorStep,
                    maxErrorRow: maxErrorRow,
                    maxErrorPrompt: maxErrorPrompt,
                    maxErrorMargin: maxErrorMargin,
                    maxErrorSerialToken: maxErrorSerialToken,
                    maxErrorBatchToken: maxErrorBatchToken,
                    checkedTokenCount: checkedTokenCount,
                    mismatchedCheckedTokens: mismatchedCheckedTokens
                )
            )
        }

        return results
    }

    private static func evaluateDynamicBatchGate(
        context: ModelContext,
        prompts: [String]
    ) async throws -> DynamicBatchGateResult {
        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)
        var inputs: [LMInput] = []
        for prompt in prompts {
            inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
        }
        let serial = try inputs.map {
            try Self.serialTrace(
                model: context.model,
                input: $0,
                parameters: parameters,
                steps: parameters.maxTokens ?? 5
            )
        }

        let generator = ContinuousBatchGenerator(model: context.model, parameters: parameters)
        var expectedStepByUID: [String: Int] = [:]
        var checkedTokenCount = 0
        var mismatchedCheckedTokens = 0
        var responseCount = 0
        var insertionCount = 0
        var removalCount = 0

        func insert(_ row: Int) throws {
            let uid = "row-\(row)"
            try generator.insert(
                uid: uid,
                input: inputs[row],
                sampling: SamplingParameters(temperature: 0)
            )
            expectedStepByUID[uid] = 0
            insertionCount += 1
        }

        func consumeStep() {
            for response in generator.next() {
                responseCount += 1
                guard let promptRow = Int(response.uid.replacingOccurrences(of: "row-", with: "")),
                    let expectedStep = expectedStepByUID[response.uid]
                else {
                    mismatchedCheckedTokens += 1
                    continue
                }

                let serialStep = serial[promptRow][expectedStep]
                let margin = Self.topOneTopTwoMargin(serialStep.logits)
                if margin > 1.25 * 4 + 1e-3 {
                    checkedTokenCount += 1
                    if response.token != serialStep.tokenId {
                        mismatchedCheckedTokens += 1
                    }
                }
                expectedStepByUID[response.uid] = expectedStep + 1
            }
        }

        try insert(0)
        try insert(1)
        consumeStep()

        try insert(2)
        consumeStep()

        Stream.gpu.synchronize()
        generator.remove(uid: "row-0")
        expectedStepByUID.removeValue(forKey: "row-0")
        removalCount += 1
        consumeStep()

        try insert(3)
        consumeStep()

        Stream.gpu.synchronize()
        generator.remove(uid: "row-2")
        expectedStepByUID.removeValue(forKey: "row-2")
        removalCount += 1
        consumeStep()

        return DynamicBatchGateResult(
            checkedTokenCount: checkedTokenCount,
            mismatchedCheckedTokens: mismatchedCheckedTokens,
            responseCount: responseCount,
            insertionCount: insertionCount,
            removalCount: removalCount
        )
    }

    private static func serialTrace(
        model: any LanguageModel,
        input: LMInput,
        parameters: GenerateParameters,
        steps: Int
    ) throws -> [DecodeTraceStep] {
        let cache = try model.newCache(parameters: parameters)
        var state: LMOutput.State?

        var currentLogits: MLXArray
        var currentToken: MLXArray

        switch try model.prepare(
            input, cache: cache, state: state, prefill: parameters.prefill) {
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

        var trace: [DecodeTraceStep] = []
        for _ in 0 ..< steps {
            eval(currentToken, currentLogits)
            trace.append(
                DecodeTraceStep(
                    tokenId: currentToken.item(Int.self),
                    logits: currentLogits
                )
            )

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
        return trace
    }

    private static func maxAbsoluteDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let difference = abs(lhs.asType(.float32) - rhs.asType(.float32))
        return difference.max().item(Float.self)
    }

    private static func topOneTopTwoMargin(_ logits: MLXArray) -> Float {
        let topValues = top(logits.asType(.float32), k: 2, axis: -1).asArray(Float.self)
        let sorted = topValues.sorted(by: >)
        return sorted[0] - sorted[1]
    }
}
