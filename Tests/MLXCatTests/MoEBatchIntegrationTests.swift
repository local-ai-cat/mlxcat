import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers
import XCTest

/// Batched-vs-serial greedy equality for a MoE family, driven through the real
/// `MLXCatEngine` insert/decode/extract path.
///
/// This gate tested exactly one width — 2 — for its whole life, which is the
/// width the `qwen3_moe` exclusion is least likely to be wrong at: the recorded
/// measurement behind `batchDecodeRegressedModelTypes` is a logit error of 0.94
/// at width 2 (inside the 1.25 tolerance) and **2.69 at widths 4 and 8**. A gate
/// pinned to width 2 would therefore stay green if that exclusion were lifted
/// wrongly — it could not see the defect it exists to guard. It now runs the
/// same widths the invariance gate uses, and says which one broke.
///
/// It is deliberately not the same check as `BatchInvarianceTests`: that one
/// compares logits from `evaluateBatchGate` directly, this one compares emitted
/// tokens through the serving engine, so a defect in admission, row insertion or
/// extraction shows up here and nowhere else.
final class MoEBatchIntegrationTests: XCTestCase {
    /// Eight distinct prompts so every width slices a different mix of lengths;
    /// ragged rows are what the scalar-offset defects get wrong.
    private static let prompts = [
        "Write a Swift function name for parsing JSON:",
        "Complete this sentence: A router selects experts by",
        "The capital of France is",
        "List three colors:",
        "In Swift, an actor protects",
        "What is 2 + 2?",
        "Translate hello to Spanish:",
        "Name one sorting algorithm with O(n log n) average time:",
    ]

    func testMoEBatchMatchesSerialGreedyTokens() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_MOE_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_MOE_TEST_MODEL to a MoE model to run this gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath),
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let tokenCount = 4
            let parameters = GenerateParameters(maxTokens: tokenCount, temperature: 0)
            var inputs: [LMInput] = []
            for prompt in Self.prompts {
                inputs.append(
                    try await context.processor.prepare(input: UserInput(prompt: prompt))
                )
            }
            // Serial is the reference and is width-independent, so it is computed
            // once for all eight prompts and sliced per width rather than being
            // recomputed three times on a 30B model.
            let serial = try inputs.map {
                try SerialGreedyTokenHelper.tokens(
                    model: context.model,
                    input: $0,
                    parameters: parameters,
                    steps: tokenCount
                )
            }

            var failures: [String] = []
            for width in [2, 4, 8] {
                let engine = MLXCatEngine(
                    model: context.model,
                    parameters: parameters,
                    maxConcurrentRequests: width
                )
                let batched = try await engine.generate(
                    (0 ..< width).map { index in
                        Request(
                            uid: "moe-batch-\(width)-\(index)",
                            input: inputs[index],
                            maxTokens: tokenCount,
                            sampling: SamplingParameters(temperature: 0)
                        )
                    }
                )
                for index in 0 ..< width {
                    let got = batched["moe-batch-\(width)-\(index)"]
                    if got != serial[index] {
                        failures.append(
                            "width \(width) row \(index) ('\(Self.prompts[index])'): "
                                + "serial \(serial[index]) vs batched \(got.map(String.init(describing:)) ?? "nil")"
                        )
                    }
                }
            }

            XCTAssertTrue(
                failures.isEmpty,
                "batched decode diverged from serial:\n" + failures.joined(separator: "\n")
            )
        }
    }
}
