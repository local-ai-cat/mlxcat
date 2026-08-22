import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXCat

/// Narrowing probe for the gemma-4 batched-vs-serial divergence
/// (`docs/KNOWN-FAILURES.md` §1). Ruled out so far: the sliding window
/// (gpt-oss-20b passes the same gate while genuinely crossing its 128-token
/// window) and per-row RoPE offsets (`BatchRoPEOffsetTests`).
///
/// Remaining suspect: something that differs per row. Two IDENTICAL prompts
/// have identical length, so left padding is zero for both and every per-row
/// quantity collapses to the scalar case. If the batch still diverges from
/// serial there, the defect is not about ragged rows at all.
final class GemmaBatchDivergenceProbeTests: XCTestCase {

    private func model() throws -> String {
        guard let path = ProcessInfo.processInfo.environment["MLXSERVE_SLIDING_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_SLIDING_TEST_MODEL")
        }
        return path
    }

    private func divergence(prompts: [String], steps: Int) async throws -> [Int] {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: try model()),
            using: #huggingFaceTokenizerLoader()
        )
        return try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: steps, temperature: 0)
            var inputs: [LMInput] = []
            for prompt in prompts {
                inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }
            let serial = try inputs.map {
                try SerialGreedyTokenHelper.tokens(
                    model: context.model, input: $0, parameters: parameters, steps: steps)
            }
            let engine = MLXCatEngine(
                model: context.model, parameters: parameters,
                maxConcurrentRequests: prompts.count)
            let batched = try await engine.generate(
                inputs.enumerated().map { index, input in
                    Request(
                        uid: "probe-\(index)", input: input, maxTokens: steps,
                        sampling: SamplingParameters(temperature: 0))
                })
            return try inputs.indices.map { index in
                let actual = try XCTUnwrap(batched["probe-\(index)"])
                let expected = serial[index]
                return zip(actual, expected).filter { $0 != $1 }.count
                    + abs(actual.count - expected.count)
            }
        }
    }

    /// Identical prompts: no left padding, every per-row quantity is uniform.
    func testIdenticalPromptsInABatch() async throws {
        try MLXMetalRuntime.requireAvailable()
        let prompt = "Continue this technical note about attention in concise prose:"
        let mismatches = try await divergence(prompts: [prompt, prompt], steps: 64)
        print("PROBE identical: mismatches=\(mismatches)")
        XCTAssertEqual(mismatches, [0, 0], "identical rows diverged — the defect is not raggedness")
    }

    /// A batch of one: batching machinery engaged, nothing to interleave.
    func testSingleRowBatch() async throws {
        try MLXMetalRuntime.requireAvailable()
        let mismatches = try await divergence(
            prompts: ["Continue this technical note about attention in concise prose:"], steps: 64)
        print("PROBE width-1: mismatches=\(mismatches)")
        XCTAssertEqual(mismatches, [0], "a single-row batch diverged from serial")
    }

    /// Ragged rows, short horizon — is it immediate or does it accumulate?
    func testRaggedPromptsShortHorizon() async throws {
        try MLXMetalRuntime.requireAvailable()
        let mismatches = try await divergence(
            prompts: [
                "Continue this technical note about attention in concise prose:",
                "Write a compact implementation diary for a Swift batch inference cache:",
            ], steps: 24)
        print("PROBE ragged/24: mismatches=\(mismatches)")
        XCTAssertEqual(mismatches, [0, 0])
    }
}
