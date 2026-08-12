import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import Tokenizers
import XCTest

/// Explicit-demand diagnostic for the package 54 batch-decode regression.
///
/// The campaign measured *aggregate two-in-flight throughput over HTTP* and
/// concluded that batched decode is slower than serial decode for `qwen2`,
/// `qwen3`, and `qwen3_moe` — which is why pin `201ddaee` serializes those
/// families. That measurement cannot distinguish an engine problem from a
/// transport problem, and the two smallest models in the sweep collapsed to
/// within 0.1% of each other (164.2 and 164.4 tok/s) despite a 10x size
/// difference, which does not look like model compute.
///
/// This test measures the decode loop alone: no HTTP, no SSE, no scheduler
/// admission — just `ContinuousBatchGenerator.next()` at batch 1 versus batch N.
/// If batching scales here, the regression is above the engine and serializing
/// model families is treating the wrong layer.
///
/// Run:
///   MLXSERVE_TEST_MODEL=~/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit \
///   swift test --filter BatchDecodeThroughputTests
final class BatchDecodeThroughputTests: XCTestCase {
    private static let prompts = [
        "The capital of France is",
        "Write a haiku about GPU kernels.",
        "In Swift, an actor protects",
        "List three colors:",
        "Explain unified memory in one sentence.",
        "A tokenizer converts",
        "The fastest sorting algorithm is",
        "Metal shaders run on"
    ]

    /// Decode steps timed per batch size. Deliberately modest: this is a rate
    /// measurement, not a soak.
    private static let measuredSteps = 64
    private static let warmupSteps = 8

    func test_batchedDecodeThroughputScalesWithBatchSize() async throws {
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to measure batch decode throughput.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        let batchSizes = [1, 2, 4]
        let measurements = try await container.perform { context in
            try await Self.measure(context: context, batchSizes: batchSizes)
        }

        var lines: [String] = [
            "",
            "batch-decode throughput — \(resolution.url.lastPathComponent)",
            // %s would take a C string; Swift Strings must go through %@.
            String(format: "%6@ %14@ %14@ %10@", "batch", "steps/s", "tokens/s", "vs b=1")
        ]
        let single = measurements[1]?.tokensPerSecond ?? 0
        for size in batchSizes {
            guard let measurement = measurements[size] else { continue }
            let ratio = single > 0 ? measurement.tokensPerSecond / single : 0
            lines.append(String(
                format: "%6d %14.1f %14.1f %9.2fx",
                size,
                measurement.stepsPerSecond,
                measurement.tokensPerSecond,
                ratio
            ))
        }
        print(lines.joined(separator: "\n"))

        let baseline = try XCTUnwrap(measurements[1]).tokensPerSecond
        let batched = try XCTUnwrap(measurements[2]).tokensPerSecond
        XCTAssertGreaterThan(
            batched,
            baseline,
            """
            Batch 2 produced \(String(format: "%.1f", batched)) tok/s against \
            \(String(format: "%.1f", baseline)) tok/s at batch 1. Two rows share one \
            forward pass, so aggregate throughput must exceed a single row unless the \
            decode path itself is the bottleneck.
            """
        )
    }

    private struct Measurement {
        let stepsPerSecond: Double
        let tokensPerSecond: Double
    }

    private static func measure(
        context: ModelContext,
        batchSizes: [Int]
    ) async throws -> [Int: Measurement] {
        var measurements: [Int: Measurement] = [:]
        for batchSize in batchSizes {
            // Prepared inputs are re-created per batch size on purpose. Handing the
            // same LMInput (and therefore the same MLXArrays) to a second generator
            // after the first one has torn down its rows segfaults.
            var inputs: [LMInput] = []
            for prompt in prompts.prefix(batchSize) {
                inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }

            let parameters = GenerateParameters(maxTokens: measuredSteps + warmupSteps, temperature: 0)
            let generator = ContinuousBatchGenerator(model: context.model, parameters: parameters)
            for row in 0 ..< batchSize {
                try generator.insert(
                    uid: "row-\(row)",
                    input: inputs[row],
                    sampling: SamplingParameters(temperature: 0)
                )
            }

            for _ in 0 ..< warmupSteps {
                _ = generator.next()
            }
            Stream.gpu.synchronize()

            let started = DispatchTime.now().uptimeNanoseconds
            var producedTokens = 0
            for _ in 0 ..< measuredSteps {
                producedTokens += generator.next().count
            }
            Stream.gpu.synchronize()
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000

            Stream.gpu.synchronize()
            for row in 0 ..< batchSize {
                generator.remove(uid: "row-\(row)")
            }
            Stream.gpu.synchronize()

            measurements[batchSize] = Measurement(
                stepsPerSecond: Double(measuredSteps) / elapsed,
                tokensPerSecond: Double(producedTokens) / elapsed
            )
        }
        return measurements
    }
}
