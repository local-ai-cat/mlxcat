import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
@testable import MLXServeHTTP
@testable import MLXServeNative
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
        let wide = try XCTUnwrap(measurements[4]).tokensPerSecond

        // Asserted on batch 4, not batch 2, and with margin. Batch 2 is only ~1.5x
        // on an idle machine, which is inside this host's noise: with a build
        // running concurrently the very same code measured 0.91x at batch 2 while
        // still managing 1.88x at batch 4. Guarding on the marginal comparison
        // produces false alarms; guarding on the strong one still catches a real
        // decode regression, which would flatten every batch size at once.
        XCTAssertGreaterThan(
            wide,
            baseline * 1.5,
            """
            Batch 4 produced \(String(format: "%.1f", wide)) tok/s against \
            \(String(format: "%.1f", baseline)) tok/s at batch 1 — under the 1.5x floor. \
            Four rows share one forward pass, so this should be far above 1.0x unless \
            batched decode has genuinely regressed. Re-check on an idle host before \
            believing it: this measurement is load-sensitive.
            """
        )
    }

    /// Second bisection point. `BatchGenerator` scales (test above) but two
    /// concurrent HTTP streams do not, so the loss is somewhere between them.
    /// `MLXServeEngine` is the midpoint: it owns the scheduler, admission and the
    /// pump loop, but no HTTP, SSE or sockets.
    ///
    /// Scales here  -> the loss is in the HTTP/SSE layer.
    /// Collapses here -> the loss is in scheduler admission or the pump.
    func test_schedulerLevelConcurrencyScales() async throws {
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to measure scheduler concurrency.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        let (single, concurrent) = try await container.perform { context in
            let maxTokens = 128
            let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
            var inputs: [LMInput] = []
            for prompt in Self.prompts.prefix(3) {
                inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }

            let engine = MLXServeEngine(
                model: context.model,
                parameters: parameters,
                // Matches the shipped sidecar default, so this measures the same
                // configuration the campaign drove over HTTP.
                maxConcurrentRequests: 8
            )

            func request(_ uid: String, _ input: LMInput) -> Request {
                Request(
                    uid: uid,
                    input: input,
                    maxTokens: maxTokens,
                    sampling: SamplingParameters(temperature: 0)
                )
            }

            func count(_ stream: AsyncThrowingStream<Response, Error>) async throws -> Int {
                var tokens = 0
                for try await _ in stream { tokens += 1 }
                return tokens
            }

            _ = try await count(engine.stream(request("warmup", inputs[0])))

            var started = DispatchTime.now().uptimeNanoseconds
            let singleTokens = try await count(engine.stream(request("solo", inputs[0])))
            let singleElapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000

            // Streams are created before the concurrent consumers: Request holds
            // MLXArrays and is not Sendable, so building it inside an `async let`
            // is a sending violation. The stream of Sendable responses is fine.
            let firstStream = engine.stream(request("pair-a", inputs[1]))
            let secondStream = engine.stream(request("pair-b", inputs[2]))

            started = DispatchTime.now().uptimeNanoseconds
            async let first = count(firstStream)
            async let second = count(secondStream)
            let pairTokens = try await first + second
            let pairElapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000

            return (Double(singleTokens) / singleElapsed, Double(pairTokens) / pairElapsed)
        }

        print("""

            scheduler-level concurrency
              one request      \(String(format: "%.1f", single)) tok/s
              two in flight    \(String(format: "%.1f", concurrent)) tok/s \
            (\(String(format: "%.2f", concurrent / single))x)
            """)

        XCTAssertGreaterThan(
            concurrent,
            single,
            """
            Two requests through MLXServeEngine produced \
            \(String(format: "%.1f", concurrent)) tok/s against \
            \(String(format: "%.1f", single)) tok/s for one. The decode loop scales at \
            this batch size, so a collapse here localizes the defect to scheduler \
            admission or the pump loop.
            """
        )
    }

    /// Third bisection point. `MLXServeEngine` scales 2.11x at two in flight, but
    /// the shipped sidecar only manages 0.95x — and the socket layer is exonerated
    /// (engine-reported and client-observed per-request rates agree within 1%).
    /// The difference between those two is this layer: `PoolBackedChatBackend`
    /// over `EnginePool`, including its lease/ticket admission.
    ///
    /// Scales here  -> the remaining suspect is prefill interleaving for skewed
    ///                 arrivals, which this test does not reproduce (it starts
    ///                 every request together).
    /// Collapses here -> the lease/admission machinery owns the regression.
    func test_poolBackedBackendConcurrencyScales() async throws {
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to measure pool-backed concurrency.")
        }

        let modelID = resolution.url.lastPathComponent
        let pool = EnginePool(
            models: [
                modelID: DiscoveredModel(
                    id: modelID,
                    modelURL: resolution.url,
                    estimatedSize: 2_000_000_000
                )
            ],
            // 8 matches the shipped sidecar default, which is what the campaign drove.
            loader: NativeModelLoader(maxConcurrentRequests: 8),
            finalCeiling: 32_000_000_000
        )
        let backend = PoolBackedChatBackend(pool: pool, modelIDs: [modelID])

        func request(_ prompt: String) -> OpenAIChatRequest {
            OpenAIChatRequest(
                model: modelID,
                messages: [OpenAIChatMessage(role: "user", content: prompt)],
                maxTokens: 128,
                temperature: 0,
                stream: true
            )
        }

        func drain(_ prompt: String) async throws -> Int {
            let stream = try await backend.startChatCompletion(request(prompt))
            var tokens = 0
            for try await _ in stream.chunks { tokens += 1 }
            return tokens
        }

        _ = try await drain(Self.prompts[0])  // warm the pool + load the model

        var rates: [Int: Double] = [:]
        for width in [1, 2, 4] {
            let started = DispatchTime.now().uptimeNanoseconds
            var produced = 0
            try await withThrowingTaskGroup(of: Int.self) { group in
                for row in 0 ..< width {
                    let prompt = Self.prompts[row % Self.prompts.count]
                    group.addTask { try await drain(prompt) }
                }
                for try await tokens in group { produced += tokens }
            }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
            rates[width] = Double(produced) / elapsed
        }

        let baseline = try XCTUnwrap(rates[1])
        print("""

            pool-backed backend concurrency
              1 in flight   \(String(format: "%.1f", baseline)) tok/s
              2 in flight   \(String(format: "%.1f", rates[2] ?? 0)) tok/s \
            (\(String(format: "%.2f", (rates[2] ?? 0) / baseline))x)
              4 in flight   \(String(format: "%.1f", rates[4] ?? 0)) tok/s \
            (\(String(format: "%.2f", (rates[4] ?? 0) / baseline))x)
            """)

        // Reported, not asserted. This test exists to localize a known regression;
        // failing it would only restate what the sidecar already demonstrates.
        // The asserted guard is the decode-level one above.
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
