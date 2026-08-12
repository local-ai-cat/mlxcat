import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import Tokenizers
import XCTest

/// Explicit-demand diagnostic for the single-request throughput gap.
///
/// The three-engine campaign measured the native engine at 74–92% of the legacy
/// app path for a single request, worst on DeepSeek 1.5B. The legacy path is a
/// `TokenIterator` loop; the native path is `MLXServeEngine` over
/// `ContinuousBatchGenerator`, and in the app (`NativeModelEngine`) it always
/// carries a `SessionPrefixKVStore`.
///
/// Code reading names three per-token costs the legacy loop does not pay:
///  1. `ContinuousBatchGenerator.decodeOneToken()` forces `.asArray` on the
///     tokens it just `asyncEval`-ed, so per-token CPU work serializes with the
///     GPU instead of overlapping it the way `TokenIterator.next()` does.
///  2. `Scheduler.step()` publishes prefix-cache blocks every token: a full
///     per-row KV extraction plus a `SessionPrefixKVStore.store` per generated
///     token, with cost growing in context length.
///  3. No per-step `autoreleasepool` and no periodic `Memory.clearCache()`,
///     both of which the legacy iterator performs.
///
/// This test separates those: the same model generates the same number of
/// tokens at width 1 through (a) a raw `TokenIterator` loop, (b) the engine
/// without a prefix store, and (c) the engine with the app's prefix-store
/// configuration. Variants are interleaved across rounds so host drift lands on
/// each equally, and the reported rate is decode-only (first token excluded).
///
/// Run:
///   MLXSERVE_TEST_MODEL=~/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit \
///   swift test --filter Width1ThroughputTests
final class Width1ThroughputTests: XCTestCase {
    private static let generatedTokens = 192
    private static let measuredRounds = 3

    private struct Sample {
        let decodeTokensPerSecond: Double
        let totalSeconds: Double
        let firstTokenMilliseconds: Double
    }

    func test_width1EngineThroughputAgainstRawIterator() async throws {
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to measure width-1 throughput.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        let report = try await container.perform { context in
            try await Self.measure(context: context)
        }
        print(report)
    }

    private static func measure(context: ModelContext) async throws -> String {
        // A chat-sized prompt: long enough that per-token KV snapshot cost is
        // representative, short enough that prefill stays a small fraction.
        var lastRawGenerator: ContinuousBatchGenerator?
        let prompt = Array(
            repeating: "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
            count: 12
        ).joined(separator: " ")
        let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
        let promptTokenCount = input.text.tokens.size

        let engineParameters = GenerateParameters(maxTokens: 16, temperature: 0)
        let bareEngine = MLXServeEngine(
            model: context.model,
            parameters: engineParameters,
            maxConcurrentRequests: 8
        )
        // Matches NativeModelEngine: SessionPrefixKVStore with maxBytes 0
        // (MLXSERVE_PREFIX_CACHE_MAX_BYTES is unset in the app).
        let prefixStore = SessionPrefixKVStore(maxBytes: 0)
        let storeEngine = MLXServeEngine(
            model: context.model,
            parameters: engineParameters,
            maxConcurrentRequests: 8,
            prefixStore: prefixStore
        )

        func runIterator() throws -> Sample {
            let started = DispatchTime.now().uptimeNanoseconds
            var iterator = try TokenIterator(
                input: input,
                model: context.model,
                parameters: GenerateParameters(maxTokens: generatedTokens, temperature: 0)
            )
            var count = 0
            var firstTokenAt = started
            while iterator.next() != nil {
                count += 1
                if count == 1 {
                    firstTokenAt = DispatchTime.now().uptimeNanoseconds
                }
            }
            let ended = DispatchTime.now().uptimeNanoseconds
            return sample(count: count, firstTokenAt: firstTokenAt, started: started, ended: ended)
        }

        func runRawGenerator(round: Int) throws -> Sample {
            let generator = ContinuousBatchGenerator(
                model: context.model,
                parameters: GenerateParameters(maxTokens: 16, temperature: 0)
            )
            lastRawGenerator = generator
            let started = DispatchTime.now().uptimeNanoseconds
            var count = 0
            var firstTokenAt = started
            if try generator.insert(
                uid: "raw-\(round)",
                input: input,
                sampling: SamplingParameters(temperature: 0)
            ) != nil {
                count += 1
                firstTokenAt = DispatchTime.now().uptimeNanoseconds
            }
            while count < generatedTokens {
                for _ in generator.next() {
                    count += 1
                    if count == 1 {
                        firstTokenAt = DispatchTime.now().uptimeNanoseconds
                    }
                }
            }
            let ended = DispatchTime.now().uptimeNanoseconds
            generator.remove(uid: "raw-\(round)")
            return sample(count: count, firstTokenAt: firstTokenAt, started: started, ended: ended)
        }

        func runEngine(_ engine: MLXServeEngine, uid: String) async throws -> Sample {
            let request = Request(
                uid: uid,
                input: input,
                maxTokens: generatedTokens,
                sampling: SamplingParameters(temperature: 0)
            )
            let started = DispatchTime.now().uptimeNanoseconds
            var firstTokenAt = started
            var count = 0
            for try await response in engine.stream(request) {
                guard response.token >= 0 else { continue }
                count += 1
                if count == 1 {
                    firstTokenAt = DispatchTime.now().uptimeNanoseconds
                }
            }
            let ended = DispatchTime.now().uptimeNanoseconds
            return sample(count: count, firstTokenAt: firstTokenAt, started: started, ended: ended)
        }

        // Warmup: one short run through each path so lazy kernel compilation
        // and cache growth land outside the measurement.
        _ = try runIterator()
        _ = try runRawGenerator(round: -1)
        _ = try await runEngine(bareEngine, uid: "warmup-bare")
        _ = try await runEngine(storeEngine, uid: "warmup-store")

        var iteratorSamples: [Sample] = []
        var rawSamples: [Sample] = []
        var bareSamples: [Sample] = []
        var storeSamples: [Sample] = []
        for round in 0 ..< measuredRounds {
            iteratorSamples.append(try runIterator())
            rawSamples.append(try runRawGenerator(round: round))
            bareSamples.append(try await runEngine(bareEngine, uid: "bare-\(round)"))
            storeSamples.append(try await runEngine(storeEngine, uid: "store-\(round)"))
        }

        let stats = prefixStore.stats
        var lines: [String] = [
            "",
            "width-1 throughput — prompt \(promptTokenCount) tokens, \(generatedTokens) generated, \(measuredRounds) interleaved rounds",
            String(format: "%-28@ %12@ %20@ %12@", "variant", "median t/s", "min–max t/s", "ttft"),
            row("TokenIterator (legacy)", iteratorSamples),
            row("raw ContinuousBatchGenerator", rawSamples),
            row("engine, no prefix store", bareSamples),
            row("engine, app prefix store", storeSamples),
            "prefix store: \(stats.storeCount) stores, \(stats.fetchHitCount) hits, \(stats.currentBytes) bytes in \(stats.slotCount) slots",
        ]
        if let phases = lastRawGenerator?.phaseTimingSummary {
            lines.append("raw generator " + phases)
        }
        lines += [
        ]
        let legacyRate = median(iteratorSamples.map(\.decodeTokensPerSecond))
        let bareRate = median(bareSamples.map(\.decodeTokensPerSecond))
        let storeRate = median(storeSamples.map(\.decodeTokensPerSecond))
        if legacyRate > 0 {
            lines.append(String(
                format: "engine/legacy: bare %.2fx, with store %.2fx",
                bareRate / legacyRate,
                storeRate / legacyRate
            ))
        }
        return lines.joined(separator: "\n")
    }

    private static func sample(
        count: Int,
        firstTokenAt: UInt64,
        started: UInt64,
        ended: UInt64
    ) -> Sample {
        let decodeSeconds = Double(ended - firstTokenAt) / 1_000_000_000
        let totalSeconds = Double(ended - started) / 1_000_000_000
        let decodeTokens = max(0, count - 1)
        return Sample(
            decodeTokensPerSecond: decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0,
            totalSeconds: totalSeconds,
            firstTokenMilliseconds: Double(firstTokenAt - started) / 1_000_000
        )
    }

    private static func row(_ label: String, _ samples: [Sample]) -> String {
        let rates = samples.map(\.decodeTokensPerSecond).sorted()
        let ttfts = samples.map(\.firstTokenMilliseconds).sorted()
        return String(
            format: "%-28@ %12.1f %11.1f–%.1f %10.1fms",
            label,
            median(rates),
            rates.first ?? 0,
            rates.last ?? 0,
            median(ttfts)
        )
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        return sorted[sorted.count / 2]
    }
}
