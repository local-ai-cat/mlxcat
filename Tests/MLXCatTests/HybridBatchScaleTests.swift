import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXCat

/// Batched decode over a hybrid cache at the shape that actually breaks.
///
/// `HybridBatchIntegrationTests` passes (2/2) at three rows and 2-8 generated
/// tokens. The benchmark, at four rows and 1024 generated tokens, does not:
/// width 4 returns a completion with no `usage` frame at all, and width 8 kills
/// the server process outright — silently, with nothing in its log and without
/// tripping the runaway guard, so it is a hard crash rather than a memory kill.
///
/// A green integration suite and a dead server is exactly the gap that "verify
/// the artifact, not just the diff" exists for. This gate reproduces the bench's
/// shape so the defect is fixable from a test instead of a benchmark run.
///
/// `qwen3_5` covers two of the six benchmark models (Qwen3.5-4B and
/// Qwen3.8-27B), both currently serialized, at 37 s TTFT under load on an M4 Pro.
/// It is the largest single win left on the board.
final class HybridBatchScaleTests: XCTestCase {

    func testHybridBatchAtBenchmarkScale() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_HYBRID_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_HYBRID_TEST_MODEL to run the hybrid batch scale gate.")
        }

        let rows = Int(ProcessInfo.processInfo.environment["MLXCAT_HYBRID_SCALE_ROWS"] ?? "") ?? 4
        let generate = Int(ProcessInfo.processInfo.environment["MLXCAT_HYBRID_SCALE_TOKENS"] ?? "") ?? 96
        // The bench's longgen tier uses a ~1024-token prompt; the small gates use
        // a handful. Prompt length is what makes chunked prefill engage and what
        // decides how many cache blocks a row spans.
        let fillerRepeats = Int(ProcessInfo.processInfo.environment["MLXCAT_HYBRID_SCALE_FILLER"] ?? "") ?? 12

        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader())

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: generate, temperature: 0)
            // A prompt long enough to cross cache-block boundaries, which is what
            // the bench's 1k-token workload does and the small gates never do.
            let filler = String(
                repeating: "The scheduler admits a request, prefills its prompt, and merges the "
                    + "resulting cache into the running batch. ", count: fillerRepeats)
            var inputs: [LMInput] = []
            for index in 0 ..< rows {
                inputs.append(
                    try await context.processor.prepare(
                        input: UserInput(prompt: "\(filler) Question \(index): summarise the above.")))
            }

            // WITH a prefix store, because the server always has one
            // (NativeModelEngine builds a SessionPrefixKVStore unconditionally)
            // and the in-process engine defaults to none. Batched decode over a
            // hybrid cache at 8 rows x 512 tokens is clean WITHOUT it, so if the
            // bench dies and this does not, the prefix-cache interaction is what
            // differs — and cache extract/merge across a hybrid layout is exactly
            // where the type-preservation bug lived.
            let store = SessionPrefixKVStore(maxBytes: 8 << 30)
            let engine = MLXCatEngine(
                model: context.model, parameters: parameters,
                maxConcurrentRequests: rows, prefixStore: store,
                serializationPolicy: .never)
            let requests = inputs.enumerated().map { index, input in
                Request(
                    uid: "scale-\(index)", input: input, maxTokens: generate,
                    sampling: SamplingParameters(temperature: 0))
            }
            let out = try await engine.generate(requests)

            var empty: [String] = []
            for request in requests {
                let tokens = out[request.uid] ?? []
                print("HYBRIDSCALE \(request.uid) tokens=\(tokens.count)")
                if tokens.isEmpty { empty.append(request.uid) }
            }
            let stats = store.stats
            print("HYBRIDSCALE prefix fetchHits=\(stats.fetchHitCount) stores=\(stats.storeCount)")
            XCTAssertTrue(
                empty.isEmpty,
                "rows produced nothing under batched decode over a hybrid cache: \(empty)")
        }
    }
}
