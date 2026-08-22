import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXCat

/// Does the tiered prefix KV cache actually HIT on repeated traffic?
///
/// Nobody knew. `TrackAPrefixCacheTests` proves reconstruct is *correct*, and
/// now passes — but correctness is not reuse. The first warm-vs-cold benchmark
/// in this repo's history (2026-08-23) measured the cache's end-to-end value at
/// 1.02x-1.64x, and 0.84x — actively slower — for Qwen3.5-4B at a 4k prompt.
/// Those numbers are unreadable without knowing whether a hit even occurred:
/// a 1.0x warm result means something very different if the hit rate is zero.
///
/// `SessionPrefixKVStore` keys slots by session and falls back to
/// `anonymousSlots` when a request carries no `cacheSession`, which is how every
/// OpenAI-API request arrives unless a client sends `x-cache-session`. That
/// fallback path is the one real traffic uses and the one nothing tested.
final class PrefixCacheHitRateTests: XCTestCase {

    func testRepeatedPromptHitsTheCacheWithoutASessionKey() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run the prefix-cache hit-rate gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url, using: #huggingFaceTokenizerLoader())

        try await container.perform { context in
            let store = SessionPrefixKVStore(maxBytes: 2 << 30)
            let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
            // Long enough to span whole cache blocks; a short prompt can be
            // entirely inside the trailing partial block that is never stored.
            let prompt = String(
                repeating: "The kernel cache stores reusable attention state for a shared prefix. ",
                count: 24)
            let input = try await context.processor.prepare(input: UserInput(prompt: prompt))

            let engine = MLXCatEngine(
                model: context.model, parameters: parameters,
                maxConcurrentRequests: 1, prefixStore: store)

            // First request populates; the rest must reuse it. No cacheSession,
            // deliberately — that is what real traffic looks like.
            for index in 0 ..< 3 {
                _ = try await engine.generate([
                    Request(
                        uid: "hit-\(index)", input: input, maxTokens: 8,
                        sampling: SamplingParameters(temperature: 0))
                ])
            }

            let stats = store.stats
            print(
                "PREFIXHITS fetchHits=\(stats.fetchHitCount) stores=\(stats.storeCount) "
                    + "slots=\(stats.slotCount) evictions=\(stats.evictionCount) "
                    + "bytes=\(stats.currentBytes)")
            XCTAssertGreaterThan(
                stats.storeCount, 0,
                "nothing was ever stored — the cache cannot hit what it did not keep")
            XCTAssertGreaterThan(
                stats.fetchHitCount, 0,
                """
                three identical prompts produced zero prefix-cache hits. Real OpenAI-API \
                traffic carries no cacheSession, so it uses the anonymous slot path; if that \
                path never hits, mlxcat's headline feature does nothing for ordinary traffic.
                """)
        }
    }
}
