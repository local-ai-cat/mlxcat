import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
@testable import MLXCat
import Tokenizers
import XCTest

/// When the MLX buffer cache goes back to the OS.
///
/// The 2026-08-23 board made this concrete: gemma-4-E2B's 16k tier peaked at
/// 8.39 GiB and every later cell in the same process reported ~7.8 GiB, because
/// MLX keeps freed Metal buffers on a free list and nothing handed them back.
/// The benchmark noticed it as a peak "regression" on cells that had not
/// regressed; a user notices it as a server sitting on gigabytes for nobody.
final class CacheReleasePolicyTests: XCTestCase {

    func testDefaultsMatchTheReferenceInterval() {
        // 512 is mlx-lm's (`guest/mlx-lm/mlx_lm/generate.py:1779`). The idle
        // release is ours — mlx-lm's server does not hold one process across
        // unrelated workloads the way a long-lived local server does.
        let policy = Scheduler.CacheReleasePolicy.default
        XCTAssertEqual(policy.decodeStepInterval, 512)
        XCTAssertTrue(policy.releasesWhenIdle)
    }

    func testEnvironmentOverridesEachKnobIndependently() {
        var policy = Scheduler.CacheReleasePolicy.fromEnvironment([
            "MLXCAT_DECODE_CLEAR_CACHE_STEPS": "64"
        ])
        XCTAssertEqual(policy.decodeStepInterval, 64)
        XCTAssertTrue(policy.releasesWhenIdle, "the decode knob must not move the idle knob")

        policy = Scheduler.CacheReleasePolicy.fromEnvironment(["MLXCAT_IDLE_CLEAR_CACHE": "0"])
        XCTAssertFalse(policy.releasesWhenIdle)
        XCTAssertEqual(policy.decodeStepInterval, 512, "the idle knob must not move the interval")
    }

    func testZeroDisablesThePeriodicClearAndNegativesAreRejected() {
        XCTAssertEqual(
            Scheduler.CacheReleasePolicy.fromEnvironment(
                ["MLXCAT_DECODE_CLEAR_CACHE_STEPS": "0"]
            ).decodeStepInterval, 0)
        // A negative interval would clear on every step; it is ignored rather
        // than clamped to zero, so a typo cannot silently disable the feature.
        XCTAssertEqual(
            Scheduler.CacheReleasePolicy.fromEnvironment(
                ["MLXCAT_DECODE_CLEAR_CACHE_STEPS": "-1"]
            ).decodeStepInterval, 512)
        XCTAssertEqual(
            Scheduler.CacheReleasePolicy.fromEnvironment(
                ["MLXCAT_DECODE_CLEAR_CACHE_STEPS": "not-a-number"]
            ).decodeStepInterval, 512)
    }

    func testEmptyEnvironmentIsTheDefault() {
        XCTAssertEqual(Scheduler.CacheReleasePolicy.fromEnvironment([:]), .default)
        XCTAssertEqual(
            Scheduler.CacheReleasePolicy.never,
            Scheduler.CacheReleasePolicy(decodeStepInterval: 0, releasesWhenIdle: false))
    }
}

/// Helpers live outside the test case: the Swift 6 isolation checker rejects
/// capturing a non-Sendable `XCTestCase` inside `container.perform`'s
/// `@Sendable` closure.
enum CacheReleaseProbe {
    static func promptTokens(count: Int) -> MLXArray {
        // Arbitrary in-vocabulary ids; the point is the SHAPE of the prefill, so
        // the content does not matter and a fixed pattern keeps the run
        // reproducible.
        MLXArray((0 ..< count).map { Int32(1000 + ($0 % 4096)) })
    }

    static func cacheBytesAfterRun(policy: Scheduler.CacheReleasePolicy, model: any LanguageModel)
        async throws -> Int
    {
        Memory.clearCache()
        let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
        let engine = MLXCatEngine(
            model: model,
            parameters: parameters,
            maxConcurrentRequests: 1,
            cacheReleasePolicy: policy
        )
        _ = try await engine.generate([
            Request(
                uid: "release",
                input: LMInput(text: LMInput.Text(tokens: promptTokens(count: 4096))),
                maxTokens: 8,
                sampling: SamplingParameters(temperature: 0)
            )
        ])
        return Memory.cacheMemory
    }
}

/// The A/B the policy exists for: same work, both arms, and the cache is
/// actually measured rather than assumed.
final class CacheReleaseIntegrationTests: XCTestCase {

    func testDrainingToIdleHandsTheBufferCacheBack() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run the cache-release A/B.")
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url, using: #huggingFaceTokenizerLoader())

        let (held, released) = try await container.perform { context in
            let held = try await CacheReleaseProbe.cacheBytesAfterRun(
                policy: .never, model: context.model)
            let released = try await CacheReleaseProbe.cacheBytesAfterRun(
                policy: .default, model: context.model)
            return (held, released)
        }

        // Printed unconditionally: the number is the result.
        print(
            "CACHERELEASE held=\(held / 1_048_576) MiB released=\(released / 1_048_576) MiB")

        // The control arm has to actually hold something, or the test proves
        // nothing about the treatment arm.
        XCTAssertGreaterThan(
            held, 16 * 1_048_576,
            "the `.never` arm freed the cache anyway — this A/B has no control")
        XCTAssertLessThan(
            released, held / 2,
            "draining to idle did not hand the buffer cache back")
    }
}
