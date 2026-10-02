import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers
import XCTest

/// `MLXCAT_PREFIX_FULL_MATCH_REUSE`: an exact repeat of a stored prompt reuses
/// all but its last token instead of re-prefilling cold.
final class FullPromptMatchReusePolicyTests: XCTestCase {
    func testOffByDefault() {
        XCTAssertFalse(Scheduler.fullPromptMatchReuseEnabled(environment: [:]))
        XCTAssertFalse(Scheduler.fullPromptMatchReuseEnabled(environment: ["MLXCAT_PREFIX_FULL_MATCH_REUSE": "0"]))
        XCTAssertFalse(Scheduler.fullPromptMatchReuseEnabled(environment: ["MLXCAT_PREFIX_FULL_MATCH_REUSE": "yes"]))
        XCTAssertTrue(Scheduler.fullPromptMatchReuseEnabled(environment: ["MLXCAT_PREFIX_FULL_MATCH_REUSE": "1"]))
        XCTAssertTrue(Scheduler.fullPromptMatchReuseEnabled(environment: ["MLXCAT_PREFIX_FULL_MATCH_REUSE": "ON"]))
    }
}

private enum FullPromptMatchProbe {
    static func tokens(count: Int) -> [Int] {
        (0 ..< count).map { 1000 + ($0 * 7 % 4096) }
    }

    static func request(_ uid: String, _ tokens: [Int]) -> Request {
        Request(
            uid: uid,
            input: LMInput(text: LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)))),
            maxTokens: 12,
            sampling: SamplingParameters(temperature: 0)
        )
    }

    /// Cold run, then the identical prompt again on the same engine and store.
    /// Returns (cold tokens, replay tokens, store fetch hits).
    static func replay(model: any LanguageModel, leverOn: Bool) async throws -> ([Int], [Int], Int) {
        if leverOn {
            setenv("MLXCAT_PREFIX_FULL_MATCH_REUSE", "1", 1)
        } else {
            unsetenv("MLXCAT_PREFIX_FULL_MATCH_REUSE")
        }
        defer { unsetenv("MLXCAT_PREFIX_FULL_MATCH_REUSE") }
        let prompt = tokens(count: 1200)
        let store = SessionPrefixKVStore()
        let engine = MLXCatEngine(
            model: model,
            parameters: GenerateParameters(maxTokens: 12, temperature: 0),
            maxConcurrentRequests: 1,
            prefixStore: store
        )
        let cold = try await engine.generate([request("cold", prompt)])["cold", default: []]
        let again = try await engine.generate([request("again", prompt)])["again", default: []]
        return (cold, again, store.stats.fetchHitCount)
    }
}

final class FullPromptMatchReuseIntegrationTests: XCTestCase {
    func testExactReplayReusesThePrefixAndMatchesTheColdRun() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run the full-match reuse check.")
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url, using: #huggingFaceTokenizerLoader())

        let (off, on, hybrid) = try await container.perform { context in
            let off = try await FullPromptMatchProbe.replay(model: context.model, leverOn: false)
            let on = try await FullPromptMatchProbe.replay(model: context.model, leverOn: true)
            let hybrid = try context.model.newCache(parameters: nil).contains { $0 is MambaCache }
            return (off, on, hybrid)
        }
        print("FULLMATCH hybrid=\(hybrid) off hits=\(off.2) on hits=\(on.2)")

        XCTAssertEqual(off.0, off.1, "default replay must equal the cold run")
        XCTAssertEqual(off.2, 1)
        // Here the stored slot runs past the prompt (it holds the generated
        // tokens too). A hybrid slot then already answers with its deepest
        // checkpoint, so the lever has nothing to do. A trimmable cache matches
        // every prompt token, takes a second, one-shorter fetch and uses it.
        // (When a hybrid slot ends exactly at the prompt, e.g. max_tokens 1, the
        // lever does engage and resumes from the checkpoint.)
        XCTAssertEqual(on.2, hybrid ? 1 : 2, "unexpected number of prefix fetches with the lever on")
        XCTAssertFalse(on.0.isEmpty)
        XCTAssertEqual(on.0, on.1, "reusing N-1 tokens changed greedy output")
        XCTAssertEqual(on.0, off.0, "the lever changed the cold run itself")
    }
}
