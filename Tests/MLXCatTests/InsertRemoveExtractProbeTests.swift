import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXCat

/// Pins `next()`'s contract at the exact spot the two mid-batch gates used to
/// misread it (resolved 2026-08-23): `next()` returns responses for whatever
/// advanced THIS call, not one per active row per call. A row inserted while a
/// launched-ahead step is outstanding has nothing pending (`pendingEmission`
/// starts false) and joins the NEXT step, so the first call after such an
/// insert carries only the pre-existing rows — and the following call carries
/// everyone. `Scheduler.step()` consumes responses per-uid and never assumes
/// per-row coverage; assuming it in a test was the bug.
///
/// The probe that established the resolution also showed the one-call offset is
/// independent of any token limit (same shape at maxTokens 2 and 8), so both
/// budgets are kept.
final class InsertRemoveExtractProbeTests: XCTestCase {

    private func assertContractAfterInsertRemoveExtract(maxTokens: Int) async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_SLIDING_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_SLIDING_TEST_MODEL")
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: modelPath), using: #huggingFaceTokenizerLoader())
        try await container.perform { context in
            let generator = ContinuousBatchGenerator(
                model: context.model,
                parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
            var inputs: [LMInput] = []
            for prompt in ["The kernel cache stores", "Explain batching briefly:", "Name two GPU facts:"] {
                inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }
            try generator.insert(uid: "p0", input: inputs[0], sampling: SamplingParameters(temperature: 0))
            try generator.insert(uid: "p1", input: inputs[1], sampling: SamplingParameters(temperature: 0))
            XCTAssertEqual(generator.next().count, 2)
            try generator.insert(uid: "p2", input: inputs[2], sampling: SamplingParameters(temperature: 0))
            generator.remove(uid: "p1")
            XCTAssertEqual(generator.uids, ["p0", "p2"])
            XCTAssertNotNil(generator.extractCache(uid: "p0"))
            XCTAssertNotNil(generator.extractCache(uid: "p2"))

            // Call 1: only p0's launched-ahead token is pending; p2 was
            // admitted mid-flight and joins the step launched by this call.
            let first = generator.next()
            XCTAssertEqual(first.map(\.uid), ["p0"])

            // Call 2: the full batch advanced together.
            let second = generator.next()
            XCTAssertEqual(Set(second.map(\.uid)), ["p0", "p2"])
        }
    }

    /// The old failing fixtures' own budget: p0 reaches its limit right here.
    func testAtTheTokenLimit() async throws {
        try MLXMetalRuntime.requireAvailable()
        try await assertContractAfterInsertRemoveExtract(maxTokens: 2)
    }

    /// Well clear of the limit: the one-call offset is contract, not boundary.
    func testWellInsideTheTokenLimit() async throws {
        try MLXMetalRuntime.requireAvailable()
        try await assertContractAfterInsertRemoveExtract(maxTokens: 8)
    }
}
