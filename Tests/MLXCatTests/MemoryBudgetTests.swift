import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers
import XCTest

/// Long-context peak-memory budget — the regression gate pkg 102 ruled for.
///
/// The 2026-08-19 matrix measured mlxcat at ~3× legacy peak footprint at 16k
/// context (Qwen3-0.6B 11.87 vs 3.45 GiB; 27B 47.88 vs 17.77 GiB), and the
/// 2026-08-20 root cause found two mechanisms (whole-prompt logits tensor on the
/// single-pass path; non-repeating attention-scratch shapes on the chunked path).
/// A number measured once in a planning doc is not a gate, so this test loads a
/// model, runs a ~16k prefill + short decode through `MLXCatEngine`, and asserts
/// the process's peak physical footprint against a budget.
///
/// Env (all required for the test to run; it SKIPS loudly otherwise):
///   MLXCAT_MEMORY_BUDGET_MODEL          model directory (mlx-community layout)
///   MLXCAT_MEMORY_BUDGET_BYTES          peak budget in bytes (default: 12 GiB — set per model!)
///   MLXCAT_MEMORY_BUDGET_PROMPT_TOKENS  prompt length (default 16384)
///   MLXCAT_MEMORY_BUDGET_GENERATE       tokens to generate (default 32)
///
/// The budget is deliberately a parameter: the right number is per model and per
/// device, and `scripts/nightly-models.sh` sets it from the matrix. A default that
/// "always passes" would be worse than no test.
final class MemoryBudgetTests: XCTestCase {
    private struct Footprint {
        let phys: UInt64
        let lifetimeMax: UInt64

        static func read() -> Footprint? {
            var info = rusage_info_v4()
            let rc = withUnsafeMutablePointer(to: &info) { pointer -> Int32 in
                pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { raw in
                    proc_pid_rusage(getpid(), RUSAGE_INFO_V4, raw)
                }
            }
            guard rc == 0 else { return nil }
            return Footprint(phys: info.ri_phys_footprint, lifetimeMax: info.ri_lifetime_max_phys_footprint)
        }
    }

    func test_longContextPeakFootprintStaysWithinBudget() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MLXCAT_MEMORY_BUDGET_MODEL"], !modelPath.isEmpty else {
            throw XCTSkip("Set MLXCAT_MEMORY_BUDGET_MODEL (and _BYTES) to run the 16k memory budget gate.")
        }
        let budgetBytes = UInt64(environment["MLXCAT_MEMORY_BUDGET_BYTES"] ?? "") ?? (12 << 30)
        let promptTarget = Int(environment["MLXCAT_MEMORY_BUDGET_PROMPT_TOKENS"] ?? "") ?? 16_384
        let generate = Int(environment["MLXCAT_MEMORY_BUDGET_GENERATE"] ?? "") ?? 32
        let modelURL = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath)

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )
        let afterLoad = try XCTUnwrap(Footprint.read(), "proc_pid_rusage failed")

        let report = try await container.perform { context -> (promptTokens: Int, generated: Int) in
            let filler = "The river runs past the old mill and under the stone bridge, where the light breaks on the water in the early morning. "
            let fillerTokens = max(context.tokenizer.encode(text: filler).count, 1)
            let prompt = String(repeating: filler, count: max(promptTarget / fillerTokens, 1))
                + " Summarize the passage above in one sentence."
            let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
            let engine = MLXCatEngine(
                model: context.model,
                parameters: GenerateParameters(maxTokens: generate, temperature: 0),
                maxConcurrentRequests: 1
            )
            let request = Request(
                uid: "memory-budget",
                input: input,
                maxTokens: generate,
                sampling: SamplingParameters(temperature: 0)
            )
            var generated = 0
            for try await response in engine.stream(request) where response.token >= 0 {
                generated += 1
            }
            return (input.text.tokens.size, generated)
        }

        let afterRun = try XCTUnwrap(Footprint.read())
        let gib = { (bytes: UInt64) in String(format: "%.2f GiB", Double(bytes) / 1_073_741_824) }
        let line = "memory budget — \(modelURL.lastPathComponent): prompt \(report.promptTokens) tok, generated \(report.generated); "
            + "after load \(gib(afterLoad.phys)), lifetime max after run \(gib(afterRun.lifetimeMax)), budget \(gib(budgetBytes))"
        print("📏 \(line)")

        XCTAssertGreaterThan(report.generated, 0, "engine produced no tokens")
        XCTAssertGreaterThanOrEqual(
            report.promptTokens, promptTarget * 9 / 10,
            "prompt was \(report.promptTokens) tokens; the gate is meaningless below ~90% of the \(promptTarget) target"
        )
        XCTAssertLessThanOrEqual(
            afterRun.lifetimeMax, budgetBytes,
            "peak footprint \(gib(afterRun.lifetimeMax)) exceeded budget \(gib(budgetBytes)) — the long-context allocation cliff is back"
        )
    }
}
