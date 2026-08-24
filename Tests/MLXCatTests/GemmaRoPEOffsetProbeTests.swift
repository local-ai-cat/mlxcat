import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
@testable import MLXCat
import Tokenizers
import XCTest

/// Does gemma's batched divergence come from LEFT PADDING?
///
/// `BatchInvarianceTests` reports gemma-4-12B at 1.656 against a 1.25 tolerance
/// at every width >= 2, and gemma-4-E2B at 0.688. The proposed cause is a scalar
/// RoPE offset: `Gemma4TextAttention` in **MLXVLM** rotates both queries and
/// newly-cached keys at `cache?.offset ?? 0`
/// (`Gemma4.swift:872,880,903`) — the PADDED batch length — where its MLXLLM
/// twin uses the per-row-aware `applyRotaryPosition(rope, to:, offset:
/// cache?.ropeOffset)` (`Gemma4Text.swift:313,331,362`). If that is the cause,
/// every row shorter than the longest decodes at a position inflated by its own
/// padding, and the error must VANISH when every row is the same length.
///
/// **The two factories do not agree, and that is the point of this file.**
/// `LLMModelFactory` maps both `gemma4` and `gemma4_unified` to `Gemma4Model`
/// (`LLMModelFactory.swift:38-39`), the MLXLLM class with the per-row offset.
/// `VLMModelFactory` maps them to `Gemma4`/`Gemma4Unified`
/// (`VLMModelFactory.swift:99-100`), the MLXVLM classes with the scalar. Our
/// `NativeModelLoader` routes both through the VLM factory in production, and
/// `BatchInvarianceTests` loads them through the LLM one — so the number the
/// exclusion list is built on describes a path production never selects.
///
/// That is the third time in two days a gate has been found measuring the wrong
/// path (see `docs/KNOWN-FAILURES.md` §1d and the memory-budget corrections), so
/// this probe pins the axis explicitly: production factory, ragged vs equal
/// lengths, same everything else.
///
///     MLXCAT_GEMMA_PROBE_MODEL=~/Library/Caches/models/mlx-community/gemma-4-12B-it-qat-4bit \
///       swift test --filter GemmaRoPEOffsetProbeTests
final class GemmaRoPEOffsetProbeTests: XCTestCase {

    private static let width = 4
    private static let generated = 12

    /// Shared machinery: `BatchRowDivergenceProbe` (extracted from this file
    /// once `MoEWidthNumericsProbeTests` needed the same three arm shapes).
    private static func loadContainer(_ url: URL) async throws -> ModelContainer {
        try await BatchRowDivergenceProbe.loadContainer(url)
    }

    private static func divergentRows(
        model: any LanguageModel, lengths: [Int]
    ) async throws -> (diverged: Int, firstStep: Int) {
        try await BatchRowDivergenceProbe.divergentRows(
            model: model, lengths: lengths, generated: generated)
    }

    private static func divergentRows(
        model: any LanguageModel, inputs: [LMInput]
    ) async throws -> (diverged: Int, firstStep: Int) {
        try await BatchRowDivergenceProbe.divergentRows(
            model: model, inputs: inputs, generated: generated)
    }

    private static func identicalRowSplit(
        model: any LanguageModel, input: LMInput, width: Int
    ) async throws -> (vsSerial: Int, vsEachOther: Int) {
        try await BatchRowDivergenceProbe.identicalRowSplit(
            model: model, input: input, width: width, generated: generated)
    }

    func testRaggedRowsDivergeAndEqualLengthRowsDoNot() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let path = ProcessInfo.processInfo.environment["MLXCAT_GEMMA_PROBE_MODEL"] else {
            throw XCTSkip("Set MLXCAT_GEMMA_PROBE_MODEL to a gemma-4 checkpoint to run this probe.")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)

        // The PRODUCTION factory. Loading through LLMModelFactory here would
        // reproduce the very mistake this probe exists to expose.
        let container = try await Self.loadContainer(url)

        let (control, equal, ragged) = try await container.perform { context in
            // WIDTH 1 IS THE CONTROL and it has to come first. A single row
            // through the batched path has no other row to be contaminated by,
            // so anything it shows is the batched code path's own numerics
            // against the serial reference — not batching. Without it, a
            // divergence caused by the two paths simply differing is
            // indistinguishable from rows corrupting each other, and every
            // conclusion below would be built on that confusion.
            let control = try await Self.divergentRows(model: context.model, lengths: [96])
            let equal = try await Self.divergentRows(
                model: context.model, lengths: Array(repeating: 96, count: Self.width))
            let ragged = try await Self.divergentRows(
                model: context.model, lengths: [96, 72, 48, 24])
            return (control, equal, ragged)
        }

        // Printed unconditionally: the comparison IS the result.
        print(
            "GEMMAPROBE \(url.lastPathComponent)"
                + " control(w1)=\(control.diverged)/1 (first step \(control.firstStep))"
                + " equal(w4)=\(equal.diverged)/\(Self.width) (first step \(equal.firstStep))"
                + " ragged(w4)=\(ragged.diverged)/\(Self.width) (first step \(ragged.firstStep))")

        XCTAssertEqual(
            control.diverged, 0,
            """
            width 1 diverged from serial, so this harness is measuring a PATH \
            difference and not batching — every other arm here is uninterpretable \
            until that is explained
            """)
        // Equal-length rows carry no padding, so a scalar cache offset is
        // correct for every one of them. If these still diverge, left padding is
        // not the cause and the proposed per-row RoPE fix is aimed at the wrong
        // thing.
        XCTAssertEqual(
            equal.diverged, 0,
            "equal-length rows diverged, so left padding is not the (only) cause")
    }

    /// P2: rows that are identical in every respect. If four copies of the same
    /// prompt, decoded together, disagree with EACH OTHER, then rows are
    /// crossing and the bug is ours and locatable. If they agree with each other
    /// but all differ from serial, the divergence is width-level numerics and no
    /// per-row indexing fix will touch it.
    ///
    /// Also prints whether decode pipelining was actually on, because the arm
    /// that turns it off is worthless if the switch never flipped.
    func testIdenticalRowsAgreeWithEachOther() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let path = ProcessInfo.processInfo.environment["MLXCAT_GEMMA_PROBE_MODEL"] else {
            throw XCTSkip("Set MLXCAT_GEMMA_PROBE_MODEL to a gemma-4 checkpoint to run this probe.")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let container = try await Self.loadContainer(url)

        let (w2, w4) = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(prompt: "In two sentences, explain what a KV cache stores during autoregressive decoding."))
            let w2 = try await Self.identicalRowSplit(model: context.model, input: input, width: 2)
            let w4 = try await Self.identicalRowSplit(model: context.model, input: input, width: 4)
            return (w2, w4)
        }

        print(
            "GEMMASAME \(url.lastPathComponent) pipelining="
                + (ContinuousBatchGenerator.pipeliningEnabled ? "on" : "OFF")
                + " w2 vsSerial=\(w2.vsSerial)/2 vsEachOther=\(w2.vsEachOther)/2"
                + " w4 vsSerial=\(w4.vsSerial)/4 vsEachOther=\(w4.vsEachOther)/4")

        XCTAssertEqual(
            w4.vsEachOther, 0,
            """
            four IDENTICAL prompts decoded together disagreed with each other — \
            rows are crossing, and that is a per-row indexing bug in the width >= 2 \
            path, not width-level numerics
            """)
    }

    /// The same question on REAL text, because the arms above feed synthetic
    /// token ids and greedy decoding over gibberish sits near ties constantly —
    /// a divergence there can be a property of the input rather than of the
    /// engine. This one decides whether it matters in practice.
    ///
    /// Equal lengths are made by trimming each prepared prompt to the shortest
    /// of them, so the text stays real and LENGTH is still the only variable.
    func testRealPromptsOnTheProductionFactory() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let path = ProcessInfo.processInfo.environment["MLXCAT_GEMMA_PROBE_MODEL"] else {
            throw XCTSkip("Set MLXCAT_GEMMA_PROBE_MODEL to a gemma-4 checkpoint to run this probe.")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let container = try await Self.loadContainer(url)

        let prompts = [
            "Explain, in two sentences, what a KV cache stores during autoregressive decoding and why it makes generation cheaper than re-reading the prompt every step.",
            "In one sentence, say why batching requests of different prompt lengths needs left padding.",
            "Name three primary colours, separated by commas.",
            "What is the capital of France?",
        ]

        let (control, pair, equal, ragged) = try await container.perform { context in
            var prepared: [LMInput] = []
            for prompt in prompts {
                prepared.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }
            let shortest = prepared.map { $0.text.tokens.dim(-1) }.min() ?? 1
            let trimmed = prepared.map { input -> LMInput in
                let flat = input.text.tokens.reshaped([-1])
                return LMInput(text: LMInput.Text(tokens: flat[0 ..< shortest]))
            }
            let control = try await Self.divergentRows(
                model: context.model, inputs: [prepared[0]])
            // Width 2 decides the recommendation: if the smallest batch anyone
            // can form is already inexact, no ceiling above 1 is defensible.
            let pair = try await Self.divergentRows(
                model: context.model, inputs: Array(prepared.prefix(2)))
            let equal = try await Self.divergentRows(model: context.model, inputs: trimmed)
            let ragged = try await Self.divergentRows(model: context.model, inputs: prepared)
            return (control, pair, equal, ragged)
        }

        print(
            "GEMMAPROBEREAL \(url.lastPathComponent)"
                + " control(w1)=\(control.diverged)/1"
                + " ragged(w2)=\(pair.diverged)/2"
                + " equal(w4)=\(equal.diverged)/\(Self.width)"
                + " ragged(w4)=\(ragged.diverged)/\(Self.width)")

        XCTAssertEqual(
            control.diverged, 0,
            "width 1 diverged from serial on real text — a path difference, not batching")
        XCTAssertEqual(
            ragged.diverged, 0,
            """
            gemma diverges from serial at width 4 on REAL prompts, on the factory \
            production loads. That is the width `batchDecodeWidthCeilings` currently \
            permits, and the evidence the cap was chosen from was measured through \
            LLMModelFactory — a different attention implementation
            """)
    }
}
