import MLX
import MLXLMCommon
@testable import MLXCat
import XCTest

/// Is `qwen3_moe`'s width >= 4 logit error contamination, or accumulation order?
///
/// `batchDecodeWidthCeilings["qwen3_moe"] = 2` rests on one measurement
/// (2026-08-23): maxLogitError 0.94 at width 2, 2.69 at widths 4 and 8 against
/// the 1.25 tolerance — with ZERO wide-margin tokens flipped. The 2026-08-24
/// concurrency-cliff decomposition prices that ceiling at ~2/3 of the whole c8
/// gap (the uncapped arm measures +46-48% aggregate on the coder cells), so
/// whether the error is a real defect decides the biggest number on the board.
///
/// The discriminator is the same one that settled gemma (KNOWN-FAILURES 1f):
/// IDENTICAL rows decoded together. If copies of one prompt disagree with EACH
/// OTHER, rows are crossing — per-row contamination, a locatable bug, ceiling
/// stays. If they agree with each other while differing from serial, the error
/// is width-level numerics — for an MoE, the expected suspects are expert-mix
/// accumulation order at a wider reduction — and the question becomes one of
/// adjudication (token flips vs a logit tolerance), which is Phil's call.
///
///     MLXCAT_MOE_PROBE_MODEL=~/Library/Caches/models/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
///       swift test --filter MoEWidthNumericsProbeTests
final class MoEWidthNumericsProbeTests: XCTestCase {

    private static let generated = 12
    private static let environmentKey = "MLXCAT_MOE_PROBE_MODEL"

    /// Identical rows at the widths the ceiling excludes (4 and 8), plus the
    /// permitted width 2 as a reference point.
    func testIdenticalRowsAgreeWithEachOtherAcrossWidths() async throws {
        try MLXMetalRuntime.requireAvailable()
        let url = try BatchRowDivergenceProbe.modelURL(fromEnvironment: Self.environmentKey)
        let container = try await BatchRowDivergenceProbe.loadContainer(url)

        let (w2, w4, w8) = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(
                    prompt:
                        "In two sentences, explain what a KV cache stores during autoregressive decoding."
                ))
            let w2 = try await BatchRowDivergenceProbe.identicalRowSplit(
                model: context.model, input: input, width: 2, generated: Self.generated)
            let w4 = try await BatchRowDivergenceProbe.identicalRowSplit(
                model: context.model, input: input, width: 4, generated: Self.generated)
            let w8 = try await BatchRowDivergenceProbe.identicalRowSplit(
                model: context.model, input: input, width: 8, generated: Self.generated)
            return (w2, w4, w8)
        }

        // Printed unconditionally: the comparison IS the result.
        print(
            "MOESAME \(url.lastPathComponent)"
                + " w2 vsSerial=\(w2.vsSerial)/2 vsEachOther=\(w2.vsEachOther)/2"
                + " w4 vsSerial=\(w4.vsSerial)/4 vsEachOther=\(w4.vsEachOther)/4"
                + " w8 vsSerial=\(w8.vsSerial)/8 vsEachOther=\(w8.vsEachOther)/8")

        XCTAssertEqual(
            w4.vsEachOther, 0,
            """
            identical prompts decoded together at width 4 disagreed with each \
            other — rows are crossing, which is per-row contamination, not \
            accumulation order; the width ceiling is guarding a real defect
            """)
        XCTAssertEqual(
            w8.vsEachOther, 0,
            """
            identical prompts decoded together at width 8 disagreed with each \
            other — rows are crossing, which is per-row contamination, not \
            accumulation order; the width ceiling is guarding a real defect
            """)
    }

    /// Real prompts, greedy, token-level: does anything actually FLIP at the
    /// widths the ceiling excludes? The logit sweep said no wide-margin token
    /// flipped; this asks the blunter question end to end through the engine.
    /// The control and width-2 arms are asserted (width 2 is what the ceiling
    /// already permits, so a divergence there indicts today's shipping config);
    /// widths 4 and 8 are the evidence being gathered, so they print rather
    /// than assert — the raise decision weighs them against the tolerance
    /// story, and pre-judging it here would just make discovery look like CI
    /// flake.
    func testRealPromptTokensAcrossWidths() async throws {
        try MLXMetalRuntime.requireAvailable()
        let url = try BatchRowDivergenceProbe.modelURL(fromEnvironment: Self.environmentKey)
        let container = try await BatchRowDivergenceProbe.loadContainer(url)

        let prompts = [
            "Explain, in two sentences, what a KV cache stores during autoregressive decoding and why it makes generation cheaper than re-reading the prompt every step.",
            "In one sentence, say why batching requests of different prompt lengths needs left padding.",
            "Name three primary colours, separated by commas.",
            "What is the capital of France?",
            "Write a one-line shell command that counts files in the current directory.",
            "In one sentence, what does an MoE router decide per token?",
            "Complete the sequence and nothing else: 2, 4, 8,",
            "Translate 'good morning' to German.",
        ]

        let (control, w2, w4, w8) = try await container.perform { context in
            var prepared: [LMInput] = []
            for prompt in prompts {
                prepared.append(
                    try await context.processor.prepare(input: UserInput(prompt: prompt)))
            }
            let control = try await BatchRowDivergenceProbe.divergentRows(
                model: context.model, inputs: [prepared[0]], generated: Self.generated)
            let w2 = try await BatchRowDivergenceProbe.divergentRows(
                model: context.model, inputs: Array(prepared.prefix(2)), generated: Self.generated)
            let w4 = try await BatchRowDivergenceProbe.divergentRows(
                model: context.model, inputs: Array(prepared.prefix(4)), generated: Self.generated)
            let w8 = try await BatchRowDivergenceProbe.divergentRows(
                model: context.model, inputs: prepared, generated: Self.generated)
            return (control, w2, w4, w8)
        }

        print(
            "MOEPROBEREAL \(url.lastPathComponent)"
                + " control(w1)=\(control.diverged)/1"
                + " ragged(w2)=\(w2.diverged)/2 (first step \(w2.firstStep))"
                + " ragged(w4)=\(w4.diverged)/4 (first step \(w4.firstStep))"
                + " ragged(w8)=\(w8.diverged)/8 (first step \(w8.firstStep))")

        XCTAssertEqual(
            control.diverged, 0,
            "width 1 diverged from serial — a path difference, not batching; the other arms are uninterpretable")
        XCTAssertEqual(
            w2.diverged, 0,
            """
            qwen3_moe diverges from serial at width 2 on real prompts — the width \
            the ceiling PERMITS today. If this fires, the shipping config is \
            already outside the evidence, independent of any raise decision
            """)
    }
}
