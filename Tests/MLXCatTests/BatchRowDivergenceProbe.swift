import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
@testable import MLXCat
import Tokenizers
import XCTest

/// Shared machinery for the batched-row divergence probes
/// (`GemmaRoPEOffsetProbeTests`, `MoEWidthNumericsProbeTests`).
///
/// The method, proven on the gemma RoPE pair: decode the same inputs serially
/// and batched at temperature 0 and count rows whose tokens differ. Three arm
/// shapes ask three different questions —
///
///   * width 1 through the batched path is the CONTROL: it has no other row to
///     be contaminated by, so any divergence is the batched code path's own
///     numerics vs the serial reference, and every other arm is uninterpretable
///     until it reads 0;
///   * IDENTICAL rows decoded together split the verdict: rows disagreeing with
///     EACH OTHER is per-row indexing/contamination (a locatable bug), rows
///     agreeing with each other while differing from serial is width-level
///     numerics that no per-row fix will touch;
///   * ragged vs equal-length rows isolate padding-derived causes.
enum BatchRowDivergenceProbe {

    /// The VLM factory first because that is what `NativeModelLoader` uses in
    /// production for every type it can; the LLM factory as the fallback for
    /// text-only families. A probe hard-wired to one factory measures a path
    /// production may never select — the exact mistake that produced the wrong
    /// gemma exclusion evidence (docs/KNOWN-FAILURES.md 1f).
    static func loadContainer(_ url: URL) async throws -> ModelContainer {
        do {
            return try await VLMModelFactory.shared.loadContainer(
                from: url, using: #huggingFaceTokenizerLoader())
        } catch {
            return try await LLMModelFactory.shared.loadContainer(
                from: url, using: #huggingFaceTokenizerLoader())
        }
    }

    static func modelURL(fromEnvironment key: String) throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[key] else {
            throw XCTSkip("Set \(key) to a model checkpoint to run this probe.")
        }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    /// Raw token arrays rather than chat-templated prompts, so LENGTH is the
    /// only variable between arms. Distinct content per row, controlled counts.
    static func tokens(lengths: [Int]) -> [MLXArray] {
        lengths.enumerated().map { row, length in
            MLXArray((0 ..< length).map { Int32(1000 + row * 97 + ($0 % 512)) })
        }
    }

    static func divergentRows(
        model: any LanguageModel, lengths: [Int], generated: Int
    ) async throws -> (diverged: Int, firstStep: Int) {
        try await divergentRows(
            model: model,
            inputs: tokens(lengths: lengths).map { LMInput(text: LMInput.Text(tokens: $0)) },
            generated: generated)
    }

    static func divergentRows(
        model: any LanguageModel, inputs: [LMInput], generated: Int
    ) async throws -> (diverged: Int, firstStep: Int) {
        let parameters = GenerateParameters(maxTokens: generated, temperature: 0)
        let serial = try inputs.map {
            try SerialGreedyTokenHelper.tokens(
                model: model, input: $0, parameters: parameters, steps: generated)
        }
        let engine = MLXCatEngine(
            model: model,
            parameters: parameters,
            maxConcurrentRequests: inputs.count,
            serializationPolicy: .never
        )
        let batched = try await engine.generate(
            inputs.enumerated().map { index, input in
                Request(
                    uid: "row-\(index)", input: input, maxTokens: generated,
                    sampling: SamplingParameters(temperature: 0))
            }
        )

        var diverged = 0
        var firstStep = Int.max
        for index in inputs.indices {
            let got = batched["row-\(index)"] ?? []
            guard got != serial[index] else { continue }
            diverged += 1
            for step in 0 ..< min(got.count, serial[index].count)
            where got[step] != serial[index][step] {
                firstStep = min(firstStep, step)
                break
            }
        }
        return (diverged, firstStep == Int.max ? -1 : firstStep)
    }

    /// Rows that are IDENTICAL in both length and content — see the type
    /// comment for how the two readouts split the verdict.
    static func identicalRowSplit(
        model: any LanguageModel, input: LMInput, width: Int, generated: Int
    ) async throws -> (vsSerial: Int, vsEachOther: Int) {
        let parameters = GenerateParameters(maxTokens: generated, temperature: 0)
        let serial = try SerialGreedyTokenHelper.tokens(
            model: model, input: input, parameters: parameters, steps: generated)
        let engine = MLXCatEngine(
            model: model, parameters: parameters,
            maxConcurrentRequests: width, serializationPolicy: .never)
        let batched = try await engine.generate(
            (0 ..< width).map { index in
                Request(
                    uid: "same-\(index)", input: input, maxTokens: generated,
                    sampling: SamplingParameters(temperature: 0))
            }
        )
        var vsSerial = 0
        var vsEachOther = 0
        let first = batched["same-0"] ?? []
        for index in 0 ..< width {
            let got = batched["same-\(index)"] ?? []
            if got != serial { vsSerial += 1 }
            if got != first { vsEachOther += 1 }
        }
        return (vsSerial, vsEachOther)
    }
}
