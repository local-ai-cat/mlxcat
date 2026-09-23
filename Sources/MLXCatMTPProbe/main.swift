// MLXCatMTPProbe — single-stream native-MTP prototype (overnight 2026-09-23, packet 100/148).
//
// Loads a Qwen3.5-family checkpoint as the target plus a separately-configured MTP drafter
// directory, and decodes each prompt either plainly (`TokenIterator`) or with
// `MTPSpeculativeTokenIterator`. No Scheduler/BatchGenerator involvement: this answers
// "is MTP exact and faster on this machine", nothing about serving.
//
// If the drafter fails to load, `--mode mtp` falls back to plain decode and says so in every
// row (`fallback` field) — a probe row can never claim MTP it did not run.
//
// Usage:
//   mlxcat-mtp-probe --model DIR --drafter DIR --prompts FILE.jsonl --mode plain|mtp
//                    [--max-tokens 128] [--out FILE.jsonl] [--repeat 1]
// Prompt file rows: {"id": "...", "prompt": "..."}; rendered through the model's chat template.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

struct ProbeConfig {
    var modelPath = ""
    var drafterPath = ""
    var promptsPath = ""
    var mode = "plain"
    var maxTokens = 128
    var outPath: String?
    var repeatCount = 1

    static func parse(_ args: [String]) throws -> ProbeConfig {
        var config = ProbeConfig()
        var index = 1
        func value() throws -> String {
            index += 1
            guard index < args.count else { throw ProbeError.usage("missing value for \(args[index - 1])") }
            return args[index]
        }
        while index < args.count {
            switch args[index] {
            case "--model": config.modelPath = try value()
            case "--drafter": config.drafterPath = try value()
            case "--prompts": config.promptsPath = try value()
            case "--mode": config.mode = try value()
            case "--max-tokens": config.maxTokens = Int(try value()) ?? config.maxTokens
            case "--out": config.outPath = try value()
            case "--repeat": config.repeatCount = Int(try value()) ?? 1
            default: throw ProbeError.usage("unknown argument \(args[index])")
            }
            index += 1
        }
        guard !config.modelPath.isEmpty, !config.promptsPath.isEmpty,
            ["plain", "mtp"].contains(config.mode)
        else { throw ProbeError.usage("need --model, --prompts, --mode plain|mtp") }
        return config
    }
}

enum ProbeError: Error { case usage(String) }

struct ProbeRow: Encodable {
    let id: String
    let rep: Int
    let mode: String
    let fallback: String?
    let promptTokens: Int
    let generatedTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let proposed: Int
    let accepted: Int
    let passthroughReason: String?
    let sabotageArmed: Bool
    let tokens: [Int]
    let textHead: String
}

private func loadPrompts(_ path: String) throws -> [(String, String)] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return try text.split(separator: "\n").compactMap { line in
        guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
        let object = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
        guard let id = object?["id"] as? String, let prompt = object?["prompt"] as? String else {
            return nil
        }
        return (id, prompt)
    }
}

private func stopTokens(_ context: ModelContext) -> Set<Int> {
    var stops = Set<Int>()
    if let eos = context.tokenizer.eosTokenId { stops.insert(eos) }
    for extra in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(extra) { stops.insert(id) }
    }
    for literal in ["<|im_end|>", "<|endoftext|>"] {
        if let id = context.tokenizer.convertTokenToId(literal) { stops.insert(id) }
    }
    return stops
}

/// Drain an iterator, timing prefill (construction) separately from decode.
private func drain<I: TokenIteratorProtocol>(
    _ iterator: inout I, stops: Set<Int>, maxTokens: Int
) -> (tokens: [Int], decodeSeconds: Double) {
    var tokens = [Int]()
    var firstTokenAt: Double?
    while tokens.count < maxTokens, let token = iterator.next() {
        if firstTokenAt == nil { firstTokenAt = Date.timeIntervalSinceReferenceDate }
        tokens.append(token)
        if stops.contains(token) { break }
    }
    let end = Date.timeIntervalSinceReferenceDate
    return (tokens, end - (firstTokenAt ?? end))
}

@main
struct MLXCatMTPProbe {
    static func main() async throws {
        let config = try ProbeConfig.parse(CommandLine.arguments)
        let loader = #huggingFaceTokenizerLoader()
        let context = try await LLMModelFactory.shared.load(
            from: URL(fileURLWithPath: config.modelPath), using: loader)

        var drafter: (any MTPDrafterModel)?
        var fallback: String?
        if config.mode == "mtp" {
            await Qwen35TextMTPRegistration.register()
            do {
                let drafterContext = try await MTPDrafterModelFactory.shared.load(
                    from: URL(fileURLWithPath: config.drafterPath), using: loader)
                drafter = drafterContext.model
            } catch {
                fallback = "drafter load failed: \(error)"
                FileHandle.standardError.write(Data("⚠️ \(fallback!) — plain decode\n".utf8))
            }
        }

        let prompts = try loadPrompts(config.promptsPath)
        let stops = stopTokens(context)
        let sabotage = ProcessInfo.processInfo.environment["MLX_MTP_SABOTAGE_ACCEPT_WRONG"] == "1"
        var out = ""
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase

        for rep in 0 ..< config.repeatCount {
            for (id, prompt) in prompts {
                let promptTokens = try context.tokenizer.applyChatTemplate(
                    messages: [["role": "user", "content": prompt]])
                let input = LMInput(tokens: MLXArray(promptTokens.map { Int32($0) }))
                let parameters = GenerateParameters(maxTokens: config.maxTokens, temperature: 0)

                let start = Date.timeIntervalSinceReferenceDate
                let tokens: [Int]
                let decodeSeconds: Double
                var proposed = 0
                var accepted = 0
                var passthrough: String?
                var prefill: Double
                if let drafter {
                    var iterator = try MTPSpeculativeTokenIterator(
                        input: input, mainModel: context.model, drafter: drafter,
                        parameters: parameters, blockSize: 2)
                    prefill = Date.timeIntervalSinceReferenceDate - start
                    (tokens, decodeSeconds) = drain(
                        &iterator, stops: stops, maxTokens: config.maxTokens)
                    proposed = iterator.proposedDraftTokens
                    accepted = iterator.acceptedDraftTokens
                    passthrough = iterator.passthroughReason
                } else {
                    var iterator = try TokenIterator(
                        input: input, model: context.model, parameters: parameters)
                    prefill = Date.timeIntervalSinceReferenceDate - start
                    (tokens, decodeSeconds) = drain(
                        &iterator, stops: stops, maxTokens: config.maxTokens)
                }
                let rate = tokens.count > 1 && decodeSeconds > 0
                    ? Double(tokens.count - 1) / decodeSeconds : 0
                let row = ProbeRow(
                    id: id, rep: rep, mode: config.mode, fallback: fallback,
                    promptTokens: promptTokens.count, generatedTokens: tokens.count,
                    prefillSeconds: prefill, decodeSeconds: decodeSeconds,
                    decodeTokensPerSecond: rate, proposed: proposed, accepted: accepted,
                    passthroughReason: passthrough, sabotageArmed: sabotage, tokens: tokens,
                    textHead: String(context.tokenizer.decode(tokenIds: tokens).prefix(160)))
                let line = String(decoding: try encoder.encode(row), as: UTF8.self)
                print(
                    "\(id) rep\(rep) \(config.mode) gen=\(tokens.count) "
                        + String(format: "%.2f tok/s", rate)
                        + " proposed=\(proposed) accepted=\(accepted)")
                out += line + "\n"
                Memory.clearCache()
            }
        }
        if let outPath = config.outPath {
            try out.write(toFile: outPath, atomically: true, encoding: .utf8)
        }
    }
}
