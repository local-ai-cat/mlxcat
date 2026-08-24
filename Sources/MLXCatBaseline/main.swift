// mlxcat-baseline — in-process reference rows for the leaderboard.
//
// Thin CLI over `MLXCatBaselineKit.BaselineProducer`, which owns the two arms
// (raw TokenIterator = the legacy LLMEvaluator mechanism; MLXCatEngine
// in-process), the prompt corpus, and the footprint metric. The split exists
// so an on-device XCTest can produce iOS rows through the same code — this
// file is only argument parsing, stdout emission, and process exit.
//
// Output: one JSON object per line on stdout, schema `mlxcat-bench/1`, with
// `engine.transport = "in-process"`. `bench/run.py` drives this binary as a
// "producer" engine and stamps device/host/validity before appending the rows.
//
//   swift run -c release mlxcat-baseline --model-dir ~/Library/Caches/models/mlx-community/Qwen3.5-4B-MLX-4bit \
//       --prompt-tokens 256 --prompt-tokens 4096 --max-tokens 128 --runs 3 --warmup 1 --variant both

import Foundation
import MLXCatBaselineKit

struct Options {
    var modelDir: String = ""
    var cells: [BaselineCell] = []
    var maxTokens: Int = 128
    var runs: Int = 3
    var warmup: Int = 1
    var variants: [BaselineVariant] = BaselineVariant.allCases
    var modelID: String = ""
    var memoryCeilingBytes: Int64 = 0

    static func parse(_ args: [String]) -> Options {
        var options = Options()
        var promptTokens: [Int] = []
        var cells: [BaselineCell] = []
        var index = 1
        func value() -> String {
            index += 1
            guard index < args.count else {
                fputs("missing value for \(args[index - 1])\n", stderr)
                exit(64)
            }
            return args[index]
        }
        while index < args.count {
            switch args[index] {
            case "--model-dir": options.modelDir = value()
            case "--model-id": options.modelID = value()
            case "--prompt-tokens": promptTokens.append(Int(value()) ?? 256)
            case "--cell":
                // "PROMPT:MAX" — per-cell token budgets, so a 16k cell can carry a
                // smaller generation budget than a short cell (matrix parity).
                // Malformed input is rejected, never silently defaulted — a wrong
                // benchmark is worse than no benchmark.
                let raw = value()
                let parts = raw.split(separator: ":", omittingEmptySubsequences: false)
                guard parts.count == 2, let prompt = Int(parts[0]), let max = Int(parts[1]), prompt > 0, max > 0 else {
                    fputs("invalid --cell '\(raw)' (expected PROMPT:MAX, both positive integers)\n", stderr)
                    exit(64)
                }
                cells.append(BaselineCell(promptTokens: prompt, maxTokens: max))
            case "--memory-ceiling-bytes": options.memoryCeilingBytes = Int64(value()) ?? 0
            case "--max-tokens": options.maxTokens = Int(value()) ?? 128
            case "--runs": options.runs = Int(value()) ?? 3
            case "--warmup": options.warmup = Int(value()) ?? 1
            case "--variant":
                let v = value()
                if v == "both" {
                    options.variants = BaselineVariant.allCases
                } else if let variant = BaselineVariant(rawValue: v) {
                    options.variants = [variant]
                } else {
                    fputs("unknown variant \(v) (expected tokeniterator|mlxcat|both)\n", stderr)
                    exit(64)
                }
            case "--help", "-h":
                print("usage: mlxcat-baseline --model-dir DIR [--cell PROMPT:MAX]... [--prompt-tokens N]... [--max-tokens N] [--runs N] [--warmup N] [--variant tokeniterator|mlxcat|both] [--memory-ceiling-bytes N]")
                exit(0)
            default:
                fputs("unknown argument \(args[index])\n", stderr)
                exit(64)
            }
            index += 1
        }
        if !cells.isEmpty {
            options.cells = cells
        } else if !promptTokens.isEmpty {
            options.cells = promptTokens.map { BaselineCell(promptTokens: $0, maxTokens: options.maxTokens) }
        } else {
            options.cells = [BaselineCell(promptTokens: 256, maxTokens: options.maxTokens)]
        }
        if options.modelDir.isEmpty {
            fputs("--model-dir is required\n", stderr)
            exit(64)
        }
        if options.modelID.isEmpty {
            options.modelID = URL(fileURLWithPath: options.modelDir).lastPathComponent
        }
        return options
    }
}

let options = Options.parse(CommandLine.arguments)

let configuration = BaselineConfiguration(
    modelDirectory: URL(fileURLWithPath: (options.modelDir as NSString).expandingTildeInPath),
    modelID: options.modelID,
    cells: options.cells,
    runs: options.runs,
    warmup: options.warmup,
    variants: options.variants,
    memoryCeilingBytes: options.memoryCeilingBytes
)

do {
    try await BaselineProducer.run(
        configuration,
        emit: { record in
            guard let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
                  let text = String(data: data, encoding: .utf8) else { return }
            print(text)
            fflush(stdout)
        },
        log: { line in fputs(line + "\n", stderr) }
    )
} catch {
    fputs("mlxcat-baseline failed: \(error)\n", stderr)
    exit(1)
}
