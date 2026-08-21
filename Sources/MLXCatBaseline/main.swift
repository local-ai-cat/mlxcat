// mlxcat-baseline — in-process reference rows for the leaderboard.
//
// Two variants, measured the same way, in the same process, on the same prompt:
//
//   tokeniterator  mlx-swift-lm's raw `TokenIterator` loop. This is exactly the
//                  mechanism behind Local AI Cat's legacy `LLMEvaluator` (the
//                  mlx-swift-examples lineage) — the single-sequence baseline
//                  mlxcat replaced, and the one that out-ran it at width 1 for a
//                  while in 2026-08. If mlxcat is slower than this row, that is a
//                  regression, full stop.
//   mlxcat         `MLXCatEngine` in-process (continuous-batch generator, no
//                  HTTP) — separates engine cost from transport cost when read
//                  next to the `mlxcat` HTTP row.
//
// Output: one JSON object per line on stdout, schema `mlxcat-bench/1`, with
// `engine.transport = "in-process"`. `bench/run.py` drives this binary as a
// "producer" engine and stamps device/host/validity before appending the rows.
//
//   swift run -c release mlxcat-baseline --model-dir ~/Library/Caches/models/mlx-community/Qwen3.5-4B-MLX-4bit \
//       --prompt-tokens 256 --prompt-tokens 4096 --max-tokens 128 --runs 3 --warmup 1 --variant both

import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers

// MARK: - CLI

struct Options {
    var modelDir: String = ""
    var cells: [(promptTokens: Int, maxTokens: Int)] = [(256, 128)]
    var maxTokens: Int = 128
    var runs: Int = 3
    var warmup: Int = 1
    var variants: [String] = ["tokeniterator", "mlxcat"]
    var modelID: String = ""

    static func parse(_ args: [String]) -> Options {
        var options = Options()
        var promptTokens: [Int] = []
        var cells: [(promptTokens: Int, maxTokens: Int)] = []
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
                cells.append((prompt, max))
            case "--max-tokens": options.maxTokens = Int(value()) ?? 128
            case "--runs": options.runs = Int(value()) ?? 3
            case "--warmup": options.warmup = Int(value()) ?? 1
            case "--variant":
                let v = value()
                options.variants = v == "both" ? ["tokeniterator", "mlxcat"] : [v]
            case "--help", "-h":
                print("usage: mlxcat-baseline --model-dir DIR [--cell PROMPT:MAX]... [--prompt-tokens N]... [--max-tokens N] [--runs N] [--warmup N] [--variant tokeniterator|mlxcat|both]")
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
            options.cells = promptTokens.map { ($0, options.maxTokens) }
        } else {
            options.cells = [(256, options.maxTokens)]
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

// MARK: - Footprint (same metric as bench/run.py: proc_pid_rusage / RUSAGE_INFO_V4)

enum Footprint {
    static func read() -> (phys: UInt64, lifetimeMax: UInt64)? {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) { pointer -> Int32 in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { raw in
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, raw)
            }
        }
        guard rc == 0 else { return nil }
        return (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
    }
}

final class FootprintSampler: @unchecked Sendable {
    private let queue = DispatchQueue(label: "mlxcat-baseline.footprint")
    private var timer: DispatchSourceTimer?
    private(set) var peak: UInt64 = 0

    func start() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(50))
        timer.setEventHandler { [weak self] in
            guard let self, let reading = Footprint.read() else { return }
            self.peak = max(self.peak, reading.phys)
        }
        timer.resume()
        self.timer = timer
    }

    func stop() -> UInt64 {
        timer?.cancel()
        timer = nil
        return queue.sync { peak }
    }
}

// MARK: - Prompt corpus (identical text to bench/run.py)

let filler = "The river runs past the old mill and under the stone bridge, where the light breaks on the water in the early morning. Farmers bring grain in carts drawn by patient horses, and the miller weighs each sack on a brass scale that has hung from the same beam for a hundred years. "
let question = "Summarize the passage above in three sentences, then explain in plain prose how a CPU executes an instruction. Keep writing until you have several complete paragraphs."

// MARK: - Stats

struct Sample {
    let ttftMilliseconds: Double
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let e2eTokensPerSecond: Double
    let generated: Int
}

func spread(_ values: [Double]) -> [String: Any]? {
    let clean = values.filter { $0.isFinite }.sorted()
    guard let first = clean.first, let last = clean.last else { return nil }
    let median: Double
    if clean.count % 2 == 1 {
        median = clean[clean.count / 2]
    } else {
        median = (clean[clean.count / 2 - 1] + clean[clean.count / 2]) / 2
    }
    return ["median": median, "min": first, "max": last, "n": clean.count, "spread_ratio": last / max(first, 1e-9)]
}

func nowISO() -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.string(from: Date())
}

func emit(_ record: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
          let text = String(data: data, encoding: .utf8) else { return }
    print(text)
    fflush(stdout)
}

// MARK: - Main

let options = Options.parse(CommandLine.arguments)
let modelURL = URL(fileURLWithPath: (options.modelDir as NSString).expandingTildeInPath)

@MainActor
func run() async throws {
    let container = try await LLMModelFactory.shared.loadContainer(from: modelURL, using: #huggingFaceTokenizerLoader())
    let loadPeak = Footprint.read()?.lifetimeMax ?? 0
    fputs("loaded \(options.modelID) (lifetime max footprint after load \(Double(loadPeak) / 1_073_741_824) GiB)\n", stderr)

    try await container.perform { context in
        // Calibrate chars/token on this tokenizer.
        let questionTokens = context.tokenizer.encode(text: question).count
        let fillerTokens = max(context.tokenizer.encode(text: filler).count, 1)

        for (target, cellMaxTokens) in options.cells {
            let repeats = max((target - questionTokens) / fillerTokens, 0)
            let prompt = String(repeating: filler, count: repeats) + question
            let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
            let promptTokenCount = input.text.tokens.size

            for variant in options.variants {
                func oneRun() async throws -> Sample {
                    let started = DispatchTime.now().uptimeNanoseconds
                    var firstTokenAt = started
                    var count = 0
                    switch variant {
                    case "tokeniterator":
                        var iterator = try TokenIterator(
                            input: input,
                            model: context.model,
                            parameters: GenerateParameters(maxTokens: cellMaxTokens, temperature: 0)
                        )
                        while iterator.next() != nil {
                            count += 1
                            if count == 1 { firstTokenAt = DispatchTime.now().uptimeNanoseconds }
                        }
                    case "mlxcat":
                        let engine = MLXCatEngine(
                            model: context.model,
                            parameters: GenerateParameters(maxTokens: cellMaxTokens, temperature: 0),
                            maxConcurrentRequests: 8
                        )
                        let request = Request(
                            uid: "baseline-\(UUID().uuidString)",
                            input: input,
                            maxTokens: cellMaxTokens,
                            sampling: SamplingParameters(temperature: 0)
                        )
                        for try await response in engine.stream(request) {
                            guard response.token >= 0 else { continue }
                            count += 1
                            if count == 1 { firstTokenAt = DispatchTime.now().uptimeNanoseconds }
                        }
                    default:
                        fputs("unknown variant \(variant)\n", stderr)
                        exit(64)
                    }
                    let ended = DispatchTime.now().uptimeNanoseconds
                    let ttft = Double(firstTokenAt - started) / 1_000_000_000
                    let decodeSeconds = Double(ended - firstTokenAt) / 1_000_000_000
                    let total = Double(ended - started) / 1_000_000_000
                    return Sample(
                        ttftMilliseconds: ttft * 1000,
                        prefillTokensPerSecond: ttft > 0 ? Double(promptTokenCount) / ttft : .nan,
                        decodeTokensPerSecond: decodeSeconds > 0 ? Double(max(count - 1, 0)) / decodeSeconds : .nan,
                        e2eTokensPerSecond: total > 0 ? Double(count) / total : .nan,
                        generated: count
                    )
                }

                let coldStarted = DispatchTime.now().uptimeNanoseconds
                _ = try await oneRun()
                let coldMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - coldStarted) / 1_000_000
                for _ in 0 ..< options.warmup { _ = try await oneRun() }

                let sampler = FootprintSampler()
                sampler.start()
                var samples: [Sample] = []
                for _ in 0 ..< options.runs { samples.append(try await oneRun()) }
                let peak = sampler.stop()
                let lifetimeMax = Footprint.read()?.lifetimeMax ?? 0

                let engineName = variant == "tokeniterator" ? "mlx-swift-lm-tokeniterator" : "mlxcat-inprocess"
                let notes = variant == "tokeniterator"
                    ? "raw mlx-swift-lm TokenIterator — the legacy LLMEvaluator mechanism (single sequence, no HTTP)"
                    : "MLXCatEngine in-process (no HTTP) — engine cost without transport"
                let record: [String: Any] = [
                    "schema": "mlxcat-bench/1",
                    "timestamp": nowISO(),
                    "engine": [
                        "name": engineName,
                        "version": NSNull(),
                        "transport": "in-process",
                        "weights": "mlx-community safetensors (same files for every engine)",
                        "notes": notes,
                    ],
                    "model": ["id": options.modelID, "offered_as": options.modelID],
                    "workload": [
                        "context_tier": "tokens-\(target)",
                        "prompt_tokens_target": target,
                        "max_tokens": cellMaxTokens,
                        "concurrency": 1,
                        "temperature": 0,
                        "runs": options.runs,
                        "warmup": options.warmup,
                    ],
                    "metrics": [
                        "ttft_ms": spread(samples.map(\.ttftMilliseconds)) ?? NSNull(),
                        "prefill_tps": spread(samples.map(\.prefillTokensPerSecond)) ?? NSNull(),
                        "decode_tps": spread(samples.map(\.decodeTokensPerSecond)) ?? NSNull(),
                        "e2e_tps": spread(samples.map(\.e2eTokensPerSecond)) ?? NSNull(),
                        "prompt_tokens": promptTokenCount,
                        "completion_tokens": samples.map(\.generated).sorted()[samples.count / 2],
                        "cold_first_request_ms": coldMilliseconds,
                        "peak_phys_footprint_bytes": peak,
                        "lifetime_max_phys_footprint_bytes": lifetimeMax,
                    ],
                ]
                emit(record)
            }
        }
    }
}

do {
    try await run()
} catch {
    fputs("mlxcat-baseline failed: \(error)\n", stderr)
    exit(1)
}
