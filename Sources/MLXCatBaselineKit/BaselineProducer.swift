// MLXCatBaselineKit — the measurement core of `mlxcat-baseline`, as a library.
//
// Split out of the CLI (2026-08-24) so an on-device XCTest can produce iOS
// leaderboard rows with the same two arms, the same corpus, and the same
// footprint metric as the Mac producer — pkg 113's `ios-row-producer`. The CLI
// (`Sources/MLXCatBaseline/main.swift`) stays the Mac entry point; this layer
// owns everything except argument parsing and process exit.
//
// The two arms:
//   .tokenIterator  mlx-swift-lm's raw `TokenIterator` loop — exactly the
//                   mechanism behind Local AI Cat's legacy `LLMEvaluator`. If
//                   mlxcat is slower than this row, that is a regression.
//   .mlxcat         `MLXCatEngine` in-process (continuous-batch generator,
//                   no HTTP) — engine cost without transport cost.
//
// Records are `mlxcat-bench/1` dictionaries handed to the caller's `emit`
// closure; the caller stamps platform/device/validity and decides where they
// go (stdout on the Mac CLI, a JSONL attachment in the device test).

import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import MLXCat
import Tokenizers

public enum BaselineVariant: String, CaseIterable, Sendable {
    case tokenIterator = "tokeniterator"
    case mlxcat = "mlxcat"

    var engineName: String {
        switch self {
        case .tokenIterator: return "mlx-swift-lm-tokeniterator"
        case .mlxcat: return "mlxcat-inprocess"
        }
    }

    var notes: String {
        switch self {
        case .tokenIterator:
            return "raw mlx-swift-lm TokenIterator — the legacy LLMEvaluator mechanism (single sequence, no HTTP)"
        case .mlxcat:
            return "MLXCatEngine in-process (no HTTP) — engine cost without transport"
        }
    }
}

public struct BaselineCell: Sendable {
    public let promptTokens: Int
    public let maxTokens: Int

    public init(promptTokens: Int, maxTokens: Int) {
        self.promptTokens = promptTokens
        self.maxTokens = maxTokens
    }
}

public struct BaselineConfiguration: Sendable {
    public var modelDirectory: URL
    public var modelID: String
    public var cells: [BaselineCell]
    public var runs: Int
    public var warmup: Int
    public var variants: [BaselineVariant]
    public var memoryCeilingBytes: Int64

    public init(
        modelDirectory: URL,
        modelID: String,
        cells: [BaselineCell],
        runs: Int = 3,
        warmup: Int = 1,
        variants: [BaselineVariant] = BaselineVariant.allCases,
        memoryCeilingBytes: Int64 = 0
    ) {
        self.modelDirectory = modelDirectory
        self.modelID = modelID
        self.cells = cells
        self.runs = runs
        self.warmup = warmup
        self.variants = variants
        self.memoryCeilingBytes = memoryCeilingBytes
    }
}

public enum BaselineError: Error, CustomStringConvertible {
    case noSamples

    public var description: String {
        switch self {
        case .noSamples: return "a measurement cell produced no samples"
        }
    }
}

// MARK: - Footprint (same metric as bench/run.py: proc_pid_rusage / RUSAGE_INFO_V4)

public enum BaselineFootprint {
    #if os(macOS)
    public static func read() -> (phys: UInt64, lifetimeMax: UInt64)? {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) { pointer -> Int32 in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { raw in
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, raw)
            }
        }
        guard rc == 0 else { return nil }
        return (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
    }
    #else
    // iOS has no proc_pid_rusage in the public SDK; task_vm_info reads the
    // same phys_footprint ledger for the current task, and its ledger peak is
    // the same "lifetime max" semantic. Values stay comparable to the Mac
    // rows because both report the kernel's phys_footprint accounting.
    public static func read() -> (phys: UInt64, lifetimeMax: UInt64)? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride)
        let rc = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { raw in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), raw, &count)
            }
        }
        guard rc == KERN_SUCCESS else { return nil }
        return (UInt64(info.phys_footprint), UInt64(info.ledger_phys_footprint_peak))
    }
    #endif
}

final class FootprintSampler: @unchecked Sendable {
    private let queue = DispatchQueue(label: "mlxcat-baseline.footprint")
    private var timer: DispatchSourceTimer?
    private var peak: UInt64 = 0
    private var sampled = false

    func start() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(50))
        timer.setEventHandler { [weak self] in
            guard let self, let reading = BaselineFootprint.read() else { return }
            self.peak = max(self.peak, reading.phys)
            self.sampled = true
        }
        timer.resume()
        self.timer = timer
    }

    /// nil when not a single read succeeded — an absent metric, never a
    /// measured zero. A row claiming a zero-byte footprint because task_info
    /// failed would make broken data look ideal (Codex round-2 finding).
    func stop() -> UInt64? {
        timer?.cancel()
        timer = nil
        return queue.sync { sampled ? peak : nil }
    }
}

// MARK: - Prompt corpus (identical text to bench/run.py)

private let filler = "The river runs past the old mill and under the stone bridge, where the light breaks on the water in the early morning. Farmers bring grain in carts drawn by patient horses, and the miller weighs each sack on a brass scale that has hung from the same beam for a hundred years. "
private let question = "Summarize the passage above in three sentences, then explain in plain prose how a CPU executes an instruction. Keep writing until you have several complete paragraphs."

// MARK: - Stats

private struct Sample {
    let ttftMilliseconds: Double
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let e2eTokensPerSecond: Double
    let generated: Int
}

private func spread(_ values: [Double]) -> [String: Any]? {
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

private func nowISO() -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.string(from: Date())
}

// MARK: - Producer

public enum BaselineProducer {

    /// Load the model, measure every (cell × variant), and hand each finished
    /// `mlxcat-bench/1` record to `emit`. Progress prose goes to `log` (stderr
    /// on the CLI, XCTAttachment/console in a test).
    @MainActor
    public static func run(
        _ configuration: BaselineConfiguration,
        emit: @escaping @Sendable ([String: Any]) -> Void,
        log: @escaping @Sendable (String) -> Void
    ) async throws {
        // Measure under the SAME memory policy as mlxcat-http, or the in-process
        // rows are not comparable to the server rows. Without this the tool
        // inherits MLX's defaults (cacheLimit == memoryLimit == 1.5x recommended
        // working set), and a 16k cell on a 20B model drives the host into
        // memory pressure: measured 2026-08-21, gpt-oss-20b at 16k decoded
        // 15.4 tok/s here versus 47.0 through the server on the same host.
        let applied = MemoryGuard.applyAllocatorLimits(
            ceilingBytes: configuration.memoryCeilingBytes,
            cacheLimitOverrideBytes: MemoryGuard.cacheLimitOverrideFromEnvironment()
        )
        if applied.memoryLimit > 0 {
            log("allocator: memoryLimit \(applied.memoryLimit) cacheLimit \(applied.cacheLimit)")
        }
        // VLM factory first, LLM as the text-only fallback — the same order
        // production's NativeModelLoader and the divergence probes use. Loading
        // gemma through LLMModelFactory dies on keyNotFound(language_model...):
        // the LLM Gemma4Model expects flat model.layers.* while the checkpoint
        // ships language_model.*, and only the VLM class sanitizes that. This
        // exact wrong-factory mistake is the one KNOWN-FAILURES 1f documents;
        // the first iPhone run reproduced it on device (2026-08-25).
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                from: configuration.modelDirectory, using: #huggingFaceTokenizerLoader())
        } catch {
            container = try await LLMModelFactory.shared.loadContainer(
                from: configuration.modelDirectory, using: #huggingFaceTokenizerLoader())
        }
        let loadPeak = BaselineFootprint.read()?.lifetimeMax ?? 0
        log("loaded \(configuration.modelID) (lifetime max footprint after load \(Double(loadPeak) / 1_073_741_824) GiB)")

        try await container.perform { context in
            // Calibrate chars/token on this tokenizer.
            let questionTokens = context.tokenizer.encode(text: question).count
            let fillerTokens = max(context.tokenizer.encode(text: filler).count, 1)

            for cell in configuration.cells {
                let repeats = max((cell.promptTokens - questionTokens) / fillerTokens, 0)
                let prompt = String(repeating: filler, count: repeats) + question
                let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
                let promptTokenCount = input.text.tokens.size

                for variant in configuration.variants {
                    func oneRun() async throws -> Sample {
                        let started = DispatchTime.now().uptimeNanoseconds
                        var firstTokenAt = started
                        var count = 0
                        switch variant {
                        case .tokenIterator:
                            var iterator = try TokenIterator(
                                input: input,
                                model: context.model,
                                parameters: GenerateParameters(maxTokens: cell.maxTokens, temperature: 0)
                            )
                            while iterator.next() != nil {
                                count += 1
                                if count == 1 { firstTokenAt = DispatchTime.now().uptimeNanoseconds }
                            }
                        case .mlxcat:
                            let engine = MLXCatEngine(
                                model: context.model,
                                parameters: GenerateParameters(maxTokens: cell.maxTokens, temperature: 0),
                                maxConcurrentRequests: 8
                            )
                            let request = Request(
                                uid: "baseline-\(UUID().uuidString)",
                                input: input,
                                maxTokens: cell.maxTokens,
                                sampling: SamplingParameters(temperature: 0)
                            )
                            for try await response in engine.stream(request) {
                                guard response.token >= 0 else { continue }
                                count += 1
                                if count == 1 { firstTokenAt = DispatchTime.now().uptimeNanoseconds }
                            }
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
                    for _ in 0 ..< configuration.warmup { _ = try await oneRun() }

                    let sampler = FootprintSampler()
                    sampler.start()
                    var samples: [Sample] = []
                    for _ in 0 ..< configuration.runs { samples.append(try await oneRun()) }
                    let peak = sampler.stop()
                    let lifetimeMax = BaselineFootprint.read()?.lifetimeMax
                    guard !samples.isEmpty else { throw BaselineError.noSamples }

                    // One process measures every cell, so a long-context cell
                    // would otherwise leave its scratch resident and depress the
                    // next one.
                    Memory.clearCache()

                    let peakValue: Any = peak.map { $0 as Any } ?? NSNull()
                    let lifetimeValue: Any
                    if let lifetimeMax, lifetimeMax > 0 {
                        lifetimeValue = lifetimeMax
                    } else {
                        lifetimeValue = NSNull()
                    }

                    let record: [String: Any] = [
                        "schema": "mlxcat-bench/1",
                        "timestamp": nowISO(),
                        "engine": [
                            "name": variant.engineName,
                            "version": NSNull(),
                            "transport": "in-process",
                            "weights": "mlx-community safetensors (same files for every engine)",
                            "notes": variant.notes,
                        ],
                        "model": ["id": configuration.modelID, "offered_as": configuration.modelID],
                        "workload": [
                            "context_tier": "tokens-\(cell.promptTokens)",
                            "prompt_tokens_target": cell.promptTokens,
                            "max_tokens": cell.maxTokens,
                            "concurrency": 1,
                            "temperature": 0,
                            "runs": configuration.runs,
                            "warmup": configuration.warmup,
                        ],
                        "metrics": [
                            "ttft_ms": spread(samples.map(\.ttftMilliseconds)) ?? NSNull(),
                            "prefill_tps": spread(samples.map(\.prefillTokensPerSecond)) ?? NSNull(),
                            "decode_tps": spread(samples.map(\.decodeTokensPerSecond)) ?? NSNull(),
                            "e2e_tps": spread(samples.map(\.e2eTokensPerSecond)) ?? NSNull(),
                            "prompt_tokens": promptTokenCount,
                            "completion_tokens": samples.map(\.generated).sorted()[samples.count / 2],
                            "cold_first_request_ms": coldMilliseconds,
                            "peak_phys_footprint_bytes": peakValue,
                            "lifetime_max_phys_footprint_bytes": lifetimeValue,
                        ],
                    ]
                    emit(record)
                }
            }
        }
    }
}
