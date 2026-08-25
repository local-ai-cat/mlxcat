import Hub
import MLXCatBaselineKit
import os
import XCTest

/// Walks prompt length upward on a REAL iPhone until the model stops fitting,
/// and records where that happened.
///
/// WHY THIS EXISTS. The app budgets *weights* against the device
/// (`LocalAIAppRuntime.iOSLoadableCeilingBytes` asks the kernel via
/// `os_proc_available_memory()`), but nothing budgets the *KV cache*: a
/// model's context window is a static `generationProfile.contextWindow` from
/// the catalog and never consults memory. So the real limit on a phone is not
/// a number we chose — it is jetsam, and we had never measured it. Measured
/// 2026-08-26, gemma-4-E2B's 4k cell peaks at 4.75 GB against a 5.12 GB
/// ceiling: ~0.37 GB of headroom, on BOTH an 8 GB iPhone 16 Pro Max and a
/// 12 GB iPhone 17 Pro Max (iOS caps the entitled budget rather than scaling
/// it with RAM, so both report the same 4.77 GiB ceiling / 5.96 GiB available).
///
/// This is KNOWN-FAILURES 1c ("the memory ceiling is advisory, not enforcing")
/// turned from a claim into a curve.
///
/// HOW IT SURVIVES ITS OWN RESULT. A ladder that works is a ladder that
/// eventually gets its process killed, so the answer cannot live in the return
/// value or in an XCTAttachment — a jetsam kill produces neither. Every rung
/// prints TWO console lines: `BENCHRUNG attempt` before the work and
/// `BENCHRUNG survived` after it. The ceiling is therefore always recoverable
/// as "the highest rung with a survived line", even when the run ends in a
/// SIGKILL, and an `attempt` with no matching `survived` NAMES the rung that
/// killed the device. `run-context-ladder.sh` reads exactly that.
///
/// Config via the xcodebuild ENVIRONMENT (TEST_RUNNER_ prefix):
///   BENCHHOST_LADDER=1          required — this test is opt-in and skips
///                               otherwise, so a normal roster night never
///                               pays for it (and never gets killed by it).
///   BENCHHOST_MODEL_ID          HF repo id, or a comma-separated list.
///   BENCHHOST_LADDER_RUNGS      comma-separated prompt lengths. Default walks
///                               powers of two from 256 to 131072.
///   BENCHHOST_LADDER_VARIANT    `mlxcat` (default) or `tokeniterator`. One arm
///                               only: the ladder answers "how far does THIS
///                               engine get", and running both doubles the
///                               time spent to reach the same kill.
///   BENCHHOST_MEMORY_CEILING_BYTES  allocator ceiling override.
final class ContextCeilingLadderTests: XCTestCase {

    private static let defaultRungs = [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    func testContextCeilingLadder() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("MLX needs a real Apple Silicon iPhone; the simulator cannot run this.")
        #else
        let environment = ProcessInfo.processInfo.environment
        guard environment["BENCHHOST_LADDER"] == "1" else {
            throw XCTSkip("Set BENCHHOST_LADDER=1 to run the context-ceiling ladder.")
        }

        let modelIDs = (environment["BENCHHOST_MODEL_ID"] ?? "mlx-community/gemma-4-E2B-it-qat-4bit")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let rungs = (environment["BENCHHOST_LADDER_RUNGS"].map {
            $0.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        }.flatMap { $0.isEmpty ? nil : $0 }) ?? Self.defaultRungs
        let variant = BaselineVariant(rawValue: environment["BENCHHOST_LADDER_VARIANT"] ?? "mlxcat")
            ?? .mlxcat
        // One measured run, no warmup: the ladder is looking for the cliff, not
        // for a publishable throughput number. Rungs are already expensive at
        // the top and a warmup would pay the biggest prefill twice.
        let maxTokens = Int(environment["BENCHHOST_MAX_TOKENS"] ?? "") ?? 32

        let availableBytes = Int64(os_proc_available_memory())
        let defaultCeiling = availableBytes > 0 ? Int64(Double(availableBytes) * 0.8) : 4 << 30
        let ceiling = Int64(environment["BENCHHOST_MEMORY_CEILING_BYTES"] ?? "") ?? defaultCeiling
        let deviceModel = Self.hardwareModel()
        let osVersion = await MainActor.run { "iOS " + UIDevice.current.systemVersion }

        print("BENCHHOST ladder ceiling \(ceiling) bytes (available \(availableBytes)), variant \(variant.rawValue)")

        for modelID in modelIDs {
            let hub = HubApi()
            let modelDirectory = try await hub.snapshot(
                from: modelID,
                matching: [
                    "*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja",
                ]
            )
            let shortID = URL(fileURLWithPath: modelID).lastPathComponent

            for rung in rungs {
                Self.emit("attempt", [
                    "model": shortID,
                    "prompt_tokens": rung,
                    "variant": variant.rawValue,
                    "device": ["model": deviceModel, "os": osVersion],
                    "ceiling_bytes": ceiling,
                    "available_bytes": availableBytes,
                ])

                let configuration = BaselineConfiguration(
                    modelDirectory: modelDirectory,
                    modelID: shortID,
                    cells: [BaselineCell(promptTokens: rung, maxTokens: maxTokens)],
                    runs: 1,
                    warmup: 0,
                    variants: [variant],
                    memoryCeilingBytes: ceiling
                )

                // `emit` is @Sendable, so the footprint cannot be captured as a
                // plain `var` — same reason DeviceRowProducerTests carries a
                // locked RowBox rather than an array.
                let footprint = FootprintBox()
                do {
                    try await BaselineProducer.run(
                        configuration,
                        emit: { record in
                            let metrics = record["metrics"] as? [String: Any] ?? [:]
                            footprint.set(
                                peak: metrics["peak_phys_footprint_bytes"],
                                lifetime: metrics["lifetime_max_phys_footprint_bytes"])
                        },
                        log: { line in print("BENCHHOST \(line)") }
                    )
                } catch {
                    // A THROWN error is not the interesting outcome — it is the
                    // model refusing the length (context-window overflow), which
                    // is a policy limit rather than a memory one. Recorded and
                    // distinguished, then the ladder stops: longer only overflows
                    // harder.
                    Self.emit("refused", [
                        "model": shortID,
                        "prompt_tokens": rung,
                        "variant": variant.rawValue,
                        "device": ["model": deviceModel, "os": osVersion],
                        "error": String(describing: error),
                    ])
                    break
                }

                Self.emit("survived", [
                    "model": shortID,
                    "prompt_tokens": rung,
                    "variant": variant.rawValue,
                    "device": ["model": deviceModel, "os": osVersion],
                    "peak_phys_footprint_bytes": footprint.peak,
                    "lifetime_max_phys_footprint_bytes": footprint.lifetime,
                    "ceiling_bytes": ceiling,
                ])
            }
        }
        #endif
    }

    /// Footprint carried back across the @Sendable emit callback.
    private final class FootprintBox: @unchecked Sendable {
        private let lock = NSLock()
        private var storedPeak: Any = 0
        private var storedLifetime: Any = 0
        func set(peak: Any?, lifetime: Any?) {
            lock.lock()
            defer { lock.unlock() }
            storedPeak = peak ?? 0
            storedLifetime = lifetime ?? 0
        }
        var peak: Any {
            lock.lock(); defer { lock.unlock() }; return storedPeak
        }
        var lifetime: Any {
            lock.lock(); defer { lock.unlock() }; return storedLifetime
        }
    }

    /// One line per event, flushed through `print` — the console is the only
    /// channel that outlives a jetsam kill.
    private static func emit(_ kind: String, _ payload: [String: Any]) {
        var stamped = payload
        stamped["event"] = kind
        stamped["schema"] = "mlxcat-ladder/1"
        guard
            let data = try? JSONSerialization.data(withJSONObject: stamped, options: [.sortedKeys]),
            let line = String(data: data, encoding: .utf8)
        else { return }
        print("BENCHRUNG \(kind) \(line)")
    }

    private static func hardwareModel() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafeBytes(of: &systemInfo.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}
