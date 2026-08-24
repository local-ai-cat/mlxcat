import Hub
import MLXCatBaselineKit
import os
import XCTest

/// Produces iOS leaderboard rows on a REAL iPhone — mlxcat-bench/1 JSONL via
/// the same `BaselineProducer` the Mac CLI uses, so the two arms (mlxcat
/// in-process vs the raw-TokenIterator legacy-LLMEvaluator mechanism), the
/// prompt corpus, and the footprint metric are identical across platforms.
///
/// The model downloads once into the host app's container (HubApi snapshot)
/// and is reused on later runs. Rows land as an XCTAttachment named
/// `mlxcat-bench-ios.jsonl` — pull them from the xcresult and append to
/// `bench/results/` after stamping validity (a phone has no loadavg; the
/// stamp is "screen locked, on power" by the operator).
///
/// Config via the test plan / xcodebuild environment:
///   BENCHHOST_MODEL_ID    HF repo id (default mlx-community/gemma-4-E2B-it-qat-4bit,
///                         the matrix's device-floor model)
///   BENCHHOST_MAX_TOKENS  generation budget per cell (default 128)
///   BENCHHOST_RUNS        measured runs per cell (default 3)
final class DeviceRowProducerTests: XCTestCase {

    private static let defaultModelID = "mlx-community/gemma-4-E2B-it-qat-4bit"

    func testProduceLeaderboardRows() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("MLX needs a real Apple Silicon iPhone; the simulator cannot run this.")
        #else
        let environment = ProcessInfo.processInfo.environment
        let modelID = environment["BENCHHOST_MODEL_ID"] ?? Self.defaultModelID
        let maxTokens = Int(environment["BENCHHOST_MAX_TOKENS"] ?? "") ?? 128
        let runs = Int(environment["BENCHHOST_RUNS"] ?? "") ?? 3

        // Snapshot into the app container; a second run reuses the files.
        let hub = HubApi()
        let modelDirectory = try await hub.snapshot(
            from: modelID,
            matching: ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"]
        )

        // Jetsam guard: without an allocator ceiling MLX defaults to a cache
        // limit of 1.5x the working set, and the first iPhone night ended at
        // 41 minutes with "Test crashed with signal kill" — the OS memory
        // kill — during gemma-E2B's 4k cell. os_proc_available_memory is the
        // kernel's own statement of this process's remaining budget; 80% of
        // it leaves headroom for the transient peak the sampler can't cap.
        let availableBytes = Int64(os_proc_available_memory())
        let defaultCeiling = availableBytes > 0 ? Int64(Double(availableBytes) * 0.8) : 4 << 30
        let ceiling = Int64(environment["BENCHHOST_MEMORY_CEILING_BYTES"] ?? "") ?? defaultCeiling
        print("BENCHHOST memory ceiling \(ceiling) bytes (available \(availableBytes))")

        let configuration = BaselineConfiguration(
            modelDirectory: modelDirectory,
            modelID: URL(fileURLWithPath: modelID).lastPathComponent,
            // The device tiers: short and 4k. 16k on a phone is a memory
            // campaign of its own — add it deliberately, not by default.
            cells: [
                BaselineCell(promptTokens: 256, maxTokens: maxTokens),
                BaselineCell(promptTokens: 4096, maxTokens: maxTokens),
            ],
            runs: runs,
            warmup: 1,
            memoryCeilingBytes: ceiling
        )

        var lines: [String] = []
        try await BaselineProducer.run(
            configuration,
            emit: { record in
                var stamped = record
                stamped["platform"] = "ios"
                stamped["device"] = [
                    "model": Self.hardwareModel(),
                    "os": "iOS " + UIDevice.current.systemVersion,
                ]
                // Validity is stamped false here on purpose: the harness's
                // quiet-machine guard has no equivalent on the phone, so the
                // operator promotes rows to valid when the run conditions
                // (locked screen, on power, cool device) actually held.
                stamped["valid_for_leaderboard"] = false
                stamped["invalid_reason"] = "ios producer: operator must confirm run conditions"
                guard let data = try? JSONSerialization.data(withJSONObject: stamped, options: [.sortedKeys]),
                      let line = String(data: data, encoding: .utf8) else { return }
                lines.append(line)
                print("BENCHROW \(line)")
            },
            log: { line in print("BENCHHOST \(line)") }
        )

        XCTAssertFalse(lines.isEmpty, "the producer emitted no rows")

        let attachment = XCTAttachment(
            data: Data((lines.joined(separator: "\n") + "\n").utf8),
            uniformTypeIdentifier: "public.plain-text"
        )
        attachment.name = "mlxcat-bench-ios.jsonl"
        attachment.lifetime = .keepAlways
        add(attachment)
        #endif
    }

    private static func hardwareModel() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafeBytes(of: &systemInfo.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}
