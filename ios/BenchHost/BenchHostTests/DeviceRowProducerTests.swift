import Hub
import MLXCatBaselineKit
import os
import XCTest

/// Produces iOS leaderboard rows on a REAL iPhone — mlxcat-bench/1 JSONL via
/// the same `BaselineProducer` the Mac CLI uses, so the two arms (mlxcat
/// in-process vs the raw-TokenIterator legacy-LLMEvaluator mechanism), the
/// prompt corpus, and the footprint metric are identical across platforms.
///
/// Models download once into the host app's container (HubApi snapshot) and
/// are reused on later runs. Rows land as an XCTAttachment named
/// `mlxcat-bench-ios.jsonl` AND as `BENCHROW` console lines — the console copy
/// is what survives a crash (the driver recovers it), the attachment is the
/// tidy path when the whole session completes.
///
/// Config via the xcodebuild ENVIRONMENT (TEST_RUNNER_ prefix):
///   BENCHHOST_MODEL_ID    HF repo id, or a COMMA-SEPARATED list (default
///                         mlx-community/gemma-4-E2B-it-qat-4bit). The whole
///                         list runs inside ONE app session on purpose: a
///                         passcode phone can auto-lock in the gap between
///                         separate xcodebuild launches, and a locked device
///                         fails the next install — one launch, one night.
///   BENCHHOST_MAX_TOKENS  generation budget per cell (default 128)
///   BENCHHOST_RUNS        measured runs per cell (default 3)
///   BENCHHOST_MEMORY_CEILING_BYTES  allocator ceiling override
final class DeviceRowProducerTests: XCTestCase {

    private static let defaultModelID = "mlx-community/gemma-4-E2B-it-qat-4bit"

    /// Rows collected across the @Sendable emit callback.
    private final class RowBox: @unchecked Sendable {
        private let lock = NSLock()
        private var rows: [String] = []
        func append(_ line: String) {
            lock.lock()
            rows.append(line)
            lock.unlock()
        }
        var all: [String] {
            lock.lock()
            defer { lock.unlock() }
            return rows
        }
    }

    func testProduceLeaderboardRows() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("MLX needs a real Apple Silicon iPhone; the simulator cannot run this.")
        #else
        let environment = ProcessInfo.processInfo.environment
        let modelIDs = (environment["BENCHHOST_MODEL_ID"] ?? Self.defaultModelID)
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let maxTokens = Int(environment["BENCHHOST_MAX_TOKENS"] ?? "") ?? 128
        let runs = Int(environment["BENCHHOST_RUNS"] ?? "") ?? 3

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

        let box = RowBox()
        let deviceModel = Self.hardwareModel()
        let osVersion = await MainActor.run { "iOS " + UIDevice.current.systemVersion }

        for modelID in modelIDs {
            print("BENCHHOST model \(modelID) starting")
            // Snapshot into the app container; a second run reuses the files.
            let hub = HubApi()
            let modelDirectory = try await hub.snapshot(
                from: modelID,
                matching: ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"]
            )

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

            try await BaselineProducer.run(
                configuration,
                emit: { record in
                    var stamped = record
                    stamped["platform"] = "ios"
                    stamped["device"] = [
                        "model": deviceModel,
                        "os": osVersion,
                    ]
                    // Validity is stamped false on purpose: the harness's
                    // quiet-machine guard has no equivalent on the phone, so
                    // the operator promotes rows to valid when the run
                    // conditions (untouched screen, on power, cool device)
                    // actually held.
                    stamped["valid_for_leaderboard"] = false
                    stamped["invalid_reason"] = "ios producer: operator must confirm run conditions"
                    guard
                        let data = try? JSONSerialization.data(
                            withJSONObject: stamped, options: [.sortedKeys]),
                        let line = String(data: data, encoding: .utf8)
                    else { return }
                    box.append(line)
                    print("BENCHROW \(line)")
                },
                log: { line in print("BENCHHOST \(line)") }
            )
        }

        let lines = box.all
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
