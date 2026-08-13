import Cmlx
import Foundation
import MLX

/// Experimental (`MLXSERVE_ENCODE_THREAD=1`): a dedicated thread that owns the
/// pipelined-decode `asyncEval`, moving the per-token Metal kernel-encode cost
/// (~1.6ms/step at width 1, measured dispatch-bound at ~4µs/dispatch) off the
/// decode loop thread. Single-stream by construction — this thread only
/// *encodes* graphs the loop thread already built on the default stream, so no
/// cross-stream events exist (the failure mode that killed the multi-stream
/// prototype; see planning/runs/2026-08-13-mlx-encode-loop/AUDIT-track-a.md).
///
/// Correctness contract:
/// - Jobs run the PUBLIC `asyncEval`, so they hold mlx-swift's `evalLock`
///   like every other eval in the process: unrelated in-process MLX users
///   (tests, transcription, other engines) serialize against in-flight
///   encoding instead of corrupting the stream's CommandEncoder. The decode
///   loop's overlap window (emit, HTTP streaming, next-step graph build)
///   takes no MLX lock, so the overlap survives the locking.
/// - The generator still `flush()`es before waiting on submitted arrays and
///   before every non-pipelined eval path, keeping its own two threads off
///   the encoder at once and its waits on already-scheduled arrays.
/// - Known accepted risk (experiment-only): the loop thread builds the next
///   step's graph while this thread schedules the previous one; MLX array
///   status fields are not atomic. Gated behind the env flag until the
///   exactness + suite gates say otherwise.
final class EncodeThread {
    static let enabled =
        ProcessInfo.processInfo.environment["MLXSERVE_ENCODE_THREAD"] == "1"

    /// Diagnostic only (`MLXSERVE_ENCODE_THREAD_RAW=1`): encode via Cmlx
    /// WITHOUT `evalLock`, to attribute build-phase contention to the Swift
    /// lock vs the C++ core. Unsafe outside a single-request harness where
    /// the generator owns every MLX call.
    static let rawUnlocked =
        ProcessInfo.processInfo.environment["MLXSERVE_ENCODE_THREAD_RAW"] == "1"

    private let condition = NSCondition()
    private var pending: [[MLXArray]] = []
    private var busy = false
    private var stopped = false

    init() {
        let thread = Thread { [weak self] in self?.run() }
        thread.name = "mlxserve.encode"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    deinit {
        stop()
    }

    /// Queue the arrays for scheduling with MLX. The arrays' graphs must be
    /// fully built; this call never blocks.
    func submit(_ arrays: [MLXArray]) {
        condition.lock()
        pending.append(arrays)
        condition.signal()
        condition.unlock()
    }

    /// Block until every submitted job has been handed to MLX. After this
    /// returns, waiting on the submitted arrays cannot re-enter encoding.
    func flush() {
        condition.lock()
        while busy || !pending.isEmpty {
            condition.wait()
        }
        condition.unlock()
    }

    func stop() {
        condition.lock()
        stopped = true
        condition.signal()
        condition.unlock()
    }

    private func run() {
        while true {
            condition.lock()
            while pending.isEmpty && !stopped {
                condition.wait()
            }
            if pending.isEmpty {
                condition.unlock()
                return
            }
            let job = pending.removeFirst()
            busy = true
            condition.unlock()

            autoreleasepool {
                if Self.rawUnlocked {
                    withExtendedLifetime(job) {
                        let vector = mlx_vector_array_new_data(job.map(\.ctx), job.count)
                        mlx_async_eval(vector)
                        mlx_vector_array_free(vector)
                    }
                } else {
                    asyncEval(job)
                }
            }

            condition.lock()
            busy = false
            condition.broadcast()
            condition.unlock()
        }
    }
}
