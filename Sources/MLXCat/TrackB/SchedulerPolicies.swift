import Foundation

extension Scheduler {
    public struct PressureSnapshot: Sendable, Equatable {
        public let runningUIDs: [String]
        public let waitingCount: Int
        public let admissionInProgressUID: String?

        public init(runningUIDs: [String], waitingCount: Int, admissionInProgressUID: String?) {
            self.runningUIDs = runningUIDs
            self.waitingCount = waitingCount
            self.admissionInProgressUID = admissionInProgressUID
        }
    }

    public struct PressurePolicy: Sendable {
        public let shouldPreempt: @Sendable (PressureSnapshot) -> Bool

        public init(shouldPreempt: @escaping @Sendable (PressureSnapshot) -> Bool) {
            self.shouldPreempt = shouldPreempt
        }

        public static let disabled = PressurePolicy { _ in false }
    }

    /// Whether the last prompt token is prefilled alone so the one forward whose
    /// logits we read is `[1, 1, vocab]` rather than `[1, chunk, vocab]`.
    ///
    /// mlx-lm always does this — it loops `while y.size > 1` and hands the single
    /// remaining token to the step that samples
    /// (`guest/mlx-lm/mlx_lm/generate.py:580-587`). It is safe for them because
    /// that remaining token goes through `_step()`, the SAME single-token decode
    /// path every later token uses.
    ///
    /// Ours does not: the extra one-token forward still goes through the PREFILL
    /// call site. A rotating (sliding-window) KV cache has separate multi-token
    /// and single-token update paths, so that extra prefill-path `S == 1` update
    /// desynchronises the ring. Measured on gpt-oss-20b (window 128):
    /// `SlidingWindowBatchIntegrationTests` went from 3/3 green to 60 of 160
    /// tokens diverging from serial with a sustained run of 37, starting at
    /// token 97 — and every non-windowed gate stayed green, which is what
    /// localises it. Bisected: disabling this alone restores 3/3.
    ///
    /// So it is capability-gated rather than reverted: windowed caches keep the
    /// old path, everything else takes the smaller tensor.
    /// `MLXCAT_PREFILL_LAST_TOKEN_ALONE=always|never` forces it either way, so
    /// the trade-off can be measured on a windowed model rather than argued —
    /// `always` is how you reproduce the divergence above.
    static func prefillsLastTokenAlone(
        usesWindowedKVCache: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        switch environment["MLXCAT_PREFILL_LAST_TOKEN_ALONE"]?.lowercased() {
        case "always", "1", "true": return true
        case "never", "0", "false": return false
        default: return !usesWindowedKVCache
        }
    }

    /// When the MLX buffer cache is handed back to the OS.
    ///
    /// MLX keeps freed Metal buffers in a cache and does not return them on its
    /// own, so a server that once served a 16k-token prompt keeps that footprint
    /// resident forever. Measured on the 2026-08-23 board: gemma-4-E2B's 16k
    /// tier peaked at 8.39 GiB, and every later cell in the same process
    /// reported ~7.8 GiB no matter how little it actually needed — the same
    /// process, hours later, still holding gigabytes for nobody. On a laptop
    /// that is the whole complaint.
    ///
    /// The references both do more than we did, which was to clear between
    /// prefill chunks and nowhere else:
    ///
    ///   * mlx-lm clears every 512 decode steps
    ///     (`guest/mlx-lm/mlx_lm/generate.py:1779`), so a long generation does
    ///     not accumulate.
    ///   * omlx clears on engine transitions
    ///     (`guest/omlx/omlx/engine_core.py:85`).
    ///
    /// Both knobs are on by default and both are switchable, because the cost is
    /// real and worth measuring rather than asserting: a cleared cache means the
    /// next allocation goes to the Metal allocator instead of a free list.
    /// `MLXCAT_DECODE_CLEAR_CACHE_STEPS=0` disables the periodic clear (any other
    /// non-negative integer sets the interval); `MLXCAT_IDLE_CLEAR_CACHE=0`
    /// disables the drain-to-idle clear.
    public struct CacheReleasePolicy: Sendable, Equatable {
        /// Clear every N decode steps. Zero never clears.
        public var decodeStepInterval: Int
        /// Clear once each time the scheduler drains to fully idle.
        public var releasesWhenIdle: Bool

        public init(decodeStepInterval: Int = 512, releasesWhenIdle: Bool = true) {
            self.decodeStepInterval = max(0, decodeStepInterval)
            self.releasesWhenIdle = releasesWhenIdle
        }

        /// mlx-lm's interval, plus the idle release it has no need for (its
        /// server does not hold one process across unrelated workloads).
        public static let `default` = CacheReleasePolicy()
        public static let never = CacheReleasePolicy(decodeStepInterval: 0, releasesWhenIdle: false)

        public static func fromEnvironment(
            _ environment: [String: String] = ProcessInfo.processInfo.environment
        ) -> CacheReleasePolicy {
            var policy = CacheReleasePolicy.default
            if let raw = environment["MLXCAT_DECODE_CLEAR_CACHE_STEPS"],
               let steps = Int(raw.trimmingCharacters(in: .whitespaces)), steps >= 0 {
                policy.decodeStepInterval = steps
            }
            switch environment["MLXCAT_IDLE_CLEAR_CACHE"]?.lowercased() {
            case "0", "false", "never": policy.releasesWhenIdle = false
            case "1", "true", "always": policy.releasesWhenIdle = true
            default: break
            }
            return policy
        }
    }

    /// Quantized KV is supported only in a serialized row, outside prefix stores.
    static func configuredSerializationPolicy(
        kvQuantization: KVQuantizationPolicy, serializationPolicy: SerializationPolicy?, serializedDecode: Bool
    ) -> SerializationPolicy {
        // Stage 1 of KV quantization is single-stream, and the gate is here
        // rather than in a comment. A quantized row that reaches
        // `BatchLayerCache` is misrouted by its state array count: rows of
        // different lengths throw, and — worse — two rows of the SAME length
        // pass shape validation and then have their `[step, offset, groupSize,
        // bits]` metaState read as slot metadata, which is silent wrong output.
        // `.always` keeps every row alone, so that path is unreachable rather
        // than merely unlikely.
        //
        // The prefix stores are excluded for a milder reason: both require the
        // 2-array keys/values schema and would throw on every publish, which the
        // scheduler catches and logs. That is not a crash, it is a cache that
        // has silently stopped working while filling the log — so it is turned
        // off explicitly instead.
        if kvQuantization.isEnabled { return .always }
        return serializationPolicy ?? (serializedDecode ? .always : .never)
    }
    /// Snapshotting a row's KV cache is not free: `extractCache` slices every
    /// layer and the stored copy holds live references to the KV arrays, which
    /// blocks MLX buffer donation on the next in-place cache update. Publishing
    /// on every decode step measured a ~14% single-request throughput loss, so
    /// mid-generation publishes only fire once this many new tokens accumulate.
    /// The first publish (prompt prefix) and the finish/preemption publishes
    /// are exempt: callers pass `minimumNewTokens: 1` there.
    static let midGenerationPublishInterval = 256

    static func refusesAdmission(
        policy: SerializationPolicy, running: [String: RunningRequest], nextRequest: Request?, busy: Bool
    ) -> Bool {
        // Some model architectures still derive RoPE position ids or
        // shared-KV offsets from scalar cache.offset, so mixed-offset rows
        // must not share a decode batch.
        //
        // `.always` is the blunt version of that and it is expensive: on
        // gemma-4-E2B it costs 50x TTFT and 2.17x aggregate throughput at c4
        // (`docs/COMPETITIVE.md`). `.multimodalOnly` is the same protection
        // aimed at the rows that actually need it — a request carrying
        // images, video or audio has a token count that varies per row, and
        // that is what the scalar offset gets wrong. Text rows of these same
        // families batch correctly: they pass logit-level invariance more
        // cleanly than the model the gate was pinned to, and pass
        // batched-vs-serial token equality. Ragged IMAGE rows do not — all
        // three rows stop at the same early token (measured 2026-08-23).
        // Computed from `running`, not tracked alongside it: RunningRequest
        // retains its Request, so there is no second copy of this state to
        // drift out of sync on a cancel or an error path.
        if busy, policy.refusesToJoin(running: running.values.map(\.request)) {
            return true
        }
        if busy, let next = nextRequest, policy.requiresSolitude(next) {
            // A row that needs solitude waits for the batch to drain rather
            // than joining it. Head-of-line blocking is deliberate: admitting
            // it out of order would starve it behind an unbounded text stream.
            return true
        }
        return false
    }
}
