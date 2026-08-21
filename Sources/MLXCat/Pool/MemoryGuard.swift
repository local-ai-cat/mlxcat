import Foundation
import MLX

public enum MemoryGuardTier: String, CaseIterable, Sendable {
    case safe
    case balanced
    case aggressive
}

public enum MemoryGuard {
    public static let gibibyte: Int64 = 1_073_741_824
    public static let smallSystemThresholdBytes: Int64 = 24 * gibibyte

    private static let safeReserveBytes: Int64 = 8 * gibibyte
    private static let balancedReserveBytes: Int64 = 6 * gibibyte
    private static let aggressiveReserveBytes: Int64 = 4 * gibibyte
    private static let smallSystemReserveBytes: Int64 = 4 * gibibyte

    public struct EffectiveCeiling: Equatable, Sendable {
        public let bytes: Int64
        public let source: String

        public init(bytes: Int64, source: String) {
            self.bytes = bytes
            self.source = source
        }
    }

    public static func finalCeiling(
        recommendedWorkingSetBytes: Int64,
        physicalMemoryBytes: Int64 = Int64(ProcessInfo.processInfo.physicalMemory),
        tier: MemoryGuardTier?
    ) -> Int64 {
        guard let tier, recommendedWorkingSetBytes > 0 else {
            return 0
        }

        let reserve: Int64
        if physicalMemoryBytes > 0, physicalMemoryBytes < smallSystemThresholdBytes {
            reserve = smallSystemReserveBytes
        } else {
            reserve = reserveBytes(for: tier)
        }

        return max(0, recommendedWorkingSetBytes - reserve)
    }

    public static func effectiveCeiling(
        overrideBytes: Int64?,
        recommendedWorkingSetBytes: Int64,
        physicalMemoryBytes: Int64 = Int64(ProcessInfo.processInfo.physicalMemory),
        tier: MemoryGuardTier?
    ) -> EffectiveCeiling {
        if let overrideBytes, overrideBytes > 0 {
            return EffectiveCeiling(bytes: overrideBytes, source: "override")
        }

        let tierCeiling = finalCeiling(
            recommendedWorkingSetBytes: recommendedWorkingSetBytes,
            physicalMemoryBytes: physicalMemoryBytes,
            tier: tier
        )
        if tier != nil {
            return EffectiveCeiling(bytes: tierCeiling, source: "tier")
        }

        return EffectiveCeiling(bytes: 0, source: "off")
    }

    /// Bounds the MLX allocator itself, instead of relying on the watchdog alone.
    ///
    /// MLX defaults `Memory.memoryLimit` to 1.5x the device's recommended working
    /// set, and `Memory.cacheLimit` defaults to the memory limit — on a 48 GB Mac
    /// that permits roughly 54 GB of reclaimable buffer cache. A configured
    /// ceiling therefore bounded nothing: `MemoryWatchdog` polls every 2 s and can
    /// only trim AFTER the fact, while a long prefill allocates attention scratch
    /// whose shape grows every chunk, so the pool never reuses a buffer and grows
    /// monotonically. Measured 2026-08-21 on an M4 Pro: gpt-oss-20b at a 16k prompt
    /// peaked at 35.8 GiB against a 24 GiB ceiling. Upstream's own guidance is to
    /// set a much lower cache limit for long inference runs.
    ///
    /// Called once, at engine/server construction, when a ceiling is known.
    /// Returns what was applied so the caller can log it (0 = left at MLX default).
    /// The limits a ceiling implies — pure arithmetic, no MLX calls.
    ///
    /// Split out from ``applyAllocatorLimits(ceilingBytes:cacheLimitOverrideBytes:)``
    /// so the clamping policy (the part that regresses) is testable on any machine.
    /// Touching `Memory.cacheLimit` requires a loaded Metal library, which a hosted
    /// CI runner does not have — reading it there aborts the process rather than
    /// failing a test.
    public static func plannedAllocatorLimits(
        ceilingBytes: Int64,
        cacheLimitOverrideBytes: Int64? = nil
    ) -> (memoryLimit: Int64, cacheLimit: Int64) {
        guard ceilingBytes > 0 else { return (0, 0) }

        // The ceiling is the whole budget (weights + KV + scratch). The reclaimable
        // cache is only the part that grows without bound, so it gets a slice of it,
        // clamped: too small starves buffer reuse and costs throughput, too large
        // reintroduces the cliff. An explicit override exists so the value can be
        // A/B-measured by bench/run.py rather than argued about.
        let cacheLimit: Int64
        if let cacheLimitOverrideBytes, cacheLimitOverrideBytes >= 0 {
            cacheLimit = cacheLimitOverrideBytes
        } else {
            cacheLimit = min(max(ceilingBytes / 4, 256 * 1_048_576), 4 * gibibyte)
        }
        return (ceilingBytes, cacheLimit)
    }

    /// Pushes ``plannedAllocatorLimits(ceilingBytes:cacheLimitOverrideBytes:)`` into
    /// MLX. Requires a working Metal runtime; a zero ceiling is a no-op.
    @discardableResult
    public static func applyAllocatorLimits(
        ceilingBytes: Int64,
        cacheLimitOverrideBytes: Int64? = nil
    ) -> (memoryLimit: Int64, cacheLimit: Int64) {
        let planned = plannedAllocatorLimits(
            ceilingBytes: ceilingBytes,
            cacheLimitOverrideBytes: cacheLimitOverrideBytes
        )
        guard planned.memoryLimit > 0 else { return planned }

        Memory.memoryLimit = Int(planned.memoryLimit)
        Memory.cacheLimit = Int(planned.cacheLimit)
        return planned
    }

    /// Reads the optional cache-limit override (bytes) used to A/B the policy above.
    public static func cacheLimitOverrideFromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int64? {
        guard let raw = environment["MLXCAT_CACHE_LIMIT_BYTES"],
            let parsed = Int64(raw.replacingOccurrences(of: "_", with: "")),
            parsed >= 0
        else {
            return nil
        }
        return parsed
    }

    public static func reserveBytes(for tier: MemoryGuardTier) -> Int64 {
        switch tier {
        case .safe:
            return safeReserveBytes
        case .balanced:
            return balancedReserveBytes
        case .aggressive:
            return aggressiveReserveBytes
        }
    }
}
