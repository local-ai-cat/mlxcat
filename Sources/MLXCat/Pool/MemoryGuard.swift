import Foundation

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
