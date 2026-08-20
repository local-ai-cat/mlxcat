import Foundation

/// Live memory-pressure state, consumed by admission control and (optionally) a
/// scheduler that wants to pause new work while under pressure.
public enum MemoryPressureLevel: String, Sendable, Equatable {
    /// Below the soft watermark — normal operation.
    case ok
    /// Between the soft and hard watermarks — reclaim has run; new admissions pause.
    case soft
    /// At or above the hard watermark — reclaim has run; admissions are denied.
    case hard
}

/// Soft/hard watermarks derived from a ceiling, plus the ceiling itself.
///
/// The ceiling is the same `finalCeiling` the ``EnginePool`` is built with — on
/// iOS this is the ~6 GB flat per-app cap minus a reserve (see ``MemoryGuard``).
/// Unlike the macOS enforcer this carries no `vm_stat`/sysctl dynamic ceiling: the
/// iOS jetsam limit is effectively fixed, so a static ceiling with watermarks
/// below it is the whole model.
public struct MemoryWatchdogConfiguration: Sendable, Equatable {
    public let ceilingBytes: Int64
    public let softFraction: Double
    public let hardFraction: Double

    /// Conservative defaults: start trimming/evicting at 80% and deny at 92% of
    /// the ceiling, leaving headroom under the hard jetsam limit for the
    /// unaccounted transients (KV growth, SDPA activations, framework overhead)
    /// that the ceiling reserve alone does not cover.
    public static let defaultSoftFraction = 0.80
    public static let defaultHardFraction = 0.92

    public init(
        ceilingBytes: Int64,
        softFraction: Double = defaultSoftFraction,
        hardFraction: Double = defaultHardFraction
    ) {
        self.ceilingBytes = max(0, ceilingBytes)
        // Clamp into a sane band and keep soft <= hard so watermark math never inverts.
        let clampedSoft = min(max(softFraction, 0.1), 0.99)
        let clampedHard = min(max(hardFraction, clampedSoft), 0.99)
        self.softFraction = clampedSoft
        self.hardFraction = clampedHard
    }

    public var softBytes: Int64 { fraction(softFraction) }
    public var hardBytes: Int64 { fraction(hardFraction) }

    private func fraction(_ value: Double) -> Int64 {
        guard ceilingBytes > 0 else { return 0 }
        return Int64((Double(ceilingBytes) * value).rounded(.down))
    }
}

/// The reclaim ladder's two destructive steps, injected so the watchdog stays a
/// pure, testable unit with no dependency on MLX or the pool's generic type.
public protocol MemoryWatchdogReclaimer: Sendable {
    /// Step 1 — trim reclaimable prefix/paged caches toward freeing `targetBytes`.
    /// Non-destructive to loaded models. Returns bytes actually freed (best effort).
    func trimReclaimableCaches(targetBytes: Int64) async -> Int64

    /// Step 2 — evict idle (unpinned, not-in-use) models toward freeing
    /// `targetBytes`. Returns bytes freed (best effort).
    func evictIdleModels(targetBytes: Int64) async -> Int64
}

/// Closure-backed reclaimer so callers can wire the real MLX cache-clear + pool
/// eviction without a bespoke conforming type.
public struct ClosureMemoryWatchdogReclaimer: MemoryWatchdogReclaimer {
    private let trim: @Sendable (Int64) async -> Int64
    private let evict: @Sendable (Int64) async -> Int64

    public init(
        trimReclaimableCaches: @escaping @Sendable (Int64) async -> Int64,
        evictIdleModels: @escaping @Sendable (Int64) async -> Int64
    ) {
        self.trim = trimReclaimableCaches
        self.evict = evictIdleModels
    }

    public func trimReclaimableCaches(targetBytes: Int64) async -> Int64 {
        await trim(targetBytes)
    }

    public func evictIdleModels(targetBytes: Int64) async -> Int64 {
        await evict(targetBytes)
    }
}

public enum MemoryWatchdogError: Error, Equatable, Sendable {
    /// A new load/admission could not fit under the hard ceiling even after the
    /// trim + evict ladder ran. Maps to an HTTP 507 at the boundary.
    case admissionDenied(requiredBytes: Int64, currentBytes: Int64, ceilingBytes: Int64)
}

/// A live memory watchdog that mirrors the shape of omlx's
/// `process_memory_enforcer` — watchdog → trim → evict → block — but targets the
/// iOS constraint (a roughly fixed ~6 GB per-app cap with hard jetsam) instead of
/// macOS `vm_stat`/sysctl dynamic ceilings.
///
/// It is deliberately small and side-effect-injected: the current-usage `sampler`
/// and the ``MemoryWatchdogReclaimer`` are the only outside world it touches, so
/// it unit-tests without MLX or Metal.
///
/// Consult it in two places:
/// - On admission (a new model load / request), call ``checkAdmission(additionalBytes:)``.
/// - Periodically / between generation steps, call ``poll()``.
public actor MemoryWatchdog {
    public typealias UsageSampler = @Sendable () async -> Int64

    private let configuration: MemoryWatchdogConfiguration
    private let sampler: UsageSampler
    private let reclaimer: any MemoryWatchdogReclaimer

    /// Most recently observed pressure level; updated by ``poll()`` and
    /// ``checkAdmission(additionalBytes:)``. A scheduler can read this to pause
    /// admitting new work while it is not ``MemoryPressureLevel/ok``.
    public private(set) var pressureLevel: MemoryPressureLevel = .ok

    /// True while pressure is above the soft watermark — new admissions should pause.
    public private(set) var admissionsBlocked = false

    public init(
        configuration: MemoryWatchdogConfiguration,
        sampler: @escaping UsageSampler,
        reclaimer: any MemoryWatchdogReclaimer
    ) {
        self.configuration = configuration
        self.sampler = sampler
        self.reclaimer = reclaimer
    }

    /// Disabled when no ceiling was configured (mirrors the pool's `finalCeiling
    /// == 0` "guard off" convention). A disabled watchdog is a no-op.
    public var isEnabled: Bool { configuration.ceilingBytes > 0 }

    /// Runs the escalation ladder once and returns the resulting pressure level.
    ///
    /// - `ok` (usage < soft): no action.
    /// - otherwise: trim caches (step 1), re-sample; if still over soft, evict idle
    ///   models (step 2), re-sample; then classify and set the block flag (step 3).
    @discardableResult
    public func poll() async -> MemoryPressureLevel {
        guard isEnabled else {
            setLevel(.ok)
            return .ok
        }

        let soft = configuration.softBytes
        var usage = await sampler()
        if usage < soft {
            setLevel(.ok)
            return .ok
        }

        // Step 1: trim reclaimable caches.
        _ = await reclaimer.trimReclaimableCaches(targetBytes: usage - soft)
        usage = await sampler()

        // Step 2: evict idle models if trimming was not enough.
        if usage >= soft {
            _ = await reclaimer.evictIdleModels(targetBytes: usage - soft)
            usage = await sampler()
        }

        // Step 3: classify + block.
        let level = classify(usage)
        setLevel(level)
        return level
    }

    /// Consulted before admitting a new load/request. `additionalBytes` is the
    /// footprint the admission is expected to add (e.g. a model's estimated size);
    /// pass 0 for a request against an already-loaded model to just enforce the
    /// live ceiling.
    ///
    /// If the projected usage exceeds the hard watermark it runs the trim + evict
    /// ladder to make room; if it still does not fit, it throws
    /// ``MemoryWatchdogError/admissionDenied(requiredBytes:currentBytes:ceilingBytes:)``.
    public func checkAdmission(additionalBytes: Int64) async throws {
        guard isEnabled else { return }

        let hard = configuration.hardBytes
        var usage = await sampler()
        if usage + additionalBytes <= hard {
            refreshLevel(usage: usage)
            return
        }

        // Over the hard watermark: run the same ladder poll() uses to make room.
        _ = await reclaimer.trimReclaimableCaches(targetBytes: (usage + additionalBytes) - hard)
        usage = await sampler()
        if usage + additionalBytes > hard {
            _ = await reclaimer.evictIdleModels(targetBytes: (usage + additionalBytes) - hard)
            usage = await sampler()
        }

        refreshLevel(usage: usage)
        if usage + additionalBytes > hard {
            throw MemoryWatchdogError.admissionDenied(
                requiredBytes: additionalBytes,
                currentBytes: usage,
                ceilingBytes: hard
            )
        }
    }

    private func classify(_ usage: Int64) -> MemoryPressureLevel {
        if usage < configuration.softBytes { return .ok }
        if usage < configuration.hardBytes { return .soft }
        return .hard
    }

    private func refreshLevel(usage: Int64) {
        setLevel(classify(usage))
    }

    private func setLevel(_ level: MemoryPressureLevel) {
        pressureLevel = level
        admissionsBlocked = level != .ok
    }
}
