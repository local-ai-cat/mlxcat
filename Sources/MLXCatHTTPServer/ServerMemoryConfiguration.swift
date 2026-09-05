import Foundation
import MLX
import MLXCat

func applyModelIDOverrideIfNeeded(
    _ discovered: [String: DiscoveredModel],
    override: String?
) throws -> [String: DiscoveredModel] {
    guard let override else {
        return discovered
    }
    guard discovered.count == 1, let model = discovered.values.first else {
        throw ServerConfigError.invalidArgument("--model-id can only be used with a single discovered model")
    }
    return [
        override: DiscoveredModel(
            id: override,
            modelURL: model.modelURL,
            estimatedSize: model.estimatedSize
        )
    ]
}

func finalMemoryCeiling(
    overrideBytes: Int64?,
    tier: MemoryGuardTier?
) -> MemoryGuard.EffectiveCeiling {
    let recommendedWorkingSet = Int64(GPU.maxRecommendedWorkingSetBytes() ?? 0)
    return MemoryGuard.effectiveCeiling(
        overrideBytes: overrideBytes,
        recommendedWorkingSetBytes: recommendedWorkingSet,
        tier: tier
    )
}

func nonNegativeInt64(_ value: String) -> Int64? {
    let normalized = value.replacingOccurrences(of: "_", with: "")
    guard let parsed = Int64(normalized), parsed >= 0 else {
        return nil
    }
    return parsed
}

let memoryWatchdogPollInterval: TimeInterval = 2

/// Builds the live memory watchdog when a ceiling is configured. The usage sampler
/// reads MLX's own accounting (active weights + reclaimable cache); the reclaimer's
/// trim step clears the MLX cache pool and its evict step unloads idle pool models.
func makeMemoryWatchdog<Loader: EnginePoolModelLoader>(
    ceilingBytes: Int64,
    pool: EnginePool<Loader>
) -> MemoryWatchdog? {
    guard ceilingBytes > 0 else { return nil }
    return MemoryWatchdog(
        configuration: MemoryWatchdogConfiguration(ceilingBytes: ceilingBytes),
        sampler: { Int64(Memory.activeMemory + Memory.cacheMemory) + (await pool.nonMLXAccountedBytes()) },
        reclaimer: ClosureMemoryWatchdogReclaimer(
            trimReclaimableCaches: { _ in
                let before = Memory.cacheMemory
                Memory.clearCache()
                return Int64(max(0, before - Memory.cacheMemory))
            },
            evictIdleModels: { target in
                await pool.reclaimIdleModels(targetBytes: target)
            }
        )
    )
}

func startMemoryWatchdogPoll(watchdog: MemoryWatchdog, interval: TimeInterval) {
    let sleepNanoseconds = UInt64(max(interval, 1) * 1_000_000_000)
    Task.detached {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: sleepNanoseconds)
            _ = await watchdog.poll()
        }
    }
}

func startIdleSweep<Loader: EnginePoolModelLoader>(
    pool: EnginePool<Loader>,
    interval: TimeInterval
) {
    let sleepNanoseconds = UInt64(max(interval, 1) * 1_000_000_000)
    Task.detached {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: sleepNanoseconds)
            let unloaded = await pool.sweepIdleModels()
            if !unloaded.isEmpty {
                print("Idle timeout unloaded model(s): \(unloaded.joined(separator: ", "))")
                fflush(stdout)
            }
        }
    }
}
