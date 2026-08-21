import Foundation
import MLX
import MLXCat
import XCTest

/// The allocator limits must actually bound MLX, not just be recorded.
///
/// Context: a configured memory ceiling was measured to bound nothing —
/// gpt-oss-20b at a 16k prompt peaked at 35.8 GiB against a 24 GiB ceiling on an
/// M4 Pro (2026-08-21). MLX's `cacheLimit` defaults to its `memoryLimit`, which
/// itself defaults to 1.5x the device's recommended working set, so the
/// reclaimable pool was free to grow to ~54 GB while the watchdog trimmed only
/// every 2 s.
///
/// The policy tests run anywhere. Only the test that pushes values INTO MLX needs
/// a Metal runtime: reading `Memory.cacheLimit` without a loaded metallib aborts
/// the process instead of failing a test, which is how the first version of this
/// file turned a hosted CI runner red.
final class AllocatorLimitTests: XCTestCase {

    // MARK: - Policy (no MLX, no GPU)

    func test_ceilingIsTheMemoryLimitAndCacheIsASliceOfIt() {
        let ceiling: Int64 = 24 * MemoryGuard.gibibyte
        let planned = MemoryGuard.plannedAllocatorLimits(ceilingBytes: ceiling)

        XCTAssertEqual(planned.memoryLimit, ceiling)
        XCTAssertLessThanOrEqual(
            planned.cacheLimit, ceiling,
            "a cache limit above the ceiling is the defect this guards: MLX defaults cacheLimit to memoryLimit"
        )
    }

    func test_noCeilingPlansNothing() {
        let planned = MemoryGuard.plannedAllocatorLimits(ceilingBytes: 0)
        XCTAssertEqual(planned.memoryLimit, 0, "a zero ceiling must not silently clamp the allocator")
        XCTAssertEqual(planned.cacheLimit, 0)
    }

    func test_cacheLimitIsClampedIntoASaneBand() {
        // Tiny ceiling: the floor keeps buffer reuse alive (a 0-byte cache would
        // make every allocation a fresh malloc and cost throughput).
        let tiny = MemoryGuard.plannedAllocatorLimits(ceilingBytes: 512 * 1_048_576)
        XCTAssertGreaterThanOrEqual(tiny.cacheLimit, 256 * 1_048_576)

        // Huge ceiling: the cap is what stops the pool growing to tens of GB.
        let huge = MemoryGuard.plannedAllocatorLimits(ceilingBytes: 128 * MemoryGuard.gibibyte)
        XCTAssertLessThanOrEqual(huge.cacheLimit, 4 * MemoryGuard.gibibyte)
    }

    func test_environmentOverrideWinsSoThePolicyCanBeMeasured() {
        let override = MemoryGuard.cacheLimitOverrideFromEnvironment(["MLXCAT_CACHE_LIMIT_BYTES": "1_073_741_824"])
        XCTAssertEqual(override, 1_073_741_824)

        let planned = MemoryGuard.plannedAllocatorLimits(
            ceilingBytes: 24 * MemoryGuard.gibibyte,
            cacheLimitOverrideBytes: override
        )
        XCTAssertEqual(planned.cacheLimit, 1_073_741_824)

        XCTAssertNil(MemoryGuard.cacheLimitOverrideFromEnvironment([:]))
        XCTAssertNil(MemoryGuard.cacheLimitOverrideFromEnvironment(["MLXCAT_CACHE_LIMIT_BYTES": "not-a-number"]))
        XCTAssertNil(MemoryGuard.cacheLimitOverrideFromEnvironment(["MLXCAT_CACHE_LIMIT_BYTES": "-1"]))
    }

    // MARK: - Actually reaching MLX (needs Metal)

    func test_limitsAreReallyPushedIntoMLX() throws {
        try MLXMetalRuntime.requireAvailable()

        let savedCacheLimit = Memory.cacheLimit
        let savedMemoryLimit = Memory.memoryLimit
        defer {
            Memory.cacheLimit = savedCacheLimit
            Memory.memoryLimit = savedMemoryLimit
        }

        let ceiling: Int64 = 24 * MemoryGuard.gibibyte
        let applied = MemoryGuard.applyAllocatorLimits(ceilingBytes: ceiling)

        XCTAssertEqual(Int64(Memory.memoryLimit), ceiling, "memory limit was not pushed into MLX")
        XCTAssertEqual(Int64(Memory.cacheLimit), applied.cacheLimit, "cache limit was not pushed into MLX")
    }
}
