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
/// every 2 s. These tests pin the contract that made that impossible.
final class AllocatorLimitTests: XCTestCase {
    private var savedCacheLimit: Int = 0
    private var savedMemoryLimit: Int = 0

    override func setUp() {
        super.setUp()
        savedCacheLimit = Memory.cacheLimit
        savedMemoryLimit = Memory.memoryLimit
    }

    override func tearDown() {
        Memory.cacheLimit = savedCacheLimit
        Memory.memoryLimit = savedMemoryLimit
        super.tearDown()
    }

    func test_ceilingActuallyBoundsTheAllocator() {
        let ceiling: Int64 = 24 * MemoryGuard.gibibyte
        let applied = MemoryGuard.applyAllocatorLimits(ceilingBytes: ceiling)

        XCTAssertEqual(applied.memoryLimit, ceiling)
        XCTAssertEqual(Int64(Memory.memoryLimit), ceiling, "memory limit was not pushed into MLX")
        XCTAssertLessThanOrEqual(
            applied.cacheLimit, ceiling,
            "a cache limit above the ceiling is the defect this guards: MLX defaults cacheLimit to memoryLimit"
        )
        XCTAssertEqual(Int64(Memory.cacheLimit), applied.cacheLimit, "cache limit was not pushed into MLX")
    }

    func test_noCeilingLeavesMLXDefaultsUntouched() {
        let before = (Memory.memoryLimit, Memory.cacheLimit)
        let applied = MemoryGuard.applyAllocatorLimits(ceilingBytes: 0)

        XCTAssertEqual(applied.memoryLimit, 0)
        XCTAssertEqual(applied.cacheLimit, 0)
        XCTAssertEqual(Memory.memoryLimit, before.0, "a zero ceiling must not silently clamp the allocator")
        XCTAssertEqual(Memory.cacheLimit, before.1)
    }

    func test_cacheLimitIsClampedIntoASaneBand() {
        // Tiny ceiling: the floor keeps buffer reuse alive (a 0-byte cache would
        // make every allocation a fresh malloc and cost throughput).
        let tiny = MemoryGuard.applyAllocatorLimits(ceilingBytes: 512 * 1_048_576)
        XCTAssertGreaterThanOrEqual(tiny.cacheLimit, 256 * 1_048_576)

        // Huge ceiling: the cap is what stops the pool growing to tens of GB.
        let huge = MemoryGuard.applyAllocatorLimits(ceilingBytes: 128 * MemoryGuard.gibibyte)
        XCTAssertLessThanOrEqual(huge.cacheLimit, 4 * MemoryGuard.gibibyte)
    }

    func test_environmentOverrideWinsSoThePolicyCanBeMeasured() {
        let override = MemoryGuard.cacheLimitOverrideFromEnvironment(["MLXCAT_CACHE_LIMIT_BYTES": "1_073_741_824"])
        XCTAssertEqual(override, 1_073_741_824)

        let applied = MemoryGuard.applyAllocatorLimits(
            ceilingBytes: 24 * MemoryGuard.gibibyte,
            cacheLimitOverrideBytes: override
        )
        XCTAssertEqual(applied.cacheLimit, 1_073_741_824)
        XCTAssertEqual(Int64(Memory.cacheLimit), 1_073_741_824)

        XCTAssertNil(MemoryGuard.cacheLimitOverrideFromEnvironment([:]))
        XCTAssertNil(MemoryGuard.cacheLimitOverrideFromEnvironment(["MLXCAT_CACHE_LIMIT_BYTES": "not-a-number"]))
        XCTAssertNil(MemoryGuard.cacheLimitOverrideFromEnvironment(["MLXCAT_CACHE_LIMIT_BYTES": "-1"]))
    }
}
