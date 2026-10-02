import Foundation
import MLX
import MLXCat
@testable import MLXCatNative
import XCTest

/// Where an evicted model's bytes are when the watchdog re-samples.
///
/// The live sampler reads `Memory.activeMemory + Memory.cacheMemory`. This loads a
/// real model through the pool, evicts it the way the watchdog's step 2 does, and
/// reads that sum before and after one more `Memory.clearCache()`: the gap is
/// what `MLXCAT_WATCHDOG_TRIM_AFTER_EVICT` exists to close.
///
/// Measured 2026-10-02: Llama-3.2-3B-4bit 1723 MiB loaded, 1723 after eviction
/// (all of it on the free list), 0 after the clear; Qwen3.5-4B 2894 / 2894 / 0.
final class EvictionFreeListProbeTests: XCTestCase {

    func testEvictedWeightsAreStillCountedUntilTheNextClear() async throws {
        try MLXMetalRuntime.requireAvailable()
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to probe eviction accounting.")
        }
        let modelID = resolution.url.lastPathComponent
        let pool = EnginePool(
            models: [
                modelID: DiscoveredModel(
                    id: modelID, modelURL: resolution.url, estimatedSize: 2_000_000_000)
            ],
            loader: NativeModelLoader(maxConcurrentRequests: 1),
            finalCeiling: 32_000_000_000
        )

        Memory.clearCache()
        let baseline = Memory.activeMemory + Memory.cacheMemory
        _ = try await pool.load(modelID)
        Memory.clearCache()
        let loaded = Memory.activeMemory + Memory.cacheMemory

        _ = await pool.reclaimIdleModels(targetBytes: 1)
        let afterEvict = Memory.activeMemory + Memory.cacheMemory
        let cachedAfterEvict = Memory.cacheMemory

        Memory.clearCache()
        let afterTrim = Memory.activeMemory + Memory.cacheMemory

        let mib = 1_048_576
        print(
            "EVICTPROBE baseline=\(baseline / mib) loaded=\(loaded / mib) "
                + "afterEvict=\(afterEvict / mib) (cache \(cachedAfterEvict / mib)) "
                + "afterTrim=\(afterTrim / mib) MiB")

        let modelBytes = loaded - baseline
        XCTAssertGreaterThan(modelBytes, 64 * mib, "the model never became resident; nothing to measure")
        // The gap itself. If this starts failing, eviction now returns the weights
        // before the loader's clear, and MLXCAT_WATCHDOG_TRIM_AFTER_EVICT may no
        // longer be needed.
        XCTAssertGreaterThan(
            afterEvict, loaded - modelBytes / 2,
            "eviction alone returned the model's bytes; the free-list gap is gone")
        XCTAssertGreaterThan(cachedAfterEvict, modelBytes / 2, "the evicted bytes are not on the free list")
        XCTAssertLessThan(
            afterTrim, loaded - modelBytes / 2,
            "eviction plus a clear did not return the model's bytes")
    }
}
