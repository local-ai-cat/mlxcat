import MLX
import XCTest

@testable import MLXCatNative

final class InklingAttentionTests: XCTestCase {
    func testReservedGlobalCacheKeepsTheSinkAndPreservesSnapshotsAcrossGrowth() throws {
        try MLXMetalRuntime.requireAvailable()
        let keys = arange(254).asType(.float32).reshaped(1, 1, 254, 1)
        var cache = InklingAttentionState(taps: 4)
        _ = cache.append(keys: keys, values: -keys, window: nil)
        func row(_ value: Float) -> MLXArray {
            MLXArray(value).reshaped(1, 1, 1, 1)
        }
        let first = cache.appendGlobalDecode(keys: row(254), values: row(-254))
        eval(first.0, first.1)
        var snapshot = cache

        let subsequent = cache.appendGlobalDecode(keys: row(255), values: row(-255))
        let continued = cache.appendGlobalDecode(keys: row(256), values: row(-256))
        let branch = snapshot.appendGlobalDecode(keys: row(777), values: row(-777))

        XCTAssertEqual(Array(first.0.asArray(Float.self).suffix(2)), [254, 0])
        XCTAssertEqual(Array(subsequent.0.asArray(Float.self).suffix(3)), [254, 255, 0])
        XCTAssertEqual(Array(continued.0.asArray(Float.self).suffix(4)), [254, 255, 256, 0])
        XCTAssertEqual(Array(continued.1.asArray(Float.self).suffix(4)), [-254, -255, -256, 0])
        XCTAssertEqual(Array(branch.0.asArray(Float.self).suffix(3)), [254, 777, 0])
        XCTAssertEqual(cache.keys?.dim(2), 257)
        XCTAssertEqual(snapshot.offset, 256)
    }

    func testRelativeMaskUsesBackwardDistanceAndAppendsTheSink() throws {
        try MLXMetalRuntime.requireAvailable()
        let relative = MLXArray.ones([1, 2, 1, 1])
        let projection = MLXArray([Float(3), 7]).reshaped(1, 2)
        let sink = MLXArray([Float(11), 13]).reshaped(1, 2, 1)
        let global = InklingAttention.composedMask(
            relative: relative, projection: projection, sink: sink, keyCount: 3, window: nil)
        XCTAssertEqual(global.asArray(Float.self), [7, 3, -1e30, 11, 0, 7, 3, 13])
        let local = InklingAttention.composedMask(
            relative: relative, projection: projection, sink: sink, keyCount: 3, window: 2)
        XCTAssertEqual(local.asArray(Float.self), [7, 3, -1e30, 11, -1e30, 7, 3, 13])
    }
}

extension InklingAttentionTests {
    func testMetalMaskAgreesWithTheComposedMaskOnStridedInputs() throws {
        try MLXMetalRuntime.requireAvailable()
        let relative = sin(arange(32).asType(.float32)).reshaped(1, 2, 4, 4).transposed(0, 2, 1, 3)
        let projection = cos(arange(32).asType(.float32)).reshaped(8, 4).transposed()
        let sink = arange(8).asType(.float32).reshaped(1, 4, 2)
        for window in [nil, 4] {
            let composed = InklingAttention.composedMask(
                relative: relative, projection: projection, sink: sink, keyCount: 7, window: window)
            let fused = InklingAttention.metalMask(
                relative: relative, projection: projection, sink: sink, keyCount: 7, window: window)
            XCTAssertEqual(composed.shape, fused.shape)
            XCTAssertLessThanOrEqual(abs(composed - fused).max().item(Float.self), 1e-5)
        }
    }

    func testMaskAB() throws {
        guard ProcessInfo.processInfo.environment["MLXCAT_INKLING_MASK_AB"] == "1" else {
            throw XCTSkip("Set MLXCAT_INKLING_MASK_AB=1 under an exclusive machine lease.")
        }
        try MLXMetalRuntime.requireAvailable()
        let relative = (sin(arange(128 * 24 * 16).asType(.float32)) * 0.1)
            .reshaped(1, 128, 24, 16).asType(.bfloat16)
        let projection = (cos(arange(16 * 1024).asType(.float32)) * 0.1)
            .reshaped(16, 1024).asType(.bfloat16)
        let sink = MLXArray.zeros([1, 128, 24], dtype: .bfloat16)
        eval(relative, projection, sink)
        func composed() -> MLXArray {
            InklingAttention.composedMask(relative: relative, projection: projection, sink: sink, keyCount: 8192, window: nil)
        }
        func fused() -> MLXArray {
            InklingAttention.metalMask(relative: relative, projection: projection, sink: sink, keyCount: 8192, window: nil)
        }
        eval(composed(), fused())
        let rounds = 20
        for (name, create) in [("composed", composed), ("fused", fused), ("fused", fused), ("composed", composed)] {
            Memory.clearCache()
            let before = Memory.activeMemory
            Memory.peakMemory = 0
            let start = Date()
            for _ in 0..<rounds { eval(create()) }
            let ms = Date().timeIntervalSince(start) * 1000 / Double(rounds)
            let peak = Memory.peakMemory - before
            print("Inkling mask \(name): \(ms) ms per call, \(peak) incremental peak bytes")
        }
    }
}
