import Foundation
import Metal
import MLX
import XCTest

final class GateProbeTests: XCTestCase {
    func testG1BFloat16UInt16BitReinterpretRoundTrip() throws {
        try MLXMetalRuntime.requireAvailable()

        Device.withDefaultDevice(.cpu) {
            let raw = MLXArray([UInt16(0x3f80), UInt16(0x4000), UInt16(0xbf80)])
            let bfloat = raw.view(dtype: .bfloat16, stream: .cpu)
            let roundTrip = bfloat.view(dtype: .uint16, stream: .cpu)

            XCTAssertEqual(roundTrip.dtype, .uint16)
            XCTAssertEqual(roundTrip.asArray(UInt16.self), [0x3f80, 0x4000, 0xbf80])
        }
    }

    func testG2SafetensorsMetadataRoundTrip() throws {
        try MLXMetalRuntime.requireAvailable()

        try Device.withDefaultDevice(.cpu) {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("safetensors")
            defer { try? FileManager.default.removeItem(at: url) }

            try save(
                arrays: ["tensor": MLXArray([Int32(1), Int32(2), Int32(3)])],
                metadata: [
                    "omlx_cache_format_version": "3",
                    "block_hash": "abc123",
                ],
                url: url,
                stream: .cpu
            )

            let (arrays, metadata) = try loadArraysAndMetadata(url: url, stream: .cpu)

            XCTAssertEqual(arrays["tensor"]?.asArray(Int32.self), [1, 2, 3])
            XCTAssertEqual(metadata["omlx_cache_format_version"], "3")
            XCTAssertEqual(metadata["block_hash"], "abc123")
        }
    }
}
