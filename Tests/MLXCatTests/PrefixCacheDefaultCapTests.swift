import Foundation
@testable import MLXCatNative
import XCTest

final class PrefixCacheDefaultCapTests: XCTestCase {
    func testDefaultCapIsBoundedWhenEnvUnset() throws {
        if ProcessInfo.processInfo.environment["MLXSERVE_PREFIX_CACHE_MAX_BYTES"] != nil {
            throw XCTSkip("MLXSERVE_PREFIX_CACHE_MAX_BYTES is set; default-path not testable here.")
        }
        let cap = NativeModelEngine.prefixCacheMaxBytes()
        XCTAssertGreaterThanOrEqual(cap, 512 << 20, "default prefix cap fell below 512 MB")
        XCTAssertLessThanOrEqual(cap, 4 << 30, "default prefix cap exceeds 4 GB")
    }
}
