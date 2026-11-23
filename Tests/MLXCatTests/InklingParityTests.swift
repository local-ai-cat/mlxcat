import Foundation
import XCTest

final class InklingParityTests: XCTestCase {
    func testGreedyTokensAndReferenceLogits() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MLXCAT_INKLING_VERIFIED_MODEL"] != nil,
            environment["MLXCAT_INKLING_REFERENCE"] != nil
        else {
            throw XCTSkip("Set the verified checkpoint and reference fixture directory to run Inkling parity.")
        }

        // Isolate Metal initialization from the XCTest host by running this
        // verifier in a standalone process, like mlxcat-http.
        let products = Bundle(for: InklingParityTests.self).bundleURL.deletingLastPathComponent()
        let executable = products.appending(component: "mlxcat-inkling-parity")
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            return XCTFail("Build mlxcat-inkling-parity before running the fixture suite.")
        }
        let process = Process()
        process.executableURL = executable
        let diagnostics = Pipe()
        process.standardOutput = diagnostics
        process.standardError = diagnostics
        try process.run()
        let output = String(decoding: diagnostics.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        process.waitUntilExit()
        print(output)
        XCTAssertEqual(process.terminationStatus, 0, output)
    }
}
