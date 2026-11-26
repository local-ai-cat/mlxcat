import Foundation
@testable import MLXCatHTTP
import XCTest

/// The two properties an app that launches `mlxcat-http` as its child relies on:
/// the server listens on the loopback address it was given and nowhere else, and
/// it can follow its launcher out of the door.
///
/// Both came from one incident (Local AI Cat, 2026-11-26): an app crashed at
/// launch, its `mlxcat-http` child was reparented to launchd and kept port 11435
/// for 31 hours, and `lsof` showed it listening on `*:11435` although it had been
/// started with `--host 127.0.0.1`.
final class SidecarLifetimeTests: XCTestCase {
    // MARK: - Loopback binding

    /// `requiredInterfaceType = .loopback` alone still binds the wildcard
    /// address: the socket is `*:<port>` and only the accepted connections are
    /// filtered. The address must be the one asked for.
    func testLoopbackHostListensOnLoopbackAddressOnly() async throws {
        let server = try OpenAIServer(host: "127.0.0.1", port: 0, backend: SidecarLifetimeStubBackend())
        try await server.start()
        defer { server.stop() }
        let port = try XCTUnwrap(server.boundPort)

        let addresses = try listeningAddresses(port: port)

        XCTAssertEqual(addresses, ["127.0.0.1:\(port)"], "listening on \(addresses)")
    }

    func testIPv6LoopbackHostListensOnIPv6LoopbackOnly() async throws {
        let server = try OpenAIServer(host: "::1", port: 0, backend: SidecarLifetimeStubBackend())
        try await server.start()
        defer { server.stop() }
        let port = try XCTUnwrap(server.boundPort)

        let addresses = try listeningAddresses(port: port)

        XCTAssertEqual(addresses, ["[::1]:\(port)"], "listening on \(addresses)")
    }

    func testLocalhostListensOnIPv4LoopbackOnly() async throws {
        let server = try OpenAIServer(host: "localhost", port: 0, backend: SidecarLifetimeStubBackend())
        try await server.start()
        defer { server.stop() }
        let port = try XCTUnwrap(server.boundPort)

        let addresses = try listeningAddresses(port: port)

        XCTAssertEqual(addresses, ["127.0.0.1:\(port)"], "listening on \(addresses)")
    }

    // MARK: - Parent-death watch

    func testParentPIDIsReadFromTheEnvironment() {
        XCTAssertEqual(ParentProcessWatch.parentPID(environment: ["MLXCAT_PARENT_PID": "4242"]), 4242)
        XCTAssertNil(ParentProcessWatch.parentPID(environment: [:]))
        XCTAssertNil(ParentProcessWatch.parentPID(environment: ["MLXCAT_PARENT_PID": "not-a-pid"]))
        // launchd is never anyone's launcher in this sense; watching it would be meaningless.
        XCTAssertNil(ParentProcessWatch.parentPID(environment: ["MLXCAT_PARENT_PID": "1"]))
        XCTAssertNil(ParentProcessWatch.parentPID(environment: ["MLXCAT_PARENT_PID": "0"]))
        XCTAssertNil(ParentProcessWatch.parentPID(environment: ["MLXCAT_PARENT_PID": "-5"]))
    }

    func testWatchFiresWhenTheWatchedProcessExits() throws {
        let standIn = Process()
        standIn.executableURL = URL(fileURLWithPath: "/bin/sleep")
        standIn.arguments = ["30"]
        try standIn.run()
        defer { if standIn.isRunning { standIn.terminate() } }

        let fired = expectation(description: "parent exit observed")
        let watch = ParentProcessWatch.start(parentPID: standIn.processIdentifier) {
            fired.fulfill()
        }
        defer { watch.cancel() }

        kill(standIn.processIdentifier, SIGKILL)

        wait(for: [fired], timeout: 5)
    }

    func testWatchFiresAtOnceWhenTheParentIsAlreadyGone() throws {
        let standIn = Process()
        standIn.executableURL = URL(fileURLWithPath: "/usr/bin/true")
        try standIn.run()
        standIn.waitUntilExit()

        let fired = expectation(description: "already-gone parent observed")
        let watch = ParentProcessWatch.start(parentPID: standIn.processIdentifier) {
            fired.fulfill()
        }
        defer { watch.cancel() }

        wait(for: [fired], timeout: 5)
    }

    func testWatchStaysQuietWhileTheParentLives() throws {
        let standIn = Process()
        standIn.executableURL = URL(fileURLWithPath: "/bin/sleep")
        standIn.arguments = ["30"]
        try standIn.run()
        defer { if standIn.isRunning { standIn.terminate() } }

        let fired = expectation(description: "no exit while alive")
        fired.isInverted = true
        let watch = ParentProcessWatch.start(parentPID: standIn.processIdentifier) {
            fired.fulfill()
        }
        defer { watch.cancel() }

        wait(for: [fired], timeout: 0.5)
    }

    // MARK: - Helpers

    /// Every address this process LISTENS on for `port`, as `lsof` names them.
    private func listeningAddresses(port: UInt16) throws -> [String] {
        let lsof = Process()
        lsof.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        lsof.arguments = ["-nP", "-a", "-p", "\(getpid())", "-iTCP:\(port)", "-sTCP:LISTEN", "-Fn"]
        let pipe = Pipe()
        lsof.standardOutput = pipe
        lsof.standardError = Pipe()
        try lsof.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        lsof.waitUntilExit()
        let output = String(bytes: data, encoding: .utf8) ?? ""
        return output.split(separator: "\n")
            .filter { $0.hasPrefix("n") }
            .map { String($0.dropFirst()) }
            .sorted()
    }
}

private struct SidecarLifetimeStubBackend: OpenAIChatBackend {
    var models: [OpenAIModelInfo] { [] }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        throw OpenAIServerError.invalidRequest
    }
}
