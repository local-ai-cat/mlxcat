import Foundation
import Metal
import MLX
import XCTest

/// Gives MLX a Metal library under `swift test`, without touching the test bundle.
///
/// SwiftPM cannot compile MLX's Metal shaders, so the tests build a small
/// metallib themselves. It used to be copied into the xctest bundle's
/// `Contents/MacOS`, where MLX looks first, but that invalidates the bundle's
/// signature: under Xcode 27's build backend the next incremental `swift test`
/// then fails to re-sign `MLXCatTests.xctest`.
///
/// MLX's last fallback is `default.metallib` resolved against the current
/// directory, so the library now lives in `.build/mlxcat-metal-runtime/`, and
/// the current directory points there only while the first MLX evaluation
/// constructs the Metal device (which loads the library once per process).
enum MLXMetalRuntime {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var deviceReady = false

    static func requireAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("MLX probe skipped because no Metal device is visible to this process.")
        }

        lock.lock()
        defer { lock.unlock() }
        if deviceReady {
            return
        }
        try prepareDefaultMetallib(file: file, line: line)
        deviceReady = true
    }

    private static func prepareDefaultMetallib(file: StaticString, line: UInt) throws {
        let executableDirectory = try XCTUnwrap(
            Bundle(for: GateProbeTests.self).executableURL?.deletingLastPathComponent(),
            "Unable to locate XCTest executable directory.",
            file: file,
            line: line
        )
        // A metallib colocated with the binary wins MLX's search; nothing to do.
        let colocatedLibrary = executableDirectory.appendingPathComponent("mlx.metallib")
        if FileManager.default.fileExists(atPath: colocatedLibrary.path) {
            return
        }

        let root = repositoryRoot()
        let metalSourceDirectory = root
            .appendingPathComponent(".build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal")
        guard FileManager.default.fileExists(atPath: metalSourceDirectory.path) else {
            throw XCTSkip("MLX Metal sources are not available in .build/checkouts.")
        }

        let runtimeDirectory = root.appendingPathComponent(".build/mlxcat-metal-runtime")
        let airDirectory = root.appendingPathComponent(".build/mlxcat-metal-air")
        try FileManager.default.createDirectory(
            at: airDirectory,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: runtimeDirectory,
            withIntermediateDirectories: true
        )

        let metallib = runtimeDirectory.appendingPathComponent("default.metallib")
        let airFiles = try compileAirFiles(
            metalSourceDirectory: metalSourceDirectory,
            airDirectory: airDirectory
        )
        try run(
            "/usr/bin/xcrun",
            arguments: ["-sdk", "macosx", "metallib"] + airFiles.map(\.path) + [
                "-o", metallib.path,
            ]
        )

        try loadDevice(from: runtimeDirectory)
    }

    /// Constructs MLX's Metal device with `directory` as the current directory,
    /// so MLX's relative `default.metallib` fallback resolves to it, then
    /// restores the previous current directory.
    private static func loadDevice(from directory: URL) throws {
        let fileManager = FileManager.default
        let previousDirectory = fileManager.currentDirectoryPath
        guard fileManager.changeCurrentDirectoryPath(directory.path) else {
            throw RuntimeError.changeDirectoryFailed(directory.path)
        }
        defer { fileManager.changeCurrentDirectoryPath(previousDirectory) }

        let probe = MLXArray([1, 2, 3] as [Float]) * 2
        eval(probe)
    }

    private static func compileAirFiles(
        metalSourceDirectory: URL,
        airDirectory: URL
    ) throws -> [URL] {
        let sources = [
            "arg_reduce.metal",
            "conv.metal",
            "gemv.metal",
            "layer_norm.metal",
            "random.metal",
            "rms_norm.metal",
            "rope.metal",
            "scaled_dot_product_attention.metal",
            "steel/attn/kernels/steel_attention.metal",
        ]

        return try sources.map { relativePath in
            let source = metalSourceDirectory.appendingPathComponent(relativePath)
            let airName = relativePath
                .replacingOccurrences(of: "/", with: "_")
                .replacingOccurrences(of: ".metal", with: ".air")
            let output = airDirectory.appendingPathComponent(airName)
            try run(
                "/usr/bin/xcrun",
                arguments: [
                    "-sdk", "macosx", "metal",
                    "-x", "metal",
                    "-Wall",
                    "-Wextra",
                    "-fno-fast-math",
                    "-Wno-c++17-extensions",
                    "-Wno-c++20-extensions",
                    "-mmacosx-version-min=14.0",
                    "-c", source.path,
                    "-I", metalSourceDirectory.path,
                    "-o", output.path,
                ]
            )
            return output
        }
    }

    private static func run(_ executable: String, arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let output = String(
                data: pipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
            throw RuntimeError.commandFailed(executable, arguments, output)
        }
    }

    private static func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}

private enum RuntimeError: Error, CustomStringConvertible {
    case commandFailed(String, [String], String)
    case changeDirectoryFailed(String)

    var description: String {
        switch self {
        case .commandFailed(let executable, let arguments, let output):
            return ([executable] + arguments).joined(separator: " ") + "\n" + output
        case .changeDirectoryFailed(let path):
            return "Could not change the current directory to \(path)"
        }
    }
}
