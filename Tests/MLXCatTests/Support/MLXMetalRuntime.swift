import Foundation
import Metal
import XCTest

enum MLXMetalRuntime {
    static func requireAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("MLX probe skipped because no Metal device is visible to this process.")
        }

        try prepareDefaultMetallib(file: file, line: line)
    }

    private static func prepareDefaultMetallib(file: StaticString, line: UInt) throws {
        let executableDirectory = try XCTUnwrap(
            Bundle(for: GateProbeTests.self).executableURL?.deletingLastPathComponent(),
            "Unable to locate XCTest executable directory.",
            file: file,
            line: line
        )
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

        let outputDirectory = root.appendingPathComponent(".build/arm64-apple-macosx/debug")
        let airDirectory = root.appendingPathComponent(".build/mlxcat-metal-air")
        try FileManager.default.createDirectory(
            at: airDirectory,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let airFiles = try compileAirFiles(
            metalSourceDirectory: metalSourceDirectory,
            airDirectory: airDirectory
        )
        let metallib = outputDirectory.appendingPathComponent("mlx.metallib")
        try run(
            "/usr/bin/xcrun",
            arguments: ["-sdk", "macosx", "metallib"] + airFiles.map(\.path) + [
                "-o", metallib.path,
            ]
        )

        try install(metallib: metallib, executableDirectory: executableDirectory)
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

    private static func install(metallib: URL, executableDirectory: URL) throws {
        let resourceDirectory = executableDirectory.appendingPathComponent("Resources")
        try FileManager.default.createDirectory(
            at: resourceDirectory,
            withIntermediateDirectories: true
        )

        for destination in [
            executableDirectory.appendingPathComponent("mlx.metallib"),
            resourceDirectory.appendingPathComponent("mlx.metallib"),
            resourceDirectory.appendingPathComponent("default.metallib"),
        ] {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.copyItem(at: metallib, to: destination)
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

    var description: String {
        switch self {
        case .commandFailed(let executable, let arguments, let output):
            return ([executable] + arguments).joined(separator: " ") + "\n" + output
        }
    }
}
