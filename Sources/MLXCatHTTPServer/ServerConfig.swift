import Foundation
import MLX
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXCatNative
import MLXCatSpeech
import MLXCatSpeechWhisperKit

struct ServerConfig {
    let host: String
    let port: UInt16
    let modelPath: String
    let embeddingModelPath: String?
    let whisperKitModelsPath: String?
    let modelID: String?
    let maxConcurrentRequests: Int
    let memoryGuardTier: MemoryGuardTier?
    let memoryCeilingBytes: Int64?
    let idleTimeout: TimeInterval?
    let pinnedModelIDs: [String]
    let mcpConfigPath: String?

    static func parse(_ arguments: [String]) throws -> ServerConfig {
        var draft = ServerConfigDraft()
        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            if argument == "--help" || argument == "-h" {
                printHelp()
                Foundation.exit(0)
            }
            guard ServerConfigDraft.options.contains(argument) else {
                throw ServerConfigError.invalidArgument(argument)
            }
            index += 1
            try draft.apply(argument, rawValue: value(arguments, at: index, for: argument))
            index += 1
        }
        return try draft.build()
    }

    private static func value(_ arguments: [String], at index: Int, for option: String) throws -> String {
        guard arguments.indices.contains(index) else {
            throw ServerConfigError.invalidArgument("missing value for \(option)")
        }
        return arguments[index]
    }

    private static func printHelp() {
        print(
            """
            Usage: swift run mlxcat-http [options]

            Options:
              --host HOST                       Bind host. Default: 127.0.0.1
              --port PORT                       Bind port. Default: 18181
              --model-dir PATH                  MLX model directory. Required unless MLXSERVE_MODEL_DIR is set.
              --embedding-model-dir PATH        Optional MLX embedding model directory. Also supports \
            MLXSERVE_EMBEDDING_MODEL_DIR.
              --whisperkit-models-dir PATH      Optional WhisperKit models root for /v1/audio/transcriptions. \
            Also supports MLXSERVE_WHISPERKIT_MODEL_DIR.
              --model-id ID                     Single-model compatibility override.
              --max-concurrent-requests N       Scheduler concurrency cap. Default: 8
              --memory-guard-tier TIER          safe, balanced, or aggressive. Default: off.
              --memory-ceiling-bytes N          Explicit pool ceiling in bytes. Overrides memory guard tier when > 0.
              --idle-timeout SECONDS            Unload idle, unpinned models after this many seconds. Default: off.
              --pin ID                          Pin a discovered chat model in memory. Repeatable.
              --mcp-config PATH                 Optional Claude Desktop-style MCP JSON config.
            """
        )
    }
}

enum ServerConfigError: Error, CustomStringConvertible {
    case invalidArgument(String)

    var description: String {
        switch self {
        case .invalidArgument(let message):
            return message
        }
    }
}

private struct ServerConfigDraft {
    var host = "127.0.0.1"
    var port: UInt16 = 18181
    var modelPath = ProcessInfo.processInfo.environment["MLXSERVE_MODEL_DIR"]
        ?? ProcessInfo.processInfo.environment["MLXSERVE_TEST_MODEL"]
    var embeddingModelPath = ProcessInfo.processInfo.environment["MLXSERVE_EMBEDDING_MODEL_DIR"]
    var whisperKitModelsPath = ProcessInfo.processInfo.environment["MLXSERVE_WHISPERKIT_MODEL_DIR"]
    var modelID: String?
    var maxConcurrentRequests = 8
    var memoryGuardTier: MemoryGuardTier?
    var memoryCeilingBytes: Int64?
    var idleTimeout: TimeInterval?
    var pinnedModelIDs: [String] = []
    var mcpConfigPath: String?
    static let options: Set<String> = [
        "--host",
        "--port",
        "--model-dir",
        "--embedding-model-dir",
        "--whisperkit-models-dir",
        "--model-id",
        "--max-concurrent-requests",
        "--memory-guard-tier",
        "--memory-ceiling-bytes",
        "--idle-timeout",
        "--pin",
        "--mcp-config"
    ]

    mutating func apply(_ argument: String, rawValue: String) throws {
        switch argument {
        case "--model-dir":
            modelPath = rawValue
        case "--embedding-model-dir":
            embeddingModelPath = rawValue
        case "--whisperkit-models-dir":
            whisperKitModelsPath = rawValue
        case "--model-id":
            modelID = rawValue
        case "--pin":
            pinnedModelIDs.append(rawValue)
        case "--mcp-config":
            mcpConfigPath = rawValue
        default: try applyRuntime(argument, rawValue: rawValue)
        }
    }

    private mutating func applyRuntime(_ argument: String, rawValue: String) throws {
        switch argument {
        case "--host":
            host = rawValue
        case "--port":
            guard let parsed = UInt16(rawValue) else {
                throw ServerConfigError.invalidArgument("invalid port")
            }
            port = parsed
        case "--max-concurrent-requests":
            guard let parsed = Int(rawValue) else {
                throw ServerConfigError.invalidArgument("invalid max concurrent requests")
            }
            maxConcurrentRequests = parsed
        default: try applyMemory(argument, rawValue: rawValue)
        }
    }

    private mutating func applyMemory(_ argument: String, rawValue: String) throws {
        switch argument {
        case "--memory-guard-tier":
            guard let tier = MemoryGuardTier(rawValue: rawValue) else {
                throw ServerConfigError.invalidArgument("invalid memory guard tier: \(rawValue)")
            }
            memoryGuardTier = tier
        case "--memory-ceiling-bytes":
            guard let parsed = nonNegativeInt64(rawValue) else {
                throw ServerConfigError.invalidArgument("invalid memory ceiling bytes")
            }
            memoryCeilingBytes = parsed
        case "--idle-timeout":
            guard let parsed = TimeInterval(rawValue), parsed > 0 else {
                throw ServerConfigError.invalidArgument("invalid idle timeout")
            }
            idleTimeout = parsed
        default: throw ServerConfigError.invalidArgument(argument)
        }
    }

    func build() throws -> ServerConfig {
        guard let modelPath else {
            throw ServerConfigError.invalidArgument("missing --model-dir or MLXSERVE_MODEL_DIR")
        }

        return ServerConfig(
            host: host,
            port: port,
            modelPath: modelPath,
            embeddingModelPath: embeddingModelPath,
            whisperKitModelsPath: whisperKitModelsPath,
            modelID: modelID,
            maxConcurrentRequests: maxConcurrentRequests,
            memoryGuardTier: memoryGuardTier,
            memoryCeilingBytes: memoryCeilingBytes,
            idleTimeout: idleTimeout,
            pinnedModelIDs: pinnedModelIDs,
            mcpConfigPath: mcpConfigPath
        )
    }
}
