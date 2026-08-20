import Foundation
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXCatNative
import MLXCatSpeech
import MLXCatSpeechWhisperKit
import Tokenizers

struct MLXCatHTTPServerMain {
    static func main() async throws {
        let config = try ServerConfig.parse(CommandLine.arguments)
        let modelRootURL = URL(fileURLWithPath: config.modelPath, isDirectory: true)
        let allDiscovered = try applyModelIDOverrideIfNeeded(
            try ModelDiscovery.discoverModels(in: modelRootURL),
            override: config.modelID
        )
        guard !allDiscovered.isEmpty else {
            throw ServerConfigError.invalidArgument("no models discovered in \(config.modelPath)")
        }
        let rerankDiscovered = allDiscovered.filter {
            NativeRerankBackend.isRerankModelDirectory($0.value.modelURL)
        }
        let discovered = allDiscovered.filter { rerankDiscovered[$0.key] == nil }
        let effectiveCeiling = finalMemoryCeiling(
            overrideBytes: config.memoryCeilingBytes,
            tier: config.memoryGuardTier
        )

        // M5 embeddings: a separate model class, NOT part of the LLM pool.
        let embeddingBackend: NativeEmbeddingsBackend?
        if let embeddingModelPath = config.embeddingModelPath {
            let embeddingModelURL = URL(fileURLWithPath: embeddingModelPath)
            let embeddingContainer = try await EmbedderModelFactory.shared.loadContainer(
                from: embeddingModelURL,
                using: #huggingFaceTokenizerLoader()
            )
            embeddingBackend = NativeEmbeddingsBackend(
                container: embeddingContainer,
                modelID: embeddingModelURL.lastPathComponent
            )
        } else {
            embeddingBackend = nil
        }
        let rerankBackend = rerankDiscovered.isEmpty ? nil : NativeRerankBackend(
            models: rerankDiscovered,
            memoryCeilingBytes: effectiveCeiling.bytes
        )
        let pool = EnginePool(
            models: discovered,
            loader: NativeModelLoader(maxConcurrentRequests: config.maxConcurrentRequests),
            finalCeiling: effectiveCeiling.bytes,
            idleTimeout: config.idleTimeout
        )
        for pinnedModelID in config.pinnedModelIDs {
            try await pool.setPinned(true, for: pinnedModelID)
        }

        // M8b speech: registry-backed transcription (WhisperKit adapter first).
        let speechBackend: RegistrySpeechBackend?
        if let whisperKitModelsPath = config.whisperKitModelsPath {
            let registry = SpeechEngineRegistry()
            await registry.register(
                WhisperKitSpeechAdapter(
                    modelsRoot: URL(fileURLWithPath: whisperKitModelsPath, isDirectory: true)
                )
            )
            speechBackend = await RegistrySpeechBackend(registry: registry)
        } else {
            speechBackend = nil
        }

        let memoryWatchdog = makeMemoryWatchdog(ceilingBytes: effectiveCeiling.bytes, pool: pool)
        let backend = PoolBackedChatBackend(
            pool: pool,
            modelIDs: Array(discovered.keys),
            embeddingsBackend: embeddingBackend,
            rerankBackend: rerankBackend,
            speechBackend: speechBackend,
            memoryWatchdog: memoryWatchdog
        )
        let mcpManager: MCPManager?
        if let mcpConfigPath = config.mcpConfigPath {
            let manager = try MCPManager.load(from: URL(fileURLWithPath: mcpConfigPath))
            await manager.connectAll()
            mcpManager = manager
        } else {
            mcpManager = nil
        }
        let server = try OpenAIServer(
            host: config.host,
            port: config.port,
            backend: backend,
            mcpManager: mcpManager
        )
        if config.idleTimeout != nil {
            startIdleSweep(pool: pool, interval: config.idleTimeout ?? 0)
        }
        if let memoryWatchdog {
            startMemoryWatchdogPoll(watchdog: memoryWatchdog, interval: memoryWatchdogPollInterval)
        }

        try await server.start()
        print("MLXCatHTTP listening on http://\(config.host):\(config.port)")
        print("Discovered \(allDiscovered.count) model(s) (\(discovered.count) chat, \(rerankDiscovered.count) rerank); memory ceiling: \(ModelDiscovery.formatSize(effectiveCeiling.bytes)) (\(effectiveCeiling.source))")
        fflush(stdout)
        server.waitForever()
    }
}

try await MLXCatHTTPServerMain.main()

private final class NativeEmbeddingsBackend: OpenAIEmbeddingsBackend, @unchecked Sendable {
    let embeddingModels: [OpenAIModelInfo]

    private let container: EmbedderModelContainer

    init(container: EmbedderModelContainer, modelID: String) {
        self.container = container
        self.embeddingModels = [OpenAIModelInfo(id: modelID, maxModelLength: nil)]
    }

    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> OpenAIEmbeddingsResult {
        let inputs = request.input.values
        return await container.perform { context in
            var embeddings: [[Float]] = []
            var promptTokens = 0
            embeddings.reserveCapacity(inputs.count)

            for input in inputs {
                let tokens = context.tokenizer.encode(text: input, addSpecialTokens: true)
                promptTokens += tokens.count
                let modelInput = MLXArray(tokens)[.newAxis, 0...]
                let output = context.model(
                    modelInput,
                    positionIds: nil,
                    tokenTypeIds: nil,
                    attentionMask: nil
                )
                let pooled = context.pooling(output, normalize: true)
                pooled.eval()
                embeddings.append(pooled[0].asArray(Float.self))
            }

            return OpenAIEmbeddingsResult(embeddings: embeddings, promptTokens: promptTokens)
        }
    }
}

private struct ServerConfig {
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

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--host":
                index += 1
                host = try value(arguments, at: index, for: argument)
            case "--port":
                index += 1
                guard let parsed = UInt16(try value(arguments, at: index, for: argument)) else {
                    throw ServerConfigError.invalidArgument("invalid port")
                }
                port = parsed
            case "--model-dir":
                index += 1
                modelPath = try value(arguments, at: index, for: argument)
            case "--embedding-model-dir":
                index += 1
                embeddingModelPath = try value(arguments, at: index, for: argument)
            case "--whisperkit-models-dir":
                index += 1
                whisperKitModelsPath = try value(arguments, at: index, for: argument)
            case "--model-id":
                index += 1
                modelID = try value(arguments, at: index, for: argument)
            case "--max-concurrent-requests":
                index += 1
                guard let parsed = Int(try value(arguments, at: index, for: argument)) else {
                    throw ServerConfigError.invalidArgument("invalid max concurrent requests")
                }
                maxConcurrentRequests = parsed
            case "--memory-guard-tier":
                index += 1
                let rawValue = try value(arguments, at: index, for: argument)
                guard let tier = MemoryGuardTier(rawValue: rawValue) else {
                    throw ServerConfigError.invalidArgument("invalid memory guard tier: \(rawValue)")
                }
                memoryGuardTier = tier
            case "--memory-ceiling-bytes":
                index += 1
                guard let parsed = nonNegativeInt64(try value(arguments, at: index, for: argument)) else {
                    throw ServerConfigError.invalidArgument("invalid memory ceiling bytes")
                }
                memoryCeilingBytes = parsed
            case "--idle-timeout":
                index += 1
                guard let parsed = TimeInterval(try value(arguments, at: index, for: argument)), parsed > 0 else {
                    throw ServerConfigError.invalidArgument("invalid idle timeout")
                }
                idleTimeout = parsed
            case "--pin":
                index += 1
                pinnedModelIDs.append(try value(arguments, at: index, for: argument))
            case "--mcp-config":
                index += 1
                mcpConfigPath = try value(arguments, at: index, for: argument)
            case "--help", "-h":
                printHelp()
                Foundation.exit(0)
            default:
                throw ServerConfigError.invalidArgument(argument)
            }
            index += 1
        }

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
              --embedding-model-dir PATH        Optional MLX embedding model directory. Also supports MLXSERVE_EMBEDDING_MODEL_DIR.
              --whisperkit-models-dir PATH      Optional WhisperKit models root for /v1/audio/transcriptions. Also supports MLXSERVE_WHISPERKIT_MODEL_DIR.
              --model-id ID                     Single-model compatibility override.
              --max-concurrent-requests N       Scheduler concurrency cap. Default: 8
              --memory-guard-tier TIER          safe, balanced, or aggressive. Default: off.
              --memory-ceiling-bytes N          Explicit pool ceiling in bytes. Overrides memory guard tier when > 0.
              --idle-timeout SECONDS            Unload idle, unpinned models after this many seconds. Default: off.
              --pin ID                          Pin a discovered model in memory. Repeatable.
              --mcp-config PATH                 Optional Claude Desktop-style MCP JSON config.
            """
        )
    }
}

private func applyModelIDOverrideIfNeeded(
    _ discovered: [String: DiscoveredModel],
    override: String?
) throws -> [String: DiscoveredModel] {
    guard let override else {
        return discovered
    }
    guard discovered.count == 1, let model = discovered.values.first else {
        throw ServerConfigError.invalidArgument("--model-id can only be used with a single discovered model")
    }
    return [
        override: DiscoveredModel(
            id: override,
            modelURL: model.modelURL,
            estimatedSize: model.estimatedSize
        )
    ]
}

private func finalMemoryCeiling(
    overrideBytes: Int64?,
    tier: MemoryGuardTier?
) -> MemoryGuard.EffectiveCeiling {
    let recommendedWorkingSet = Int64(GPU.maxRecommendedWorkingSetBytes() ?? 0)
    return MemoryGuard.effectiveCeiling(
        overrideBytes: overrideBytes,
        recommendedWorkingSetBytes: recommendedWorkingSet,
        tier: tier
    )
}

private func nonNegativeInt64(_ value: String) -> Int64? {
    let normalized = value.replacingOccurrences(of: "_", with: "")
    guard let parsed = Int64(normalized), parsed >= 0 else {
        return nil
    }
    return parsed
}

private let memoryWatchdogPollInterval: TimeInterval = 2

/// Builds the live memory watchdog when a ceiling is configured. The usage sampler
/// reads MLX's own accounting (active weights + reclaimable cache); the reclaimer's
/// trim step clears the MLX cache pool and its evict step unloads idle pool models.
private func makeMemoryWatchdog<Loader: EnginePoolModelLoader>(
    ceilingBytes: Int64,
    pool: EnginePool<Loader>
) -> MemoryWatchdog? {
    guard ceilingBytes > 0 else { return nil }
    return MemoryWatchdog(
        configuration: MemoryWatchdogConfiguration(ceilingBytes: ceilingBytes),
        sampler: { Int64(Memory.activeMemory + Memory.cacheMemory) },
        reclaimer: ClosureMemoryWatchdogReclaimer(
            trimReclaimableCaches: { _ in
                let before = Memory.cacheMemory
                Memory.clearCache()
                return Int64(max(0, before - Memory.cacheMemory))
            },
            evictIdleModels: { target in
                await pool.reclaimIdleModels(targetBytes: target)
            }
        )
    )
}

private func startMemoryWatchdogPoll(watchdog: MemoryWatchdog, interval: TimeInterval) {
    let sleepNanoseconds = UInt64(max(interval, 1) * 1_000_000_000)
    Task.detached {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: sleepNanoseconds)
            _ = await watchdog.poll()
        }
    }
}

private func startIdleSweep<Loader: EnginePoolModelLoader>(
    pool: EnginePool<Loader>,
    interval: TimeInterval
) {
    let sleepNanoseconds = UInt64(max(interval, 1) * 1_000_000_000)
    Task.detached {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: sleepNanoseconds)
            let unloaded = await pool.sweepIdleModels()
            if !unloaded.isEmpty {
                print("Idle timeout unloaded model(s): \(unloaded.joined(separator: ", "))")
                fflush(stdout)
            }
        }
    }
}

private enum ServerConfigError: Error, CustomStringConvertible {
    case invalidArgument(String)

    var description: String {
        switch self {
        case .invalidArgument(let message):
            return message
        }
    }
}
