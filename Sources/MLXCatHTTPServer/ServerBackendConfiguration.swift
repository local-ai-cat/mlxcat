import Foundation
import MLX
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXCatNative
import MLXCatSpeech
import MLXCatSpeechWhisperKit

struct ServerModelCatalog {
    let chat: [String: DiscoveredModel]
    let rerank: [String: DiscoveredModel]
    let embeddingURL: URL?
    let discoveredCount: Int

    init(config: ServerConfig) throws {
        let modelRootURL = URL(fileURLWithPath: config.modelPath, isDirectory: true)
        let allDiscovered = try applyModelIDOverrideIfNeeded(
            try ModelDiscovery.discoverModels(in: modelRootURL),
            override: config.modelID
        )
        guard !allDiscovered.isEmpty else {
            throw ServerConfigError.invalidArgument("no models discovered in \(config.modelPath)")
        }
        let embeddingModelURL = config.embeddingModelPath.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let nonEmbeddingDiscovered = ModelDiscovery.excludingModelDirectory(
            allDiscovered, directory: embeddingModelURL
        )
        let rerankDiscovered = nonEmbeddingDiscovered.filter {
            NativeRerankBackend.isRerankModelDirectory($0.value.modelURL)
        }
        let discovered = nonEmbeddingDiscovered.filter { rerankDiscovered[$0.key] == nil }
        chat = discovered
        rerank = rerankDiscovered
        embeddingURL = embeddingModelURL
        discoveredCount = allDiscovered.count
    }
}

extension MLXCatHTTPServerMain {
    static func makeEmbeddingBackend(modelURL: URL?, pool: any EnginePoolResourceManaging) async
        -> NativeEmbeddingsBackend? {
        let embeddingBackend: NativeEmbeddingsBackend?
        if let modelURL {
            embeddingBackend = NativeEmbeddingsBackend(
                modelURL: modelURL,
                footprint: EnginePoolFootprint(
                    bytes: try? ModelDiscovery.estimateModelSize(at: modelURL), source: .diskEstimate
                ),
                pool: pool, cache: NativeEmbeddingModelCache()
            )
            // Register metadata only; loading waits for an admitted embedding request.
            await embeddingBackend?.registerModel()
        } else {
            embeddingBackend = nil
        }
        return embeddingBackend
    }

    static func makeSpeechBackend(modelsPath: String?, pool: any EnginePoolResourceManaging) async
        -> RegistrySpeechBackend? {
        let speechBackend: RegistrySpeechBackend?
        if let whisperKitModelsPath = modelsPath {
            let registry = SpeechEngineRegistry()
            await registry.register(
                await PoolBackedSpeechAdapter(adapter: WhisperKitSpeechAdapter(
                    modelsRoot: URL(fileURLWithPath: whisperKitModelsPath, isDirectory: true)
                ), pool: pool)
            )
            speechBackend = await RegistrySpeechBackend(registry: registry)
        } else {
            speechBackend = nil
        }

        return speechBackend
    }

    static func makeMCPManager(configPath: String?) async throws -> MCPManager? {
        let mcpManager: MCPManager?
        if let mcpConfigPath = configPath {
            let manager = try MCPManager.load(from: URL(fileURLWithPath: mcpConfigPath))
            await manager.connectAll()
            mcpManager = manager
        } else {
            mcpManager = nil
        }
        return mcpManager
    }
}

extension MLXCatHTTPServerMain {
    static func makeRerankBackend(models: [String: DiscoveredModel], pool: any EnginePoolResourceManaging) async
        -> NativeRerankBackend? {
        let rerankBackend: NativeRerankBackend?
        if models.isEmpty {
            rerankBackend = nil
        } else {
            rerankBackend = await NativeRerankBackend(
                models: models, pool: pool, cache: NativeRerankModelCache()
            )
        }
        return rerankBackend
    }

    static func logStartup(config: ServerConfig, catalog: ServerModelCatalog,
                           ceiling: MemoryGuard.EffectiveCeiling, limits: (memoryLimit: Int64, cacheLimit: Int64)) {
        print("MLXCatHTTP listening on http://\(config.host):\(config.port)")
        let allocatorNote = limits.memoryLimit > 0
            ? "allocator: memoryLimit \(ModelDiscovery.formatSize(limits.memoryLimit)), "
                + "cacheLimit \(ModelDiscovery.formatSize(limits.cacheLimit))"
            : "allocator: MLX defaults (unbounded cache)"
        print(
            "Discovered \(catalog.discoveredCount) model(s) "
                + "(\(catalog.chat.count) chat, \(catalog.rerank.count) rerank); memory ceiling: "
                + "\(ModelDiscovery.formatSize(ceiling.bytes)) (\(ceiling.source)); \(allocatorNote)"
        )

        // Say out loud whether the serialization overrides were understood.
        //
        // On 2026-08-22 a benchmark A/B compared `mlxcat` against
        // `mlxcat-batched`, whose only difference is the env var below — against
        // a binary built eighteen minutes before that env var existed. The old
        // binary ignored it in silence, both arms ran identically, and the run
        // looked like a clean result contradicting the real one. A treatment
        // that cannot be observed in the output is not a treatment.
        let overrides = NativeModelLoader.serializationOverrides()
        print(
            overrides.isEmpty
                ? "serialization overrides: none (MLXCAT_UNSERIALIZE_MODEL_TYPES unset or empty)"
                : "serialization overrides: \(overrides.sorted().joined(separator: ","))"
        )
        fflush(stdout)
    }
}

extension MLXCatHTTPServerMain {
    static func installPressureAdmission<Loader: EnginePoolModelLoader>(
        pool: EnginePool<Loader>, watchdog: MemoryWatchdog?
    ) async {
        if let watchdog {
            await pool.setPressureAdmission { [weak watchdog] bytes in
                try await watchdog?.checkAdmission(additionalBytes: bytes)
            }
        }
    }
}
