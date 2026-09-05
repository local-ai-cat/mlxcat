import Foundation
import MLX
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXCatNative
import MLXCatSpeech
import MLXCatSpeechWhisperKit

struct MLXCatHTTPServerMain {
    static func main() async throws {
        let config = try ServerConfig.parse(CommandLine.arguments)
        let catalog = try ServerModelCatalog(config: config)
        let discovered = catalog.chat
        let rerankDiscovered = catalog.rerank
        let effectiveCeiling = finalMemoryCeiling(
            overrideBytes: config.memoryCeilingBytes,
            tier: config.memoryGuardTier
        )
        // Bound the ALLOCATOR, not just the watchdog. MLX's cache limit defaults to
        // its memory limit (1.5x the recommended working set), so a ceiling alone
        // left the buffer pool free to grow far past it between 2 s watchdog polls.
        let appliedLimits = MemoryGuard.applyAllocatorLimits(
            ceilingBytes: effectiveCeiling.bytes,
            cacheLimitOverrideBytes: MemoryGuard.cacheLimitOverrideFromEnvironment()
        )

        let pool = EnginePool(
            models: discovered,
            loader: NativeModelLoader(maxConcurrentRequests: config.maxConcurrentRequests),
            finalCeiling: effectiveCeiling.bytes,
            idleTimeout: config.idleTimeout
        )
        let embeddingBackend = await makeEmbeddingBackend(modelURL: catalog.embeddingURL, pool: pool)
        for pinnedModelID in config.pinnedModelIDs {
            try await pool.setPinned(true, for: pinnedModelID)
        }

        // M8b speech: registry-backed transcription (WhisperKit adapter first).
        let rerankBackend = await makeRerankBackend(models: rerankDiscovered, pool: pool)
        let speechBackend = await makeSpeechBackend(modelsPath: config.whisperKitModelsPath, pool: pool)

        let memoryWatchdog = makeMemoryWatchdog(ceilingBytes: effectiveCeiling.bytes, pool: pool)
        await installPressureAdmission(pool: pool, watchdog: memoryWatchdog)
        let backend = PoolBackedChatBackend(
            pool: pool,
            modelIDs: Array(discovered.keys),
            embeddingsBackend: embeddingBackend,
            rerankBackend: rerankBackend,
            speechBackend: speechBackend,
            memoryWatchdog: memoryWatchdog
        )
        let mcpManager = try await makeMCPManager(configPath: config.mcpConfigPath)
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
        logStartup(config: config, catalog: catalog, ceiling: effectiveCeiling, limits: appliedLimits)
        server.waitForever()
    }
}

try await MLXCatHTTPServerMain.main()
