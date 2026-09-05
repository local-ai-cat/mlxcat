import Foundation
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

struct RouteFakeEngine: Sendable {
    let id: String
}

struct RouteFakeLoader: EnginePoolModelLoader {
    func loadModel(id: String, modelURL: URL) async throws -> RouteFakeEngine {
        RouteFakeEngine(id: id)
    }

    func unloadModel(_ engine: RouteFakeEngine, id: String) async {}
}

final class FakePoolLifecycleBackend: OpenAIModelLifecycleBackend, @unchecked Sendable {
    let models: [OpenAIModelInfo] = [
        OpenAIModelInfo(id: "alpha"),
        OpenAIModelInfo(id: "beta")
    ]

    private let pool: EnginePool<RouteFakeLoader>

    init() {
        self.pool = EnginePool(
            models: [
                "alpha": DiscoveredModel(
                    id: "alpha",
                    modelURL: URL(fileURLWithPath: "/models/alpha", isDirectory: true),
                    estimatedSize: 100
                ),
                "beta": DiscoveredModel(
                    id: "beta",
                    modelURL: URL(fileURLWithPath: "/models/beta", isDirectory: true),
                    estimatedSize: 200
                )
            ],
            loader: RouteFakeLoader(),
            finalCeiling: 1_000
        )
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        OpenAIChatStream(
            promptTokens: 0,
            chunks: AsyncThrowingStream { continuation in
                continuation.finish()
            }
        )
    }

    func modelPoolStatus() async throws -> OpenAIModelPoolStatus {
        OpenAIModelPoolStatus(await pool.status())
    }

    func loadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        do {
            let result = try await pool.load(id)
            return OpenAIModelLifecycleResult(
                modelID: id,
                message: result.alreadyLoaded ? "Already loaded: \(id)" : "Loaded: \(id)"
            )
        } catch {
            throw openAIError(from: error)
        }
    }

    func unloadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        do {
            try await pool.unload(id)
            return OpenAIModelLifecycleResult(modelID: id)
        } catch {
            throw openAIError(from: error)
        }
    }

    func acquireForTest(_ id: String) async throws -> EnginePoolLease<RouteFakeEngine> {
        try await pool.acquire(id)
    }

    func releaseForTest(_ lease: EnginePoolLease<RouteFakeEngine>) async {
        await pool.release(lease)
    }

    private func openAIError(from error: Error) -> OpenAIHTTPError {
        guard let poolError = error as? EnginePoolError else {
            return OpenAIHTTPError(status: 500, message: String(describing: error))
        }
        return OpenAIHTTPError(
            status: poolError.httpStatus,
            message: poolError.message,
            retryAfterSeconds: poolError.retryAfterSeconds
        )
    }
}

private extension OpenAIModelPoolStatus {
    init(_ status: EnginePoolStatus) {
        self.init(
            finalCeiling: status.finalCeiling,
            currentModelMemory: status.currentModelMemory,
            modelCount: status.modelCount,
            loadedCount: status.loadedCount,
            models: status.models.map(OpenAIModelRuntimeStatus.init)
        )
    }
}

private extension OpenAIModelRuntimeStatus {
    init(_ status: EnginePoolModelStatus) {
        self.init(
            id: status.id,
            modelPath: status.modelPath,
            loaded: status.loaded,
            isLoading: status.isLoading,
            estimatedSize: status.estimatedSize,
            actualSize: status.actualSize,
            pinned: status.pinned,
            lastAccess: status.lastAccess,
            inUse: status.inUse
        )
    }
}
