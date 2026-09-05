import Foundation
import MLXCat
import MLXCatHTTP

/// Admission and residency use the same owner as chat, speech, and rerank.
final class NativeEmbeddingsBackend: OpenAIEmbeddingsBackend, Sendable {
    let embeddingModels: [OpenAIModelInfo]
    private let modelID: String
    private let resource: EnginePoolResource
    private let pool: any EnginePoolResourceManaging
    private let cache: any NativeEmbeddingModelServing

    init(modelURL: URL, footprint: EnginePoolFootprint, pool: any EnginePoolResourceManaging,
         cache: any NativeEmbeddingModelServing) {
        modelID = modelURL.lastPathComponent
        embeddingModels = [OpenAIModelInfo(id: modelURL.lastPathComponent, maxModelLength: nil)]
        self.pool = pool
        self.cache = cache
        resource = EnginePoolResource(
            id: "embedding://\(modelURL.lastPathComponent)", modelID: modelURL.lastPathComponent,
            modelType: "embedding", footprint: footprint, usesMLXAllocator: true,
            load: { try await cache.load(modelURL: modelURL) }, unload: { await cache.unload() }
        )
    }

    func registerModel() async { await pool.registerResource(resource) }

    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> OpenAIEmbeddingsResult {
        guard request.model == modelID else {
            throw OpenAIHTTPError(status: 404, message: "Embedding model '\(request.model)' not found.")
        }
        await registerModel()
        let lease: EnginePoolResourceLease
        do { lease = try await pool.acquireResource(resource.id) } catch let error as EnginePoolError {
            throw OpenAIHTTPError(status: error.httpStatus, message: error.description)
        }
        do {
            try Task.checkCancellation()
            let result = try await cache.embed(request)
            // Cancellation never releases the lease while the owner's scoring call is active.
            try Task.checkCancellation()
            await pool.releaseResource(lease)
            return result
        } catch {
            await pool.releaseResource(lease)
            throw error
        }
    }
}

protocol NativeEmbeddingModelServing: Sendable {
    func load(modelURL: URL) async throws
    func unload() async
    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> OpenAIEmbeddingsResult
}
