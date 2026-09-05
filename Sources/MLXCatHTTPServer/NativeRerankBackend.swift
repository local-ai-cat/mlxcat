import Foundation
import MLXCat
import MLXCatHTTP

final class NativeRerankBackend: OpenAIRerankBackend, @unchecked Sendable {
    let rerankModels: [OpenAIModelInfo]

    private let modelURLsByID: [String: URL]
    private let cache: any NativeRerankModelServing
    private let pool: any EnginePoolResourceManaging

    init(models: [String: DiscoveredModel], pool: any EnginePoolResourceManaging,
         cache: any NativeRerankModelServing) async {
        self.modelURLsByID = models.mapValues(\.modelURL)
        self.rerankModels = models.keys.sorted().map { OpenAIModelInfo(id: $0, maxModelLength: nil) }
        self.cache = cache
        self.pool = pool
        for (id, model) in models {
            await pool.registerResource(EnginePoolResource(
                id: "rerank://\(id)", modelID: id, modelType: "rerank",
                footprint: EnginePoolFootprint(bytes: model.estimatedSize, source: .diskEstimate),
                usesMLXAllocator: true,
                load: { try await cache.load(id: id, modelURL: model.modelURL) },
                unload: { await cache.unload(id: id) }
            ))
        }
    }

    func rerank(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult {
        guard modelURLsByID[request.model] != nil else {
            throw OpenAIHTTPError(status: 404, message: "Rerank model '\(request.model)' not found.")
        }
        let id = request.model
        let lease: EnginePoolResourceLease
        do { lease = try await pool.acquireResource("rerank://\(id)") } catch let error as EnginePoolError {
            throw OpenAIHTTPError(status: error.httpStatus, message: error.description)
        }
        do {
            let result = try await cache.score(request)
            await pool.releaseResource(lease)
            return result
        } catch {
            await pool.releaseResource(lease)
            throw error
        }
    }

    static func isRerankModelDirectory(_ url: URL) -> Bool {
        let name = url.lastPathComponent.lowercased()
        let hasRerankNameHint = name.contains("rerank") || name.contains("reranker")
        guard
            let data = try? Data(contentsOf: url.appendingPathComponent("config.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return hasRerankNameHint
        }

        let modelType = (object["model_type"] as? String)?.lowercased() ?? ""
        let architectures = (object["architectures"] as? [String] ?? []).map { $0.lowercased() }
        if architectures.contains(where: { architecture in
            architecture.contains("sequenceclassification")
                || architecture == "jinaforranking"
                || architecture == "bertforsequenceclassification"
                || architecture == "qwen3forsequenceclassification"
        }) {
            return true
        }
        return hasRerankNameHint
            && (modelType.contains("qwen")
                    || architectures.contains(where: {
                        $0.contains("causallm") || $0 == "qwen3vlforconditionalgeneration"
                    }))
    }
}

/// Injection seam for CPU-only orchestration tests; shipping uses the factory below.
protocol NativeRerankModelServing: Sendable {
    func load(id: String, modelURL: URL) async throws
    func unload(id: String) async
    func score(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult
}
