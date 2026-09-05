import Foundation
import MLXCat
import MLXCatHTTP

/// Resolves the same registered resources used by admission and status.
/// Canonical resource IDs are unambiguous; chat retains precedence for bare IDs.
struct PoolResourceModelLifecycle: Sendable {
    let pool: any EnginePoolResourceManaging

    func load(_ reference: String, allowBareModelID: Bool) async throws -> OpenAIModelLifecycleResult? {
        guard let resource = await resolve(reference, allowBareModelID: allowBareModelID) else { return nil }
        do {
            let lease = try await pool.acquireResource(resource.id)
            await pool.releaseResource(lease)
            return OpenAIModelLifecycleResult(modelID: reference)
        } catch let error as EnginePoolError {
            throw OpenAIHTTPError(status: error.httpStatus, message: error.description)
        }
    }

    func unload(_ reference: String, allowBareModelID: Bool) async throws -> OpenAIModelLifecycleResult? {
        guard let resource = await resolve(reference, allowBareModelID: allowBareModelID) else { return nil }
        do {
            try await pool.unloadResource(resource.id)
            return OpenAIModelLifecycleResult(modelID: reference)
        } catch let error as EnginePoolError {
            throw OpenAIHTTPError(status: error.httpStatus, message: error.description)
        }
    }

    private func resolve(_ reference: String, allowBareModelID: Bool) async -> EnginePoolResource? {
        let resources = await pool.resourceStatuses().map(\.resource)
        if let canonical = resources.first(where: { $0.id == reference }) { return canonical }
        guard allowBareModelID else { return nil }
        let matches = resources.filter { $0.modelID == reference }
        guard matches.count == 1 else { return nil }
        return matches.first
    }
}
