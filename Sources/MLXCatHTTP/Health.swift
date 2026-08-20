import Foundation

public struct OpenAIHealthInfo: Sendable {
    public let defaultModel: String?
    public let enginePool: OpenAIHealthEnginePool?

    public init(defaultModel: String?, enginePool: OpenAIHealthEnginePool?) {
        self.defaultModel = defaultModel
        self.enginePool = enginePool
    }
}

public struct OpenAIHealthEnginePool: Sendable {
    public let modelCount: Int
    public let loadedCount: Int

    public init(modelCount: Int, loadedCount: Int) {
        self.modelCount = modelCount
        self.loadedCount = loadedCount
    }
}

public protocol OpenAIHealthProviding: Sendable {
    var healthInfo: OpenAIHealthInfo { get }
}

public func buildHealthResponse(_ info: OpenAIHealthInfo) -> [String: Any] {
    var response: [String: Any] = [
        "status": "healthy",
        "default_model": info.defaultModel ?? NSNull(),
    ]
    if let enginePool = info.enginePool {
        response["engine_pool"] = [
            "model_count": enginePool.modelCount,
            "loaded_count": enginePool.loadedCount,
        ]
    } else {
        response["engine_pool"] = NSNull()
    }
    return response
}
