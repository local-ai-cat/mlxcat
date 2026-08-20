import Foundation

public struct OpenAIModelRuntimeStatus: Sendable, Equatable {
    public let id: String
    public let modelType: String
    public let modelPath: String
    public let loaded: Bool
    public let isLoading: Bool
    public let estimatedSize: Int64
    public let actualSize: Int64?
    public let pinned: Bool
    public let lastAccess: TimeInterval?
    public let inUse: Int

    public init(
        id: String,
        modelType: String = "llm",
        modelPath: String,
        loaded: Bool,
        isLoading: Bool,
        estimatedSize: Int64,
        actualSize: Int64?,
        pinned: Bool,
        lastAccess: TimeInterval?,
        inUse: Int
    ) {
        self.id = id
        self.modelType = modelType
        self.modelPath = modelPath
        self.loaded = loaded
        self.isLoading = isLoading
        self.estimatedSize = estimatedSize
        self.actualSize = actualSize
        self.pinned = pinned
        self.lastAccess = lastAccess
        self.inUse = inUse
    }
}

public struct OpenAIModelPoolStatus: Sendable, Equatable {
    public let finalCeiling: Int64
    public let currentModelMemory: Int64
    public let modelCount: Int
    public let loadedCount: Int
    public let models: [OpenAIModelRuntimeStatus]

    public init(
        finalCeiling: Int64,
        currentModelMemory: Int64,
        modelCount: Int,
        loadedCount: Int,
        models: [OpenAIModelRuntimeStatus]
    ) {
        self.finalCeiling = finalCeiling
        self.currentModelMemory = currentModelMemory
        self.modelCount = modelCount
        self.loadedCount = loadedCount
        self.models = models
    }
}

public struct OpenAIModelLifecycleResult: Sendable, Equatable {
    public let status: String
    public let modelID: String
    public let message: String?

    public init(status: String = "ok", modelID: String, message: String? = nil) {
        self.status = status
        self.modelID = modelID
        self.message = message
    }
}

public protocol OpenAIModelLifecycleBackend: OpenAIChatBackend {
    func modelPoolStatus() async throws -> OpenAIModelPoolStatus
    func loadModel(_ id: String) async throws -> OpenAIModelLifecycleResult
    func unloadModel(_ id: String) async throws -> OpenAIModelLifecycleResult
}
