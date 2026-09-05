import Foundation

/// All quantities are bytes. An estimate is not a measured working set; nil is
/// unknown, whereas zero explicitly describes a resource with no resident cost.
public struct EnginePoolFootprint: Equatable, Sendable {
    public enum Source: String, Sendable { case measured, diskEstimate, configuredEstimate, logicalExtent, unknown }
    public let bytes: Int64?
    public let source: Source

    public init(bytes: Int64?, source: Source) {
        self.bytes = bytes.flatMap { $0 >= 0 ? $0 : nil }
        self.source = self.bytes == nil ? .unknown : source
    }
}

public struct EnginePoolResource: Sendable {
    public let id: String
    public let modelID: String
    public let modelType: String
    public let footprint: EnginePoolFootprint
    /// Whether the live MLX allocator sampler already includes these allocations.
    public let usesMLXAllocator: Bool
    public let load: @Sendable () async throws -> Void
    /// Must return only once active work and the owner's retained model are gone.
    public let unload: @Sendable () async -> Void

    public init(id: String, modelID: String, modelType: String, footprint: EnginePoolFootprint,
                usesMLXAllocator: Bool, load: @escaping @Sendable () async throws -> Void,
                unload: @escaping @Sendable () async -> Void) {
        self.id = id
        self.modelID = modelID
        self.modelType = modelType
        self.footprint = footprint
        self.usesMLXAllocator = usesMLXAllocator
        self.load = load
        self.unload = unload
    }
}

public struct EnginePoolResourceLease: Sendable {
    public let resourceID: String
    public let id: UUID
}

public struct EnginePoolResourceStatus: Sendable {
    public let resource: EnginePoolResource
    public let loaded: Bool
    public let isLoading: Bool
    public let isUnloading: Bool
    public let inUse: Int
    public let lastAccess: Date?
}

/// An interface to the existing EnginePool owner, not a separate budget.
public protocol EnginePoolResourceManaging: Sendable {
    func registerResource(_ resource: EnginePoolResource) async
    func acquireResource(_ id: String) async throws -> EnginePoolResourceLease
    func releaseResource(_ lease: EnginePoolResourceLease) async
    func unloadResource(_ id: String) async throws
    func resourceStatuses() async -> [EnginePoolResourceStatus]
}
