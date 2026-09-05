import Foundation

public protocol EnginePoolModelLoader: Sendable {
    associatedtype Engine: Sendable

    /// A thrown load must acknowledge cleanup of any partially allocated state:
    /// no Engine is returned for the pool to unload on that failure path.
    func loadModel(id: String, modelURL: URL) async throws -> Engine
    func unloadModel(_ engine: Engine, id: String) async

    /// Loads a model and, when the loader can measure it, reports the model's real
    /// resident footprint. The pool then accounts for the model by measured bytes
    /// instead of `estimatedSize`.
    ///
    /// The default forwards to ``loadModel(id:modelURL:)`` and reports no
    /// measurement (`actualSizeBytes == nil`), so loaders that cannot measure —
    /// and the pool's own unit-test fakes — keep the estimate-based accounting.
    /// Loaders with access to a memory API (the MLX native loader) override this
    /// to bracket the load with a memory snapshot. The pool serializes loads
    /// behind its load gate. A process-wide delta can still include concurrent
    /// generation/freeing; it is not an isolated per-model measurement.
    func loadModelMeasuringActualSize(id: String, modelURL: URL) async throws -> EnginePoolMeasuredLoad<Engine>
}

public extension EnginePoolModelLoader {
    func loadModelMeasuringActualSize(id: String, modelURL: URL) async throws -> EnginePoolMeasuredLoad<Engine> {
        EnginePoolMeasuredLoad(engine: try await loadModel(id: id, modelURL: modelURL))
    }
}

/// A loaded engine paired with the real resident bytes it consumed, if the loader
/// measured it. `actualSizeBytes == nil` means "unmeasured — fall back to the
/// discovery-time estimate".
public struct EnginePoolMeasuredLoad<Engine: Sendable>: Sendable {
    public let engine: Engine
    public let actualSizeBytes: Int64?

    public init(engine: Engine, actualSizeBytes: Int64? = nil) {
        self.engine = engine
        self.actualSizeBytes = actualSizeBytes
    }
}

public struct EnginePoolLease<Engine: Sendable>: Sendable {
    public let id: UUID
    public let modelID: String
    public let engine: Engine

}

public struct EnginePoolQueueTicket: Sendable {
    public let id: UUID

}

public struct EnginePoolLoadResult: Equatable, Sendable {
    public let modelID: String
    public let alreadyLoaded: Bool

    public init(modelID: String, alreadyLoaded: Bool) {
        self.modelID = modelID
        self.alreadyLoaded = alreadyLoaded
    }
}

public struct EnginePoolModelStatus: Equatable, Sendable {
    public let id: String
    public let modelPath: String
    public let loaded: Bool
    public let isLoading: Bool
    public let estimatedSize: Int64
    public let actualSize: Int64?
    public let pinned: Bool
    public let lastAccess: TimeInterval?
    public let inUse: Int
}

public struct EnginePoolLoadedEngine<Engine: Sendable>: Sendable {
    public let modelID: String
    public let engine: Engine

}

public struct EnginePoolStatus: Equatable, Sendable {
    public let finalCeiling: Int64
    public let currentModelMemory: Int64
    public let modelCount: Int
    public let loadedCount: Int
    public let models: [EnginePoolModelStatus]
}

struct EnginePoolEntry<Engine: Sendable>: Sendable {
    let modelID: String
    let modelURL: URL
    let estimatedSize: Int64
    var actualSize: Int64?
    var parentReservation: (any EnginePoolParentReservation)?
    var engine: Engine?
    var lastAccess: Date?
    var isLoading: Bool
    var isPinned: Bool
    var activeLeaseIDs: Set<UUID>

    var inUse: Int {
        activeLeaseIDs.count
    }
}

struct AuxiliaryPoolEntry: Sendable {
    let resource: EnginePoolResource
    var parentReservation: (any EnginePoolParentReservation)?
    var loaded = false
    var isLoading = false
    var isUnloading = false
    var waiting = 0
    var leases: Set<UUID> = []
    var lastAccess: Date?
}
