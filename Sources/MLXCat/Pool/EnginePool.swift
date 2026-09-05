import Foundation

public actor EnginePool<Loader: EnginePoolModelLoader>: EnginePoolResourceManaging {
    var entries: [String: EnginePoolEntry<Loader.Engine>]
    let loader: Loader
    let parentBudget: (any EnginePoolParentBudget)?
    var requestParentReservations: [UUID: any EnginePoolParentReservation] = [:]
    let finalCeilingBytes: Int64
    private let idleTimeout: TimeInterval?
    private let maxWaitingQueueDepth: Int
    var pressureAdmission: (@Sendable (Int64) async throws -> Void)?
    var requestReservations: [UUID: Int64] = [:]
    var retainedRequestAllocations: [UUID: RetainedAllocation] = [:]
    var requestReleases: [UUID: Task<Void, Never>] = [:]
    public var reservedRequestBytes: Int64 { requestReservations.values.reduce(0, saturatingByteSum) }

    var resources: [String: AuxiliaryPoolEntry] = [:]
    var resourceLoads: [String: Task<Void, Error>] = [:]
    var currentModelMemory: Int64
    var loadGateLocked: Bool
    var loadGateWaiters: [CheckedContinuation<Void, Never>]
    var modelLoadWaiters: [String: [CheckedContinuation<Void, Never>]]
    private var activeQueueTickets: Set<UUID>

    public init(
        models: [String: DiscoveredModel],
        loader: Loader,
        finalCeiling: Int64 = 0,
        idleTimeout: TimeInterval? = nil,
        maxWaitingQueueDepth: Int = 0,
        parentBudget: (any EnginePoolParentBudget)? = nil
    ) {
        var entries: [String: EnginePoolEntry<Loader.Engine>] = [:]
        for (id, model) in models {
            entries[id] = EnginePoolEntry(
                modelID: id,
                modelURL: model.modelURL,
                estimatedSize: model.estimatedSize,
                actualSize: nil,
                engine: nil,
                lastAccess: nil,
                isLoading: false,
                isPinned: false,
                activeLeaseIDs: []
            )
        }

        self.entries = entries
        self.loader = loader
        self.parentBudget = parentBudget
        self.finalCeilingBytes = max(0, finalCeiling)
        self.idleTimeout = idleTimeout
        self.maxWaitingQueueDepth = max(0, maxWaitingQueueDepth)
        self.currentModelMemory = 0
        self.loadGateLocked = false
        self.loadGateWaiters = []
        self.modelLoadWaiters = [:]
        self.activeQueueTickets = []
    }

    public func availableModelIDs() -> [String] {
        entries.keys.sorted()
    }

    public func setPinned(_ isPinned: Bool, for modelID: String) throws {
        guard var entry = entries[modelID] else {
            throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
        }
        entry.isPinned = isPinned
        entries[modelID] = entry
    }

    public func acquire(_ modelID: String, now: Date = Date()) async throws -> EnginePoolLease<Loader.Engine> {
        let engine = try await getOrLoadEngine(modelID, takeLease: true, now: now)
        guard let entry = entries[modelID], let loadedEngine = entry.engine else {
            throw EnginePoolError.modelLoading(id: modelID)
        }
        return EnginePoolLease(id: engine.leaseID, modelID: modelID, engine: loadedEngine)
    }

    public func release(_ lease: EnginePoolLease<Loader.Engine>) async {
        guard var entry = entries[lease.modelID] else { return }
        guard entry.activeLeaseIDs.remove(lease.id) != nil else { return }
        entries[lease.modelID] = entry
    }

    public func isLoaded(_ modelID: String) throws -> Bool {
        guard let entry = entries[modelID] else {
            throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
        }
        return entry.engine != nil
    }

    @discardableResult
    public func load(_ modelID: String, now: Date = Date()) async throws -> EnginePoolLoadResult {
        try await getOrLoadEngine(modelID, takeLease: false, now: now).loadResult
    }

    public func unload(_ modelID: String) async throws {
        guard let entry = entries[modelID] else {
            throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
        }
        guard entry.engine != nil else {
            throw EnginePoolError.modelNotLoaded(id: modelID)
        }
        guard !entry.isLoading, entry.inUse == 0, !entry.isPinned else {
            throw EnginePoolError.modelBusy(id: modelID, operation: "unload")
        }

        guard await unloadLoadedModel(modelID) else {
            throw EnginePoolError.modelBusy(id: modelID, operation: "unload")
        }
    }

    public func sweepIdleModels(now: Date = Date()) async -> [String] {
        guard let idleTimeout, idleTimeout > 0 else {
            return []
        }

        var unloaded: [String] = []
        let candidates = entries.values
            .filter { entry in
                guard let lastAccess = entry.lastAccess else { return false }
                return entry.engine != nil
                    && !entry.isPinned
                    && !entry.isLoading
                    && entry.inUse == 0
                    && now.timeIntervalSince(lastAccess) >= idleTimeout
            }
            .sorted { left, right in
                (left.lastAccess ?? .distantPast) < (right.lastAccess ?? .distantPast)
            }

        for candidate in candidates {
            guard
                let entry = entries[candidate.modelID],
                entry.engine != nil,
                !entry.isPinned,
                !entry.isLoading,
                entry.inUse == 0,
                let lastAccess = entry.lastAccess,
                now.timeIntervalSince(lastAccess) >= idleTimeout
            else {
                continue
            }

            if await unloadLoadedModel(candidate.modelID) {
                unloaded.append(candidate.modelID)
            }
        }
        let auxiliaryIDs = resources.values.filter {
            $0.loaded && !$0.isLoading && !$0.isUnloading && $0.waiting == 0 && $0.leases.isEmpty
                && now.timeIntervalSince($0.lastAccess ?? now) >= idleTimeout
        }.sorted { ($0.lastAccess ?? .distantPast) < ($1.lastAccess ?? .distantPast) }.map { $0.resource.id }
        for id in auxiliaryIDs where await unloadAuxiliaryResource(id) { unloaded.append(id) }
        return unloaded
    }

    /// Unloads idle (loaded, unpinned, not-in-use, not-loading) models oldest-first
    /// until at least `targetBytes` of accounted model memory is freed or no idle
    /// victim remains. Returns the bytes freed, counted by measured `actualSize`
    /// when known and `estimatedSize` otherwise — the same quantity
    /// `currentModelMemory` tracks. This is the eviction step the live memory
    /// watchdog drives; unlike `sweepIdleModels` it is age-agnostic and pressure-
    /// driven rather than idle-timeout-driven.
    @discardableResult
    public func reclaimIdleModels(targetBytes: Int64) async -> Int64 {
        guard targetBytes > 0 else { return 0 }

        var freed: Int64 = 0
        while freed < targetBytes {
            guard let released = await reclaimOldestIdleResource() else { break }
            freed += released
        }
        return freed
    }

    public func status() -> EnginePoolStatus {
        let modelStatuses = entries.values
            .sorted { $0.modelID < $1.modelID }
            .map { entry in
                EnginePoolModelStatus(
                    id: entry.modelID,
                    modelPath: entry.modelURL.path,
                    loaded: entry.engine != nil,
                    isLoading: entry.isLoading,
                    estimatedSize: entry.estimatedSize,
                    actualSize: entry.actualSize,
                    pinned: entry.isPinned,
                    lastAccess: entry.lastAccess?.timeIntervalSince1970,
                    inUse: entry.inUse
                )
            }

        return EnginePoolStatus(
            finalCeiling: finalCeilingBytes,
            currentModelMemory: currentModelMemory,
            modelCount: entries.count,
            loadedCount: modelStatuses.filter(\.loaded).count,
            models: modelStatuses
        )
    }

    public func loadedEngines() -> [EnginePoolLoadedEngine<Loader.Engine>] {
        entries.values.compactMap { entry in
            guard let engine = entry.engine else { return nil }
            return EnginePoolLoadedEngine(modelID: entry.modelID, engine: engine)
        }
    }

    public func admitWaitingRequest() throws -> EnginePoolQueueTicket {
        let ticket = EnginePoolQueueTicket(id: UUID())
        guard maxWaitingQueueDepth > 0 else {
            return ticket
        }
        let currentDepth = activeQueueTickets.count
        guard currentDepth < maxWaitingQueueDepth else {
            throw EnginePoolError.schedulerQueueFull(current: currentDepth, max: maxWaitingQueueDepth)
        }
        activeQueueTickets.insert(ticket.id)
        return ticket
    }

    public func finishWaitingRequest(_ ticket: EnginePoolQueueTicket) {
        activeQueueTickets.remove(ticket.id)
    }

}
