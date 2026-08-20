import Foundation

public protocol EnginePoolModelLoader: Sendable {
    associatedtype Engine: Sendable

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
    /// behind its load gate, so a process-wide before/after delta is attributable
    /// to this one load.
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

    fileprivate init(id: UUID, modelID: String, engine: Engine) {
        self.id = id
        self.modelID = modelID
        self.engine = engine
    }
}

public struct EnginePoolQueueTicket: Sendable {
    public let id: UUID

    fileprivate init(id: UUID) {
        self.id = id
    }
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

    fileprivate init(modelID: String, engine: Engine) {
        self.modelID = modelID
        self.engine = engine
    }
}

public struct EnginePoolStatus: Equatable, Sendable {
    public let finalCeiling: Int64
    public let currentModelMemory: Int64
    public let modelCount: Int
    public let loadedCount: Int
    public let models: [EnginePoolModelStatus]
}

private struct EnginePoolEntry<Engine: Sendable>: Sendable {
    let modelID: String
    let modelURL: URL
    let estimatedSize: Int64
    var actualSize: Int64?
    var engine: Engine?
    var lastAccess: Date?
    var isLoading: Bool
    var isPinned: Bool
    var activeLeaseIDs: Set<UUID>

    var inUse: Int {
        activeLeaseIDs.count
    }
}

public actor EnginePool<Loader: EnginePoolModelLoader> {
    private var entries: [String: EnginePoolEntry<Loader.Engine>]
    private let loader: Loader
    private let finalCeilingBytes: Int64
    private let idleTimeout: TimeInterval?
    private let maxWaitingQueueDepth: Int
    private var currentModelMemory: Int64
    private var loadGateLocked: Bool
    private var loadGateWaiters: [CheckedContinuation<Void, Never>]
    private var modelLoadWaiters: [String: [CheckedContinuation<Void, Never>]]
    private var activeQueueTickets: Set<UUID>

    public init(
        models: [String: DiscoveredModel],
        loader: Loader,
        finalCeiling: Int64 = 0,
        idleTimeout: TimeInterval? = nil,
        maxWaitingQueueDepth: Int = 0
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
            guard let victim = findLRUVictim() else { break }
            let victimMemory = entries[victim].map { $0.actualSize ?? $0.estimatedSize } ?? 0
            guard await unloadLoadedModel(victim) else { break }
            freed += victimMemory
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

    private func getOrLoadEngine(
        _ modelID: String,
        takeLease: Bool,
        now: Date
    ) async throws -> (engine: Loader.Engine, leaseID: UUID, loadResult: EnginePoolLoadResult) {
        while true {
            guard var entry = entries[modelID] else {
                throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
            }

            if let engine = entry.engine {
                let leaseID = UUID()
                entry.lastAccess = now
                if takeLease {
                    entry.activeLeaseIDs.insert(leaseID)
                }
                entries[modelID] = entry
                return (
                    engine,
                    leaseID,
                    EnginePoolLoadResult(modelID: modelID, alreadyLoaded: true)
                )
            }

            if entry.isLoading {
                guard takeLease else {
                    throw EnginePoolError.modelLoading(id: modelID)
                }
                await waitForModelLoad(modelID)
                continue
            }

            await acquireLoadGate()
            do {
                let loaded = try await loadAfterAcquiringGate(
                    modelID,
                    takeLease: takeLease,
                    now: now
                )
                releaseLoadGate()
                return loaded
            } catch {
                releaseLoadGate()
                throw error
            }
        }
    }

    private func loadAfterAcquiringGate(
        _ modelID: String,
        takeLease: Bool,
        now: Date
    ) async throws -> (engine: Loader.Engine, leaseID: UUID, loadResult: EnginePoolLoadResult) {
        while true {
            guard var entry = entries[modelID] else {
                throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
            }

            if let engine = entry.engine {
                let leaseID = UUID()
                entry.lastAccess = now
                if takeLease {
                    entry.activeLeaseIDs.insert(leaseID)
                }
                entries[modelID] = entry
                return (
                    engine,
                    leaseID,
                    EnginePoolLoadResult(modelID: modelID, alreadyLoaded: true)
                )
            }

            if entry.isLoading {
                guard takeLease else {
                    throw EnginePoolError.modelLoading(id: modelID)
                }
                await waitForModelLoad(modelID)
                continue
            }

            try await admitModelForLoading(entry)

            var loadingEntry = entries[modelID] ?? entry
            loadingEntry.isLoading = true
            loadingEntry.lastAccess = now
            entries[modelID] = loadingEntry

            do {
                let measured = try await loader.loadModelMeasuringActualSize(
                    id: modelID,
                    modelURL: loadingEntry.modelURL
                )
                let engine = measured.engine
                var loaded = entries[modelID] ?? loadingEntry
                let leaseID = UUID()
                loaded.engine = engine
                loaded.actualSize = measured.actualSizeBytes
                loaded.isLoading = false
                loaded.lastAccess = now
                if takeLease {
                    loaded.activeLeaseIDs.insert(leaseID)
                }
                entries[modelID] = loaded
                currentModelMemory += loaded.actualSize ?? loaded.estimatedSize
                notifyModelLoadWaiters(modelID)
                return (
                    engine,
                    leaseID,
                    EnginePoolLoadResult(modelID: modelID, alreadyLoaded: false)
                )
            } catch {
                var failed = entries[modelID] ?? loadingEntry
                failed.isLoading = false
                entries[modelID] = failed
                notifyModelLoadWaiters(modelID)
                throw error
            }
        }
    }

    private func acquireLoadGate() async {
        if !loadGateLocked {
            loadGateLocked = true
            return
        }

        // NOTE: Gate waiters are not cancellation-aware; cancelled tasks stay queued until the gate resumes.
        await withCheckedContinuation { continuation in
            loadGateWaiters.append(continuation)
        }
    }

    private func releaseLoadGate() {
        if loadGateWaiters.isEmpty {
            loadGateLocked = false
            return
        }

        let next = loadGateWaiters.removeFirst()
        next.resume()
    }

    private func waitForModelLoad(_ modelID: String) async {
        // NOTE: Model-load waiters are not cancellation-aware; cancelled tasks self-heal when load completes.
        await withCheckedContinuation { continuation in
            modelLoadWaiters[modelID, default: []].append(continuation)
        }
    }

    private func notifyModelLoadWaiters(_ modelID: String) {
        let waiters = modelLoadWaiters.removeValue(forKey: modelID) ?? []
        for waiter in waiters {
            waiter.resume()
        }
    }

    private func admitModelForLoading(_ entry: EnginePoolEntry<Loader.Engine>) async throws {
        guard finalCeilingBytes > 0 else {
            return
        }

        if entry.estimatedSize > finalCeilingBytes {
            throw EnginePoolError.modelTooLarge(
                id: entry.modelID,
                size: entry.estimatedSize,
                ceiling: finalCeilingBytes
            )
        }

        while currentModelMemory + entry.estimatedSize > finalCeilingBytes {
            guard let victim = findLRUVictim() else {
                throw EnginePoolError.insufficientMemory(
                    id: entry.modelID,
                    required: entry.estimatedSize,
                    current: currentModelMemory,
                    ceiling: finalCeilingBytes
                )
            }
            _ = await unloadLoadedModel(victim)
        }
    }

    private func findLRUVictim() -> String? {
        entries.values
            .filter { entry in
                entry.engine != nil
                    && !entry.isPinned
                    && !entry.isLoading
                    && entry.inUse == 0
            }
            .sorted { left, right in
                (left.lastAccess ?? .distantPast) < (right.lastAccess ?? .distantPast)
            }
            .first?
            .modelID
    }

    /// Idempotently re-validates evictability before clearing state, so stale callers cannot unload a leased, pinned, or loading entry.
    @discardableResult
    private func unloadLoadedModel(_ modelID: String) async -> Bool {
        guard
            var entry = entries[modelID],
            let engine = entry.engine,
            entry.inUse == 0,
            !entry.isPinned,
            !entry.isLoading
        else {
            return false
        }

        let modelMemory = entry.actualSize ?? entry.estimatedSize
        entry.engine = nil
        entry.actualSize = nil
        entry.activeLeaseIDs.removeAll()
        currentModelMemory = max(0, currentModelMemory - modelMemory)
        entries[modelID] = entry

        await loader.unloadModel(engine, id: modelID)
        return true
    }
}
