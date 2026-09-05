import Foundation

extension EnginePool {
    /// Projected request growth competes with every model class in this owner.
    /// Retained until work acknowledges completion; release is idempotent.
    public func reserveRequest(additionalBytes: Int64?) async throws -> UUID {
        await acquireLoadGate()
        defer { releaseLoadGate() }
        try Task.checkCancellation()
        let bytes = max(0, additionalBytes ?? 0)
        try await makeRoom(id: "request-growth", bytes: bytes)
        try Task.checkCancellation()
        let id = UUID()
        requestReservations[id] = bytes
        do {
            requestParentReservations[id] = try await reserveParent(
                resourceID: id.uuidString, kind: .requestGrowth,
                footprint: .init(bytes: additionalBytes.map { max(0, $0) }, source: .configuredEstimate),
                usesMLXAllocator: true
            )
            return id
        } catch {
            requestReservations[id] = nil
            throw error
        }
    }

    public func retainedAllocation(for id: UUID) -> RetainedAllocation? {
        guard parentBudget != nil, requestReservations[id] != nil, requestReleases[id] == nil else { return nil }
        if let existing = retainedRequestAllocations[id] { return existing }
        let allocation = RetainedAllocation(
            initialBytes: requestReservations[id] ?? 0,
            reconcile: { await self.recordRetainedRequestBytes($0, id: id) },
            teardown: { await self.releaseRequestCharge(id) }
        )
        retainedRequestAllocations[id] = allocation
        return allocation
    }

    private func recordRetainedRequestBytes(_ footprint: EnginePoolFootprint, id: UUID) async {
        guard let current = requestReservations[id] else { return }
        requestReservations[id] = max(current, footprint.bytes ?? current)
        await requestParentReservations[id]?.recordRetainedFootprint(footprint)
    }

    public func releaseRequest(_ id: UUID) async {
        if let retained = retainedRequestAllocations[id] {
            await retained.finishRequest()
        } else {
            await releaseRequestCharge(id)
        }
    }

    private func releaseRequestCharge(_ id: UUID) async {
        if let release = requestReleases[id] { await release.value; return }
        guard requestReservations[id] != nil else { return }
        let parent = requestParentReservations.removeValue(forKey: id)
        let release = Task {
            await self.releaseParent(parent)
            self.requestReservations[id] = nil
            self.retainedRequestAllocations[id] = nil
        }
        requestReleases[id] = release
        await release.value
        requestReleases[id] = nil
    }

    func checkPressureAdmission(id: String, additionalBytes: Int64) async throws {
        do {
            try await pressureAdmission?(additionalBytes)
        } catch let MemoryWatchdogError.admissionDenied(required, current, ceiling) {
            throw EnginePoolError.insufficientMemory(
                id: id, required: required, current: current, ceiling: ceiling
            )
        }
    }

    public func setPressureAdmission(_ check: @escaping @Sendable (Int64) async throws -> Void) {
        pressureAdmission = check
    }

    public func registerResource(_ resource: EnginePoolResource) {
        guard resources[resource.id] == nil else { return }
        resources[resource.id] = AuxiliaryPoolEntry(resource: resource)
    }

    public func acquireResource(_ id: String) async throws -> EnginePoolResourceLease {
        try Task.checkCancellation()
        guard var entry = resources[id] else {
            throw EnginePoolError.modelNotFound(id: id, available: resources.keys.sorted())
        }
        guard !entry.isUnloading else { throw EnginePoolError.modelBusy(id: id, operation: "load") }
        entry.waiting += 1
        resources[id] = entry
        defer { resources[id]?.waiting -= 1 }
        if !entry.loaded {
            let task: Task<Void, Error>
            if let existing = resourceLoads[id] {
                task = existing
            } else {
                task = Task { try await self.loadResource(id) }
                resourceLoads[id] = task
            }
            // Cancellation affects this waiter, never the shared allocation task.
            try await task.value
        }
        // Every request checks live pressure, including already-resident models.
        // The waiter count protects this resource while the callback may reclaim others.
        // Its model bytes are already resident; add only held request projections.
        try await checkPressureAdmission(id: id, additionalBytes: reservedRequestBytes)
        try Task.checkCancellation()
        let lease = EnginePoolResourceLease(resourceID: id, id: UUID())
        resources[id]?.leases.insert(lease.id)
        resources[id]?.lastAccess = Date()
        return lease
    }

    private func loadResource(_ id: String) async throws {
        await acquireLoadGate()
        defer {
            releaseLoadGate()
            resourceLoads[id] = nil
        }
        guard let entry = resources[id], !entry.loaded else { return }
        let footprint = entry.resource.footprint
        guard let bytes = footprint.bytes ?? (finalCeilingBytes == 0 ? 0 : nil) else {
            throw EnginePoolError.unknownFootprint(id: id)
        }
        if finalCeilingBytes > 0 && bytes > finalCeilingBytes {
            throw EnginePoolError.modelTooLarge(id: id, size: bytes, ceiling: finalCeilingBytes)
        }
        try await makeRoom(id: id, bytes: bytes)
        resources[id]?.isLoading = true
        currentModelMemory += bytes
        var loadStarted = false
        do {
            let parent = try await reserveParent(
                resourceID: id, kind: .auxiliary, footprint: footprint,
                usesMLXAllocator: entry.resource.usesMLXAllocator
            )
            resources[id]?.parentReservation = parent
            // The sampler already includes non-MLX pending reservations.
            let unobservedBytes = entry.resource.usesMLXAllocator ? bytes : 0
            try await checkPressureAdmission(
                id: id, additionalBytes: saturatingByteSum(unobservedBytes, reservedRequestBytes)
            )
            loadStarted = true
            try await entry.resource.load()
            try await markParentLoaded(resources[id]?.parentReservation)
            resources[id]?.loaded = true
            resources[id]?.isLoading = false
            resources[id]?.lastAccess = Date()
        } catch {
            // Clean partially created objects before releasing their reservation.
            if loadStarted { await entry.resource.unload() }
            let parent = resources[id]?.parentReservation
            resources[id]?.parentReservation = nil
            await releaseParent(parent)
            resources[id]?.isLoading = false
            currentModelMemory -= bytes
            throw error
        }
    }

    public func releaseResource(_ lease: EnginePoolResourceLease) {
        guard resources[lease.resourceID]?.leases.remove(lease.id) != nil else { return }
        resources[lease.resourceID]?.lastAccess = Date()
    }

    public func unloadResource(_ id: String) async throws {
        guard let entry = resources[id] else {
            throw EnginePoolError.modelNotFound(id: id, available: resources.keys.sorted())
        }
        guard !entry.isLoading, !entry.isUnloading, entry.waiting == 0, entry.leases.isEmpty else {
            throw EnginePoolError.modelBusy(id: id, operation: "unload")
        }
        guard entry.loaded else { return }
        _ = await unloadAuxiliaryResource(id)
    }

    func unloadAuxiliaryResource(_ id: String) async -> Bool {
        guard let entry = resources[id], entry.loaded, !entry.isLoading, !entry.isUnloading,
              entry.waiting == 0, entry.leases.isEmpty else { return false }
        resources[id]?.isUnloading = true
        await entry.resource.unload()
        let parent = resources[id]?.parentReservation
        resources[id]?.parentReservation = nil
        await releaseParent(parent)
        resources[id]?.loaded = false
        resources[id]?.isUnloading = false
        currentModelMemory -= entry.resource.footprint.bytes ?? 0
        return true
    }

    public func resourceStatuses() -> [EnginePoolResourceStatus] {
        resources.values.map {
            EnginePoolResourceStatus(resource: $0.resource, loaded: $0.loaded,
                                     isLoading: $0.isLoading, isUnloading: $0.isUnloading,
                                     inUse: $0.leases.count, lastAccess: $0.lastAccess)
        }.sorted { $0.resource.id < $1.resource.id }
    }

    /// Add only allocations absent from the MLX sampler. Pending non-MLX loads
    /// remain charged; estimates must never be presented as measured usage.
    public func nonMLXAccountedBytes() -> Int64 {
        resources.values.filter {
            !$0.resource.usesMLXAllocator && ($0.loaded || $0.isLoading)
        }.reduce(0) { $0 + ($1.resource.footprint.bytes ?? 0) }
    }

    func reclaimOldestIdleResource() async -> Int64? {
        let chatID = findLRUVictim()
        let auxiliary = resources.values.filter {
            $0.loaded && !$0.isLoading && !$0.isUnloading && $0.waiting == 0 && $0.leases.isEmpty
        }.min { ($0.lastAccess ?? .distantPast) < ($1.lastAccess ?? .distantPast) }
        let chatAccess = chatID.flatMap { entries[$0]?.lastAccess } ?? .distantFuture
        if let auxiliary, (auxiliary.lastAccess ?? .distantPast) < chatAccess {
            guard await unloadAuxiliaryResource(auxiliary.resource.id) else { return nil }
            return auxiliary.resource.footprint.bytes ?? 0
        }
        guard let chatID else { return nil }
        let bytes = entries[chatID].map { $0.actualSize ?? $0.estimatedSize } ?? 0
        guard await unloadLoadedModel(chatID) else { return nil }
        return bytes
    }

}
