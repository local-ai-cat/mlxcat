import Foundation

extension EnginePool {
    func getOrLoadEngine(
        _ modelID: String,
        takeLease: Bool,
        now: Date
    ) async throws -> (leaseID: UUID, loadResult: EnginePoolLoadResult) {
        while true {
            try Task.checkCancellation()
            guard var entry = entries[modelID] else {
                throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
            }

            if entry.engine != nil {
                let leaseID = UUID()
                entry.lastAccess = now
                if takeLease {
                    entry.activeLeaseIDs.insert(leaseID)
                }
                entries[modelID] = entry
                return (
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
    ) async throws -> (leaseID: UUID, loadResult: EnginePoolLoadResult) {
        while true {
            try Task.checkCancellation()
            guard var entry = entries[modelID] else {
                throw EnginePoolError.modelNotFound(id: modelID, available: availableModelIDs())
            }

            if entry.engine != nil {
                let leaseID = UUID()
                entry.lastAccess = now
                if takeLease {
                    entry.activeLeaseIDs.insert(leaseID)
                }
                entries[modelID] = entry
                return (
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

            try Task.checkCancellation()
            var loadingEntry = entries[modelID] ?? entry
            loadingEntry.isLoading = true
            loadingEntry.lastAccess = now
            entries[modelID] = loadingEntry
            currentModelMemory += loadingEntry.estimatedSize

            return try await finishAdmittedLoad(modelID, loadingEntry: loadingEntry, takeLease: takeLease, now: now)
        }
    }

    private func finishAdmittedLoad(
        _ modelID: String, loadingEntry: EnginePoolEntry<Loader.Engine>, takeLease: Bool, now: Date
    ) async throws -> (leaseID: UUID, loadResult: EnginePoolLoadResult) {
        do {
            let parent = try await reserveParent(
                resourceID: modelID, kind: .model,
                footprint: .init(bytes: loadingEntry.estimatedSize, source: .diskEstimate),
                usesMLXAllocator: true
            )
            entries[modelID]?.parentReservation = parent
            try await checkPressureAdmission(
                id: modelID, additionalBytes: saturatingByteSum(loadingEntry.estimatedSize, reservedRequestBytes)
            )
            try Task.checkCancellation()
            let modelURL = loadingEntry.modelURL
            let measured = try await Task { [loader] in
                try await loader.loadModelMeasuringActualSize(id: modelID, modelURL: modelURL)
            }.value
            let actualBytes = measured.actualSizeBytes.flatMap { $0 >= 0 ? $0 : nil }
            entries[modelID]?.actualSize = actualBytes
            currentModelMemory += (actualBytes ?? loadingEntry.estimatedSize) - loadingEntry.estimatedSize
            do {
                try await markParentLoaded(parent, measuredBytes: actualBytes)
            } catch {
                await loader.unloadModel(measured.engine, id: modelID)
                throw error
            }
            var loaded = entries[modelID] ?? loadingEntry
            let leaseID = UUID()
            loaded.engine = measured.engine
            loaded.actualSize = actualBytes
            loaded.isLoading = false
            loaded.lastAccess = now
            if takeLease && !Task.isCancelled { loaded.activeLeaseIDs.insert(leaseID) }
            entries[modelID] = loaded
            notifyModelLoadWaiters(modelID)
            try Task.checkCancellation()
            return (leaseID, EnginePoolLoadResult(modelID: modelID, alreadyLoaded: false))
        } catch {
            if entries[modelID]?.engine == nil {
                let parent = entries[modelID]?.parentReservation
                entries[modelID]?.parentReservation = nil
                await releaseParent(parent)
                currentModelMemory -= entries[modelID]?.actualSize ?? loadingEntry.estimatedSize
                entries[modelID]?.actualSize = nil
            }
            entries[modelID]?.isLoading = false
            notifyModelLoadWaiters(modelID)
            throw error
        }
    }

    func acquireLoadGate() async {
        if !loadGateLocked {
            loadGateLocked = true
            return
        }

        // NOTE: Gate waiters are not cancellation-aware; cancelled tasks stay queued until the gate resumes.
        await withCheckedContinuation { continuation in
            loadGateWaiters.append(continuation)
        }
    }

    func releaseLoadGate() {
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

        try await makeRoom(id: entry.modelID, bytes: entry.estimatedSize)
    }

    func makeRoom(id: String, bytes: Int64) async throws {
        guard finalCeilingBytes > 0 else { return }
        while bytes > max(0, max(0, finalCeilingBytes - currentModelMemory) - reservedRequestBytes) {
            guard await reclaimOldestIdleResource() != nil else {
                throw EnginePoolError.insufficientMemory(
                    id: id, required: bytes,
                    current: saturatingByteSum(currentModelMemory, reservedRequestBytes), ceiling: finalCeilingBytes
                )
            }
        }
    }

    func findLRUVictim() -> String? {
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

    /// Re-validates evictability before clearing state, so stale callers cannot unload
    /// a leased, pinned, or loading entry.
    @discardableResult
    func unloadLoadedModel(_ modelID: String) async -> Bool {
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
        entry.isLoading = true // Block acquisition until unload acknowledges completion.
        entries[modelID] = entry
        await loader.unloadModel(engine, id: modelID)
        let parent = entries[modelID]?.parentReservation
        entries[modelID]?.parentReservation = nil
        await releaseParent(parent)
        // Preserve pin changes made while the actor awaited teardown.
        entries[modelID]?.actualSize = nil
        entries[modelID]?.isLoading = false
        currentModelMemory = max(0, currentModelMemory - modelMemory)
        notifyModelLoadWaiters(modelID)
        return true
    }

}
