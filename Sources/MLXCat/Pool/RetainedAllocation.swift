import Foundation

/// One already-admitted request charge shared by transient work and its retained
/// cache slots. Slot references never obtain another parent allocation.
public final class RetainedAllocation: @unchecked Sendable {
    private let lock = NSLock()
    private var requestActive = true
    private var owners: [UUID: Int64] = [:]
    private var chargedBytes: Int64
    private var source: EnginePoolFootprint.Source = .configuredEstimate
    private let reconcile: @Sendable (EnginePoolFootprint) async -> Void
    private var reconciliation: Task<Void, Never>?
    private var teardown: (@Sendable () async -> Void)?
    private var completion: Task<Void, Never>?

    init(initialBytes: Int64, reconcile: @escaping @Sendable (EnginePoolFootprint) async -> Void,
         teardown: @escaping @Sendable () async -> Void) {
        chargedBytes = initialBytes
        self.reconcile = reconcile
        self.teardown = teardown
    }

    public func retainCache(bytes: Int64 = 0) throws -> RetainedAllocationLease {
        lock.lock()
        defer { lock.unlock() }
        let id = UUID()
        guard completion == nil else { throw ResourceOwnerError.unloaded }
        owners[id] = max(0, bytes)
        let total = owners.values.reduce(Int64(0)) { total, bytes in
            saturatingByteSum(total, bytes)
        }
        if total > chargedBytes {
            chargedBytes = total
            source = .logicalExtent
            enqueueReconciliation()
        }
        return RetainedAllocationLease(owner: self, id: id)
    }

    public func finishRequest() async {
        let task = finishTransient()
        await task?.value
    }

    public func waitForReleaseIfUnused() async {
        let task = pendingCompletion()
        await task?.value
    }

    fileprivate func releaseCache(_ id: UUID) {
        lock.lock()
        defer { lock.unlock() }
        owners[id] = nil
        startReleaseIfUnused()
    }

    private func finishTransient() -> Task<Void, Never>? {
        lock.lock()
        defer { lock.unlock() }
        if requestActive, !owners.isEmpty { enqueueReconciliation() }
        requestActive = false
        startReleaseIfUnused()
        return completion ?? reconciliation
    }

    private func pendingCompletion() -> Task<Void, Never>? {
        lock.lock()
        defer { lock.unlock() }
        return completion ?? reconciliation
    }

    private func enqueueReconciliation() {
        let previous = reconciliation
        let reconcile = self.reconcile
        let footprint = EnginePoolFootprint(bytes: chargedBytes, source: source)
        reconciliation = Task { await previous?.value; await reconcile(footprint) }
    }

    private func startReleaseIfUnused() {
        guard !requestActive, owners.isEmpty, completion == nil, let teardown else { return }
        self.teardown = nil
        let pending = reconciliation
        completion = Task { await pending?.value; await teardown() }
    }
}

public final class RetainedAllocationLease: @unchecked Sendable {
    public let owner: RetainedAllocation
    private let id: UUID
    fileprivate init(owner: RetainedAllocation, id: UUID) { self.owner = owner; self.id = id }
    /// Owner must first discard the retained payload. Duplicate releases are harmless.
    public func releaseAfterTeardown() { owner.releaseCache(id) }
    deinit { owner.releaseCache(id) }
}

func saturatingByteSum(_ lhs: Int64, _ rhs: Int64) -> Int64 {
    let sum = lhs.addingReportingOverflow(rhs)
    return sum.overflow ? Int64.max : sum.partialValue
}
