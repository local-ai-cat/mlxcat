import Foundation

/// Cache-owned payload and its existing request charge retire together. Cache
/// hits take their own ownership lease before borrowing arrays from this slot.
public final class RetainedCachePayload<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value?
    private let lease: RetainedAllocationLease?
    public var allocation: RetainedAllocation? { lease?.owner }

    public init(_ value: Value, allocation: RetainedAllocation?, bytes: Int64 = 0) throws {
        stored = value
        lease = try allocation?.retainCache(bytes: bytes)
    }

    public var value: Value? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    public func discard() {
        lock.lock()
        stored = nil
        lock.unlock()
        lease?.releaseAfterTeardown()
    }

    deinit {
        stored = nil
        lease?.releaseAfterTeardown()
    }
}
