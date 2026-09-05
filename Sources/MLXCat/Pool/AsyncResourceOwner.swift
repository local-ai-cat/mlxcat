import Foundation

/// A stable handle whose resident payload can be explicitly invalidated. Borrowed
/// operations finish before teardown; retaining the handle never retains a retired payload.
public actor AsyncResourceOwner<Resource: Sendable> {
    private var resource: Resource?
    private var active = 0
    private var closing = false
    private var drained: [CheckedContinuation<Void, Never>] = []
    private var completion: Task<Void, Never>?

    public init(_ resource: Resource) { self.resource = resource }

    public func withValue<Result>(
        _ operation: @Sendable (Resource) async throws -> sending Result
    ) async throws -> sending Result {
        guard !closing, let resource else { throw ResourceOwnerError.unloaded }
        active += 1
        defer {
            active -= 1
            if active == 0 {
                let waiters = drained
                drained.removeAll()
                for waiter in waiters { waiter.resume() }
            }
        }
        return try await operation(resource)
    }

    /// Terminal callbacks must join an already-started invalidation instead of
    /// mistaking a rejected borrow for acknowledged resource teardown.
    public func withValueOrAwaitInvalidation(_ operation: @Sendable (Resource) async -> Void) async {
        if let completion { await completion.value; return }
        try? await withValue { await operation($0) }
    }

    public func invalidate(teardown: @escaping @Sendable (Resource) async -> Void) async {
        if let completion { await completion.value; return }
        closing = true
        let task = Task {
            if self.active > 0 {
                await withCheckedContinuation { self.drained.append($0) }
            }
            await self.dispose(teardown: teardown)
        }
        completion = task
        await task.value
    }

    private func dispose(teardown: @Sendable (Resource) async -> Void) async {
        if let resource { await teardown(resource) }
        resource = nil
        // This frame must return before invalidation is acknowledged: a local
        // payload reference used by teardown cannot outlive the acknowledgement.
    }
}

public enum ResourceOwnerError: Error { case unloaded }
