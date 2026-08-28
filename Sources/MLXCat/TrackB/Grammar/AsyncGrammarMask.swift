import Foundation

protocol GrammarMaskSnapshot: Sendable {
    /// Compute the allowed-token mask. Implementations MUST poll `isCancelled`
    /// between work units and return nil promptly when it flips true — a full
    /// vocabulary × prefix validation can take seconds-to-minutes on long
    /// prefixes, and an uncancellable compute outlives its request and starves
    /// the next generation (the Ornith-1.0-9B post-constrained-JSON stall,
    /// 2026-07-28).
    func allowedTokenIDs(isCancelled: @Sendable () -> Bool) -> [Int]?
}

extension GrammarMaskSnapshot {
    func allowedTokenIDs() -> [Int] {
        allowedTokenIDs(isCancelled: { false }) ?? []
    }
}

/// Off-path grammar-mask precomputation with two bounds the original lacked:
///
/// 1. **Coalescing** — at most one compute in flight plus one pending
///    ("latest wins"). Advancing state during a compute replaces the pending
///    snapshot instead of queueing one job per generated token; a 192-token
///    constrained generation keeps at most 2 outstanding computes, not 192.
/// 2. **Cancellation** — an in-flight compute observes staleness (state
///    advanced past it, or `invalidate()` on row teardown) and aborts instead
///    of finishing minutes of dead work.
final class AsyncGrammarMask<Snapshot: GrammarMaskSnapshot>: @unchecked Sendable {
    private let lock = NSLock()
    private var stateID = 0
    private var readyStateID: Int?
    private var readyAllowedTokenIDs: [Int]?
    private var computedMaskCount = 0
    private var inFlight = false
    private var pending: (snapshot: Snapshot, stateID: Int)?

    var readyTokenIDs: [Int]? {
        lock.lock()
        defer { lock.unlock() }
        guard readyStateID == stateID else {
            return nil
        }
        return readyAllowedTokenIDs
    }

    var completedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return computedMaskCount
    }

    func prepareCurrentState(from snapshot: Snapshot) {
        lock.lock()
        let targetStateID = stateID
        let launchNow = !inFlight
        if launchNow {
            inFlight = true
        } else {
            pending = (snapshot, targetStateID)
        }
        lock.unlock()
        if launchNow {
            launch(snapshot: snapshot, stateID: targetStateID)
        }
    }

    func prepareAdvancedState(from snapshot: Snapshot) {
        lock.lock()
        stateID += 1
        let targetStateID = stateID
        readyStateID = nil
        readyAllowedTokenIDs = nil
        let launchNow = !inFlight
        if launchNow {
            inFlight = true
        } else {
            pending = (snapshot, targetStateID)
        }
        lock.unlock()
        if launchNow {
            launch(snapshot: snapshot, stateID: targetStateID)
        }
    }

    /// Row teardown: makes any in-flight or pending compute stale so it aborts
    /// at its next cancellation poll instead of running to completion against
    /// a request that no longer exists.
    func invalidate() {
        lock.lock()
        stateID += 1
        readyStateID = nil
        readyAllowedTokenIDs = nil
        pending = nil
        lock.unlock()
    }

    private func launch(snapshot: Snapshot, stateID targetStateID: Int) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self, snapshot] in
            let allowedTokenIDs = snapshot.allowedTokenIDs(isCancelled: { [weak self] in
                guard let self else { return true }
                self.lock.lock()
                defer { self.lock.unlock() }
                return self.stateID != targetStateID
            })
            guard let self else { return }
            self.lock.lock()
            self.computedMaskCount += 1
            if let allowedTokenIDs, self.stateID == targetStateID {
                self.readyStateID = targetStateID
                self.readyAllowedTokenIDs = allowedTokenIDs
            }
            let next = self.pending
            self.pending = nil
            if next == nil {
                self.inFlight = false
            }
            self.lock.unlock()
            if let next {
                self.launch(snapshot: next.snapshot, stateID: next.stateID)
            }
        }
    }
}

struct PrecomputedGrammarMasks {
    var jsonAllowedTokenIDs: [Int]?
    var regexAllowedTokenIDs: [Int]?
    var gbnfAllowedTokenIDs: [Int]?
    var toolAllowedTokenIDs: [Int]?
}
