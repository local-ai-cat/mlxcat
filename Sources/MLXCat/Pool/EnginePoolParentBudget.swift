import Foundation

/// Optional enclosing-process admission. EnginePool remains the allocation and
/// eviction owner; this parent must not independently reclaim its reservations.
public protocol EnginePoolParentBudget: Sendable {
    /// Commit the charge before returning. A thrown error must leave no new
    /// charge. Existing usage leases must never create another resident charge.
    func reserve(_ allocation: EnginePoolParentAllocation) async throws -> any EnginePoolParentReservation
}

public struct EnginePoolParentAllocation: Sendable {
    public enum Kind: String, Sendable { case model, auxiliary, requestGrowth }
    public let id: UUID
    public let resourceID: String
    public let kind: Kind
    public let footprint: EnginePoolFootprint
    public let usesMLXAllocator: Bool
}

/// A single allocation generation, held by the pool until owner teardown.
public protocol EnginePoolParentReservation: Sendable {
    /// Marks successful residency and reconciles an available measurement.
    /// Underestimates must remain charged as debt, never disappear on denial.
    /// Throwing refuses to expose the allocation; the pool unloads it first.
    func didLoad(actualFootprint: EnginePoolFootprint?) async throws
    /// Called only after allocation cleanup/unload or request teardown has
    /// acknowledged completion. Implementations must retain a failed release
    /// conservatively and must not abandon accounting on task cancellation.
    func releaseAfterTeardown() async
    /// Marks retained-cache residency, preserving the request estimate or a
    /// larger observed logical extent (which can share tensor backing). Record
    /// any excess as debt even above the limit; never silently drop an observed charge.
    func recordRetainedFootprint(_ footprint: EnginePoolFootprint) async
}

extension EnginePool {
    func reserveParent(resourceID: String, kind: EnginePoolParentAllocation.Kind,
                       footprint: EnginePoolFootprint, usesMLXAllocator: Bool)
        async throws -> (any EnginePoolParentReservation)? {
        guard let parentBudget else { return nil }
        let reservation = try await parentBudget.reserve(.init(
            id: UUID(), resourceID: resourceID, kind: kind,
            footprint: footprint, usesMLXAllocator: usesMLXAllocator
        ))
        do {
            try Task.checkCancellation()
            return reservation
        } catch {
            await releaseParent(reservation)
            throw error
        }
    }

    func releaseParent(_ reservation: (any EnginePoolParentReservation)?) async {
        guard let reservation else { return }
        await Task { await reservation.releaseAfterTeardown() }.value
    }

    func markParentLoaded(_ reservation: (any EnginePoolParentReservation)?, measuredBytes: Int64? = nil)
        async throws {
        guard let reservation else { return }
        let measured = measuredBytes.map { EnginePoolFootprint(bytes: $0, source: .measured) }
        // An individual cancelled load waiter cannot cancel shared residency.
        try await Task { try await reservation.didLoad(actualFootprint: measured) }.value
    }
}
