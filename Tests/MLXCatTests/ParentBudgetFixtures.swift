import Foundation
import MLXCat

enum ParentBudgetFailure: Error { case denied, reconciliation }

actor ParentBudgetProbe: EnginePoolParentBudget {
    let limit: Int64
    let admissionGate: AccountingBarrier?
    let rejectOvershoot: Bool
    let releaseGate: AccountingBarrier?
    private(set) var allocations: [EnginePoolParentAllocation] = []
    private(set) var loaded: Set<UUID> = []
    private(set) var releaseCalls = 0
    private var charges: [UUID: Int64] = [:]
    var used: Int64 { charges.values.reduce(0, +) }

    init(limit: Int64, admissionGate: AccountingBarrier? = nil, rejectOvershoot: Bool = false,
         releaseGate: AccountingBarrier? = nil) {
        self.releaseGate = releaseGate
        self.limit = limit
        self.admissionGate = admissionGate
        self.rejectOvershoot = rejectOvershoot
    }

    func reserve(_ allocation: EnginePoolParentAllocation) async throws -> any EnginePoolParentReservation {
        guard let bytes = allocation.footprint.bytes, bytes >= 0, bytes <= limit - used else {
            throw ParentBudgetFailure.denied
        }
        allocations.append(allocation)
        charges[allocation.id] = bytes
        await admissionGate?.wait()
        return ParentReservationProbe(owner: self, id: allocation.id)
    }

    func recordLoad(_ id: UUID, footprint: EnginePoolFootprint?) throws {
        if let bytes = footprint?.bytes { charges[id] = bytes }
        if rejectOvershoot, used > limit { throw ParentBudgetFailure.reconciliation }
        loaded.insert(id)
    }

    func recordRetained(_ id: UUID, footprint: EnginePoolFootprint) {
        charges[id] = max(charges[id] ?? 0, footprint.bytes ?? 0)
        loaded.insert(id)
    }

    func release(_ id: UUID) async {
        releaseCalls += 1
        await releaseGate?.wait()
        charges[id] = nil
        loaded.remove(id)
    }
}

private struct ParentReservationProbe: EnginePoolParentReservation {
    let owner: ParentBudgetProbe
    let id: UUID
    func didLoad(actualFootprint: EnginePoolFootprint?) async throws {
        try await owner.recordLoad(id, footprint: actualFootprint)
    }
    func releaseAfterTeardown() async { await owner.release(id) }
    func recordRetainedFootprint(_ footprint: EnginePoolFootprint) async {
        await owner.recordRetained(id, footprint: footprint)
    }
}

actor ParentAccountingLoader: EnginePoolModelLoader {
    let loadGate: AccountingBarrier?
    let unloadGate: AccountingBarrier?
    let actual: Int64?
    private(set) var loads = 0
    private(set) var unloads = 0
    private(set) var loaderWasCancelled = false

    init(loadGate: AccountingBarrier? = nil, unloadGate: AccountingBarrier? = nil, actual: Int64? = nil) {
        self.loadGate = loadGate
        self.unloadGate = unloadGate
        self.actual = actual
    }
    func loadModel(id: String, modelURL: URL) async throws -> String {
        loads += 1
        await loadGate?.wait()
        loaderWasCancelled = Task.isCancelled
        return id
    }
    func loadModelMeasuringActualSize(id: String, modelURL: URL) async throws -> EnginePoolMeasuredLoad<String> {
        .init(engine: try await loadModel(id: id, modelURL: modelURL), actualSizeBytes: actual)
    }
    func unloadModel(_ engine: String, id: String) async {
        unloads += 1
        await unloadGate?.wait()
    }
}

func parentBudgetModel(_ bytes: Int64 = 60) -> DiscoveredModel {
    DiscoveredModel(id: "chat", modelURL: URL(fileURLWithPath: "/unused/chat"), estimatedSize: bytes)
}
