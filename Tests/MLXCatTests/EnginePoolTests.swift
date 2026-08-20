import Foundation
@testable import MLXCat
import XCTest

private struct FakeEngine: Equatable, Sendable {
    let id: String
}

private actor FakeLoaderState {
    private(set) var loaded: [String] = []
    private(set) var unloaded: [String] = []
    private var activeLoads: Set<String> = []
    private var observedOverlap = false

    func recordLoad(_ id: String) {
        loaded.append(id)
    }

    func beginLoad(_ id: String) {
        if !activeLoads.isEmpty {
            observedOverlap = true
        }
        activeLoads.insert(id)
        loaded.append(id)
    }

    func endLoad(_ id: String) {
        activeLoads.remove(id)
    }

    func recordUnload(_ id: String) {
        unloaded.append(id)
    }

    func loadedIDs() -> [String] {
        loaded
    }

    func unloadedIDs() -> [String] {
        unloaded
    }

    func didObserveOverlap() -> Bool {
        observedOverlap
    }
}

private struct FakeLoader: EnginePoolModelLoader {
    let state: FakeLoaderState

    func loadModel(id: String, modelURL: URL) async throws -> FakeEngine {
        await state.recordLoad(id)
        return FakeEngine(id: id)
    }

    func unloadModel(_ engine: FakeEngine, id: String) async {
        await state.recordUnload(id)
    }
}

private actor BlockingLoadCoordinator {
    private var startedCount = 0
    private var loadContinuations: [CheckedContinuation<Void, Never>] = []
    private var startedWaiters: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []

    func waitUntilLoadStarted(count: Int) async {
        if startedCount >= count {
            return
        }
        await withCheckedContinuation { continuation in
            startedWaiters.append((count, continuation))
        }
    }

    func waitForReleaseAfterMarkingStarted() async {
        startedCount += 1
        resumeStartedWaitersIfNeeded()
        await withCheckedContinuation { continuation in
            loadContinuations.append(continuation)
        }
    }

    func releaseOne() {
        guard !loadContinuations.isEmpty else { return }
        let continuation = loadContinuations.removeFirst()
        continuation.resume()
    }

    func releaseAll() {
        let continuations = loadContinuations
        loadContinuations.removeAll()
        for continuation in continuations {
            continuation.resume()
        }
    }

    private func resumeStartedWaitersIfNeeded() {
        var remaining: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []
        for waiter in startedWaiters {
            if startedCount >= waiter.count {
                waiter.continuation.resume()
            } else {
                remaining.append(waiter)
            }
        }
        startedWaiters = remaining
    }
}

private struct BlockingFakeLoader: EnginePoolModelLoader {
    let state: FakeLoaderState
    let coordinator: BlockingLoadCoordinator

    func loadModel(id: String, modelURL: URL) async throws -> FakeEngine {
        await state.beginLoad(id)
        await coordinator.waitForReleaseAfterMarkingStarted()
        await state.endLoad(id)
        return FakeEngine(id: id)
    }

    func unloadModel(_ engine: FakeEngine, id: String) async {
        await state.recordUnload(id)
    }
}

private actor BlockingUnloadCoordinator {
    private var startedCount = 0
    private var unloadContinuations: [CheckedContinuation<Void, Never>] = []
    private var startedWaiters: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []

    func waitUntilUnloadStarted(count: Int) async {
        if startedCount >= count {
            return
        }
        await withCheckedContinuation { continuation in
            startedWaiters.append((count, continuation))
        }
    }

    func waitForReleaseAfterMarkingStarted() async {
        startedCount += 1
        resumeStartedWaitersIfNeeded()
        await withCheckedContinuation { continuation in
            unloadContinuations.append(continuation)
        }
    }

    func releaseOne() {
        guard !unloadContinuations.isEmpty else { return }
        let continuation = unloadContinuations.removeFirst()
        continuation.resume()
    }

    private func resumeStartedWaitersIfNeeded() {
        var remaining: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []
        for waiter in startedWaiters {
            if startedCount >= waiter.count {
                waiter.continuation.resume()
            } else {
                remaining.append(waiter)
            }
        }
        startedWaiters = remaining
    }
}

private struct BlockingUnloadFakeLoader: EnginePoolModelLoader {
    let state: FakeLoaderState
    let coordinator: BlockingUnloadCoordinator

    func loadModel(id: String, modelURL: URL) async throws -> FakeEngine {
        await state.recordLoad(id)
        return FakeEngine(id: id)
    }

    func unloadModel(_ engine: FakeEngine, id: String) async {
        await coordinator.waitForReleaseAfterMarkingStarted()
        await state.recordUnload(id)
    }
}

private struct MeasuringFakeLoader: EnginePoolModelLoader {
    let state: FakeLoaderState
    let measuredSizes: [String: Int64]

    func loadModel(id: String, modelURL: URL) async throws -> FakeEngine {
        await state.recordLoad(id)
        return FakeEngine(id: id)
    }

    func loadModelMeasuringActualSize(
        id: String,
        modelURL: URL
    ) async throws -> EnginePoolMeasuredLoad<FakeEngine> {
        let engine = try await loadModel(id: id, modelURL: modelURL)
        return EnginePoolMeasuredLoad(engine: engine, actualSizeBytes: measuredSizes[id])
    }

    func unloadModel(_ engine: FakeEngine, id: String) async {
        await state.recordUnload(id)
    }
}

final class EnginePoolTests: XCTestCase {
    func testMeasuredActualSizeReplacesEstimateInAccounting() async throws {
        var models: [String: DiscoveredModel] = [:]
        models["a"] = DiscoveredModel(
            id: "a",
            modelURL: URL(fileURLWithPath: "/tmp/a", isDirectory: true),
            estimatedSize: 100
        )
        let pool = EnginePool(
            models: models,
            loader: MeasuringFakeLoader(state: FakeLoaderState(), measuredSizes: ["a": 42])
        )

        try await pool.load("a")

        let status = await pool.status()
        XCTAssertEqual(status.models.first?.actualSize, 42)
        XCTAssertEqual(status.models.first?.estimatedSize, 100)
        XCTAssertEqual(status.currentModelMemory, 42)
    }

    func testUnmeasuredLoadFallsBackToEstimateInAccounting() async throws {
        var models: [String: DiscoveredModel] = [:]
        models["a"] = DiscoveredModel(
            id: "a",
            modelURL: URL(fileURLWithPath: "/tmp/a", isDirectory: true),
            estimatedSize: 100
        )
        // measuredSizes has no entry for "a" -> reports nil -> pool uses estimate.
        let pool = EnginePool(
            models: models,
            loader: MeasuringFakeLoader(state: FakeLoaderState(), measuredSizes: [:])
        )

        try await pool.load("a")

        let status = await pool.status()
        XCTAssertNil(status.models.first?.actualSize)
        XCTAssertEqual(status.currentModelMemory, 100)
    }

    func testReclaimIdleModelsFreesOldestUnpinnedIdleFirst() async throws {
        let state = FakeLoaderState()
        let pool = makePool(["a": 30, "b": 40, "c": 50], state: state)
        let start = Date(timeIntervalSince1970: 1_000)
        try await pool.load("a", now: start)
        try await pool.load("b", now: start.addingTimeInterval(10))
        try await pool.load("c", now: start.addingTimeInterval(20))

        // Ask for 50 bytes: should evict a (30) then b (40) = 70 >= 50, leaving c.
        let freed = await pool.reclaimIdleModels(targetBytes: 50)

        XCTAssertEqual(freed, 70)
        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["c"])
        let unloaded = await state.unloadedIDs()
        XCTAssertEqual(unloaded, ["a", "b"])
    }

    func testReclaimIdleModelsSkipsPinnedAndLeased() async throws {
        let state = FakeLoaderState()
        let pool = makePool(["a": 30, "b": 40], state: state)
        try await pool.load("a")
        try await pool.setPinned(true, for: "a")
        let lease = try await pool.acquire("b")

        let freed = await pool.reclaimIdleModels(targetBytes: 1_000)

        XCTAssertEqual(freed, 0)
        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["a", "b"])
        await pool.release(lease)
    }

    func testOnDemandLoadTakesAndReleasesLease() async throws {
        let pool = makePool(["a": 10])

        let lease = try await pool.acquire("a")
        var status = await pool.status()
        XCTAssertEqual(status.loadedCount, 1)
        XCTAssertEqual(status.currentModelMemory, 10)
        XCTAssertEqual(status.models.first?.inUse, 1)
        XCTAssertEqual(lease.engine.id, "a")

        await pool.release(lease)
        status = await pool.status()
        XCTAssertEqual(status.models.first?.inUse, 0)
    }

    func testLRUEvictionUsesOldestUnpinnedIdleModel() async throws {
        let state = FakeLoaderState()
        let pool = makePool(["a": 60, "b": 60, "c": 60], finalCeiling: 120, state: state)
        let start = Date(timeIntervalSince1970: 1_000)

        try await pool.load("a", now: start)
        try await pool.load("b", now: start.addingTimeInterval(10))
        try await pool.load("c", now: start.addingTimeInterval(20))

        let status = await pool.status()
        XCTAssertEqual(status.currentModelMemory, 120)
        XCTAssertEqual(loadedModelIDs(status), ["b", "c"])
        let unloaded = await state.unloadedIDs()
        XCTAssertEqual(unloaded, ["a"])
    }

    func testPinnedModelPreventsEviction() async throws {
        let pool = makePool(["a": 60, "b": 60], finalCeiling: 60)
        try await pool.load("a")
        try await pool.setPinned(true, for: "a")

        await XCTAssertThrowsEnginePoolError(try await pool.load("b")) { error in
            XCTAssertEqual(error.httpStatus, 507)
            if case .insufficientMemory(let id, _, _, _) = error {
                XCTAssertEqual(id, "b")
            } else {
                XCTFail("expected insufficient memory")
            }
        }

        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["a"])
    }

    func testLeasePreventsEvictionUntilReleased() async throws {
        let state = FakeLoaderState()
        let pool = makePool(["a": 60, "b": 60], finalCeiling: 60, state: state)
        let lease = try await pool.acquire("a")

        await XCTAssertThrowsEnginePoolError(try await pool.load("b")) { error in
            XCTAssertEqual(error.httpStatus, 507)
        }
        var unloaded = await state.unloadedIDs()
        XCTAssertEqual(unloaded, [])

        await pool.release(lease)
        try await pool.load("b")

        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["b"])
        unloaded = await state.unloadedIDs()
        XCTAssertEqual(unloaded, ["a"])
    }

    func testModelTooLargeUses507() async throws {
        let pool = makePool(["huge": 100], finalCeiling: 50)

        await XCTAssertThrowsEnginePoolError(try await pool.load("huge")) { error in
            XCTAssertEqual(error.httpStatus, 507)
            if case .modelTooLarge(let id, let size, let ceiling) = error {
                XCTAssertEqual(id, "huge")
                XCTAssertEqual(size, 100)
                XCTAssertEqual(ceiling, 50)
            } else {
                XCTFail("expected model too large")
            }
        }
    }

    func testIdleTimeoutUnloadsOnlyUnpinnedIdleModels() async throws {
        let pool = makePool(["a": 10, "b": 10], idleTimeout: 30)
        let start = Date(timeIntervalSince1970: 1_000)

        try await pool.load("a", now: start)
        try await pool.load("b", now: start.addingTimeInterval(1))
        try await pool.setPinned(true, for: "b")

        let unloaded = await pool.sweepIdleModels(now: start.addingTimeInterval(40))

        XCTAssertEqual(unloaded, ["a"])
        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["b"])
    }

    func testIdleSweepSkipsStaleCandidateThatGetsLeasedDuringPriorUnload() async throws {
        let state = FakeLoaderState()
        let coordinator = BlockingUnloadCoordinator()
        let pool = makeBlockingUnloadPool(
            ["a": 10, "b": 20],
            idleTimeout: 30,
            state: state,
            coordinator: coordinator
        )
        let start = Date(timeIntervalSince1970: 1_000)
        try await pool.load("a", now: start)
        try await pool.load("b", now: start.addingTimeInterval(1))

        async let sweepResult: [String] = pool.sweepIdleModels(now: start.addingTimeInterval(40))
        await coordinator.waitUntilUnloadStarted(count: 1)

        let lease = try await pool.acquire("b")
        await coordinator.releaseOne()
        let unloaded = await sweepResult

        let status = await pool.status()
        XCTAssertEqual(unloaded, ["a"])
        XCTAssertEqual(loadedModelIDs(status), ["b"])
        XCTAssertEqual(status.models.first(where: { $0.id == "b" })?.inUse, 1)
        XCTAssertEqual(status.currentModelMemory, 20)
        assertMemoryAccountingInvariant(status)
        XCTAssertEqual(lease.engine.id, "b")

        await pool.release(lease)
    }

    func testIdleSweepSkipsCandidateReusedAndReleasedDuringPriorUnload() async throws {
        let state = FakeLoaderState()
        let coordinator = BlockingUnloadCoordinator()
        let pool = makeBlockingUnloadPool(
            ["a": 10, "b": 20],
            idleTimeout: 30,
            state: state,
            coordinator: coordinator
        )
        let start = Date(timeIntervalSince1970: 1_000)
        let sweepNow = start.addingTimeInterval(40)
        try await pool.load("a", now: start)
        try await pool.load("b", now: start.addingTimeInterval(1))

        async let sweepResult: [String] = pool.sweepIdleModels(now: sweepNow)
        await coordinator.waitUntilUnloadStarted(count: 1)

        let lease = try await pool.acquire("b", now: sweepNow)
        await pool.release(lease)
        await coordinator.releaseOne()
        let unloaded = await sweepResult

        let status = await pool.status()
        XCTAssertEqual(unloaded, ["a"])
        XCTAssertEqual(loadedModelIDs(status), ["b"])
        XCTAssertEqual(status.models.first(where: { $0.id == "b" })?.inUse, 0)
        XCTAssertEqual(status.currentModelMemory, 20)
        assertMemoryAccountingInvariant(status)
        XCTAssertEqual(lease.engine.id, "b")
    }

    func testUnknownModelErrorListsAvailableModels() async throws {
        let pool = makePool(["a": 10, "b": 10])

        await XCTAssertThrowsEnginePoolError(try await pool.load("missing")) { error in
            XCTAssertEqual(error.httpStatus, 404)
            XCTAssertEqual(error.message, "Model 'missing' not found. Available models: a, b")
        }
    }

    func testSchedulerQueueFullHasRetryAfter() async throws {
        let pool = makePool(["a": 10], maxWaitingQueueDepth: 1)
        _ = try await pool.admitWaitingRequest()

        await XCTAssertThrowsEnginePoolError(try await pool.admitWaitingRequest()) { error in
            XCTAssertEqual(error.httpStatus, 503)
            XCTAssertEqual(error.retryAfterSeconds, 1)
            XCTAssertEqual(error.message, "Scheduler waiting queue full (1/1). Try again shortly.")
        }
    }

    func testConcurrentDifferentModelLoadsWaitAndSerialize() async throws {
        let state = FakeLoaderState()
        let coordinator = BlockingLoadCoordinator()
        let pool = makeBlockingPool(["a": 10, "b": 20], state: state, coordinator: coordinator)

        async let loadA: EnginePoolLoadResult = pool.load("a")
        await coordinator.waitUntilLoadStarted(count: 1)
        async let loadB: EnginePoolLoadResult = pool.load("b")

        await Task.yield()
        var observedOverlap = await state.didObserveOverlap()
        XCTAssertFalse(observedOverlap)

        await coordinator.releaseOne()
        _ = try await loadA
        await coordinator.waitUntilLoadStarted(count: 2)
        observedOverlap = await state.didObserveOverlap()
        XCTAssertFalse(observedOverlap)

        await coordinator.releaseOne()
        _ = try await loadB

        let status = await pool.status()
        XCTAssertEqual(status.currentModelMemory, 30)
        XCTAssertEqual(loadedModelIDs(status), ["a", "b"])
        let loadedIDs = await state.loadedIDs()
        XCTAssertEqual(loadedIDs, ["a", "b"])
        observedOverlap = await state.didObserveOverlap()
        XCTAssertFalse(observedOverlap)
    }

    func testConcurrentSameModelAcquireSharesSingleLoad() async throws {
        let state = FakeLoaderState()
        let coordinator = BlockingLoadCoordinator()
        let pool = makeBlockingPool(["a": 10], state: state, coordinator: coordinator)

        async let first: EnginePoolLease<FakeEngine> = pool.acquire("a")
        await coordinator.waitUntilLoadStarted(count: 1)
        async let second: EnginePoolLease<FakeEngine> = pool.acquire("a")

        await Task.yield()
        await coordinator.releaseOne()

        let firstLease = try await first
        let secondLease = try await second

        XCTAssertEqual(firstLease.engine.id, "a")
        XCTAssertEqual(secondLease.engine.id, "a")
        let loadedIDs = await state.loadedIDs()
        XCTAssertEqual(loadedIDs, ["a"])

        var status = await pool.status()
        XCTAssertEqual(status.currentModelMemory, 10)
        XCTAssertEqual(status.models.first?.inUse, 2)

        await pool.release(firstLease)
        await pool.release(secondLease)
        status = await pool.status()
        XCTAssertEqual(status.models.first?.inUse, 0)
    }

    func testConcurrentSameModelExplicitLoadReportsLoading() async throws {
        let state = FakeLoaderState()
        let coordinator = BlockingLoadCoordinator()
        let pool = makeBlockingPool(["a": 10], state: state, coordinator: coordinator)

        async let first: EnginePoolLoadResult = pool.load("a")
        await coordinator.waitUntilLoadStarted(count: 1)

        await XCTAssertThrowsEnginePoolError(try await pool.load("a")) { error in
            XCTAssertEqual(error, .modelLoading(id: "a"))
        }

        await coordinator.releaseOne()
        _ = try await first
        let loadedIDs = await state.loadedIDs()
        XCTAssertEqual(loadedIDs, ["a"])
    }

    func testMemoryGuardCeilingArithmetic() {
        let gib = MemoryGuard.gibibyte

        XCTAssertEqual(
            MemoryGuard.finalCeiling(
                recommendedWorkingSetBytes: 32 * gib,
                physicalMemoryBytes: 64 * gib,
                tier: .safe
            ),
            24 * gib
        )
        XCTAssertEqual(
            MemoryGuard.finalCeiling(
                recommendedWorkingSetBytes: 16 * gib,
                physicalMemoryBytes: 16 * gib,
                tier: .balanced
            ),
            12 * gib
        )
        XCTAssertEqual(
            MemoryGuard.finalCeiling(recommendedWorkingSetBytes: 16 * gib, tier: nil),
            0
        )
    }

    func testMemoryGuardEffectiveCeilingPrecedence() {
        let gib = MemoryGuard.gibibyte
        let recommendedWorkingSet = 32 * gib
        let physicalMemory = 64 * gib

        XCTAssertEqual(
            MemoryGuard.effectiveCeiling(
                overrideBytes: 800_000_000,
                recommendedWorkingSetBytes: recommendedWorkingSet,
                physicalMemoryBytes: physicalMemory,
                tier: .safe
            ),
            MemoryGuard.EffectiveCeiling(bytes: 800_000_000, source: "override")
        )
        XCTAssertEqual(
            MemoryGuard.effectiveCeiling(
                overrideBytes: 0,
                recommendedWorkingSetBytes: recommendedWorkingSet,
                physicalMemoryBytes: physicalMemory,
                tier: .balanced
            ),
            MemoryGuard.EffectiveCeiling(bytes: 26 * gib, source: "tier")
        )
        XCTAssertEqual(
            MemoryGuard.effectiveCeiling(
                overrideBytes: nil,
                recommendedWorkingSetBytes: recommendedWorkingSet,
                physicalMemoryBytes: physicalMemory,
                tier: nil
            ),
            MemoryGuard.EffectiveCeiling(bytes: 0, source: "off")
        )
    }

    func testMemoryAccountingInvariantAfterConcurrentSweepLeaseAndEvictingLoad() async throws {
        let state = FakeLoaderState()
        let coordinator = BlockingUnloadCoordinator()
        let pool = makeBlockingUnloadPool(
            ["a": 40, "b": 30, "c": 80],
            finalCeiling: 100,
            idleTimeout: 30,
            state: state,
            coordinator: coordinator
        )
        let start = Date(timeIntervalSince1970: 1_000)
        try await pool.load("a", now: start)
        try await pool.load("b", now: start.addingTimeInterval(1))

        async let sweepResult: [String] = pool.sweepIdleModels(now: start.addingTimeInterval(40))
        await coordinator.waitUntilUnloadStarted(count: 1)
        let lease = try await pool.acquire("b")
        await coordinator.releaseOne()
        let swept = await sweepResult
        XCTAssertEqual(swept, ["a"])
        await pool.release(lease)

        async let loadC: EnginePoolLoadResult = pool.load("c", now: start.addingTimeInterval(50))
        await coordinator.waitUntilUnloadStarted(count: 2)
        await coordinator.releaseOne()
        _ = try await loadC

        let status = await pool.status()
        XCTAssertEqual(loadedModelIDs(status), ["c"])
        XCTAssertEqual(status.currentModelMemory, 80)
        assertMemoryAccountingInvariant(status)
    }

    private func makePool(
        _ sizes: [String: Int64],
        finalCeiling: Int64 = 0,
        idleTimeout: TimeInterval? = nil,
        maxWaitingQueueDepth: Int = 0,
        state: FakeLoaderState = FakeLoaderState()
    ) -> EnginePool<FakeLoader> {
        var models: [String: DiscoveredModel] = [:]
        for (id, size) in sizes {
            models[id] = DiscoveredModel(
                id: id,
                modelURL: URL(fileURLWithPath: "/tmp/\(id)", isDirectory: true),
                estimatedSize: size
            )
        }
        return EnginePool(
            models: models,
            loader: FakeLoader(state: state),
            finalCeiling: finalCeiling,
            idleTimeout: idleTimeout,
            maxWaitingQueueDepth: maxWaitingQueueDepth
        )
    }

    private func makeBlockingUnloadPool(
        _ sizes: [String: Int64],
        finalCeiling: Int64 = 0,
        idleTimeout: TimeInterval? = nil,
        state: FakeLoaderState,
        coordinator: BlockingUnloadCoordinator
    ) -> EnginePool<BlockingUnloadFakeLoader> {
        var models: [String: DiscoveredModel] = [:]
        for (id, size) in sizes {
            models[id] = DiscoveredModel(
                id: id,
                modelURL: URL(fileURLWithPath: "/tmp/\(id)", isDirectory: true),
                estimatedSize: size
            )
        }
        return EnginePool(
            models: models,
            loader: BlockingUnloadFakeLoader(state: state, coordinator: coordinator),
            finalCeiling: finalCeiling,
            idleTimeout: idleTimeout
        )
    }

    private func makeBlockingPool(
        _ sizes: [String: Int64],
        finalCeiling: Int64 = 0,
        state: FakeLoaderState,
        coordinator: BlockingLoadCoordinator
    ) -> EnginePool<BlockingFakeLoader> {
        var models: [String: DiscoveredModel] = [:]
        for (id, size) in sizes {
            models[id] = DiscoveredModel(
                id: id,
                modelURL: URL(fileURLWithPath: "/tmp/\(id)", isDirectory: true),
                estimatedSize: size
            )
        }
        return EnginePool(
            models: models,
            loader: BlockingFakeLoader(state: state, coordinator: coordinator),
            finalCeiling: finalCeiling
        )
    }

    private func loadedModelIDs(_ status: EnginePoolStatus) -> [String] {
        status.models.filter(\.loaded).map(\.id).sorted()
    }

    private func assertMemoryAccountingInvariant(
        _ status: EnginePoolStatus,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let loadedMemory = status.models
            .filter(\.loaded)
            .reduce(Int64(0)) { total, model in
                total + (model.actualSize ?? model.estimatedSize)
            }
        XCTAssertEqual(status.currentModelMemory, loadedMemory, file: file, line: line)
    }
}

private func XCTAssertThrowsEnginePoolError(
    _ expression: @autoclosure () async throws -> some Sendable,
    file: StaticString = #filePath,
    line: UInt = #line,
    _ assertion: (EnginePoolError) -> Void
) async {
    do {
        _ = try await expression()
        XCTFail("expected EnginePoolError", file: file, line: line)
    } catch let error as EnginePoolError {
        assertion(error)
    } catch {
        XCTFail("expected EnginePoolError, got \(error)", file: file, line: line)
    }
}
