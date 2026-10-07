import Foundation
@testable import MLXCat
import XCTest

/// A scriptable usage sampler + reclaimer. `usage` is mutable so a reclaim step can
/// lower it, and every reclaim call is recorded in order so tests can assert the
/// trim-before-evict ladder.
private actor FakeMemoryWorld: MemoryWatchdogReclaimer {
    private(set) var usage: Int64
    private(set) var events: [String] = []
    /// How much each reclaim step frees, and whether to actually lower usage by it.
    private let trimFrees: Int64
    private let evictFrees: Int64

    init(usage: Int64, trimFrees: Int64 = 0, evictFrees: Int64 = 0) {
        self.usage = usage
        self.trimFrees = trimFrees
        self.evictFrees = evictFrees
    }

    func sample() -> Int64 { usage }

    func setUsage(_ value: Int64) { usage = value }

    func trimReclaimableCaches(targetBytes: Int64) async -> Int64 {
        events.append("trim(\(targetBytes))")
        let freed = min(trimFrees, usage)
        usage -= freed
        return freed
    }

    func evictIdleModels(targetBytes: Int64) async -> Int64 {
        events.append("evict(\(targetBytes))")
        let freed = min(evictFrees, usage)
        usage -= freed
        return freed
    }

    func recordedEvents() -> [String] { events }
}

/// The reclaim ladder, run with the shipped default for
/// `MLXCAT_WATCHDOG_TRIM_AFTER_EVICT`. The two subclasses below run every test
/// here again with the lever pinned off and on, so flipping
/// ``MemoryWatchdogConfiguration/defaultTrimsAfterEviction`` needs no test edit.
class MemoryWatchdogTests: XCTestCase {
    /// Overridden by the subclasses to pin the lever.
    var trimsAfterEviction: Bool { MemoryWatchdogConfiguration.defaultTrimsAfterEviction }

    // 100-byte ceiling => soft 80, hard 92 with the conservative defaults.
    private func config(ceiling: Int64 = 100) -> MemoryWatchdogConfiguration {
        MemoryWatchdogConfiguration(ceilingBytes: ceiling, trimsAfterEviction: trimsAfterEviction)
    }

    /// The events an eviction step records: `evict(target)`, followed by a
    /// second `trim(target)` when the lever is on.
    private func evictStep(_ targetBytes: Int64) -> [String] {
        if trimsAfterEviction {
            return ["evict(\(targetBytes))", "trim(\(targetBytes))"]
        }
        return ["evict(\(targetBytes))"]
    }

    private func watchdog(world: FakeMemoryWorld, ceiling: Int64 = 100) -> MemoryWatchdog {
        MemoryWatchdog(
            configuration: config(ceiling: ceiling),
            sampler: { await world.sample() },
            reclaimer: world
        )
    }

    func testConfigurationWatermarks() {
        let cfg = MemoryWatchdogConfiguration(ceilingBytes: 100)
        XCTAssertEqual(cfg.softBytes, 80)
        XCTAssertEqual(cfg.hardBytes, 92)
    }

    func testConfigurationClampsInvertedFractions() {
        // hard < soft is corrected up to soft; both clamped into band.
        let cfg = MemoryWatchdogConfiguration(ceilingBytes: 100, softFraction: 0.9, hardFraction: 0.5)
        XCTAssertGreaterThanOrEqual(cfg.hardFraction, cfg.softFraction)
    }

    func testDisabledWatchdogIsNoOp() async throws {
        let world = FakeMemoryWorld(usage: 10_000)
        let guardActor = watchdog(world: world, ceiling: 0)
        let enabled = await guardActor.isEnabled
        XCTAssertFalse(enabled)

        let level = await guardActor.poll()
        XCTAssertEqual(level, .ok)
        // No reclaim attempted, admission never denied.
        try await guardActor.checkAdmission(additionalBytes: 1_000_000)
        let events = await world.recordedEvents()
        XCTAssertEqual(events, [])
    }

    func testPollBelowSoftDoesNothing() async throws {
        let world = FakeMemoryWorld(usage: 50)
        let guardActor = watchdog(world: world)

        let level = await guardActor.poll()

        XCTAssertEqual(level, .ok)
        let events = await world.recordedEvents()
        XCTAssertEqual(events, [])
        let blocked = await guardActor.admissionsBlocked
        XCTAssertFalse(blocked)
    }

    func testPollTrimsBeforeEvictingAndRecovers() async throws {
        // Start at 95 (over hard). Trim frees 10 -> 85 (still >= soft 80),
        // so evict runs and frees 10 -> 75 (< soft) => recovers to ok. With the
        // lever on, the second trim frees 10 more (65): still ok.
        let world = FakeMemoryWorld(usage: 95, trimFrees: 10, evictFrees: 10)
        let guardActor = watchdog(world: world)

        let level = await guardActor.poll()

        XCTAssertEqual(level, .ok)
        let events = await world.recordedEvents()
        XCTAssertEqual(events, ["trim(15)"] + evictStep(5))
        let blocked = await guardActor.admissionsBlocked
        XCTAssertFalse(blocked)
    }

    func testPollTrimAloneRecoversSkipsEvict() async throws {
        // Trim frees 20 -> 75 (< soft), so the evict step never runs.
        let world = FakeMemoryWorld(usage: 95, trimFrees: 20, evictFrees: 100)
        let guardActor = watchdog(world: world)

        let level = await guardActor.poll()

        XCTAssertEqual(level, .ok)
        let events = await world.recordedEvents()
        XCTAssertEqual(events, ["trim(15)"])
    }

    func testPollStaysSoftWhenReclaimInsufficient() async throws {
        // 88 is between soft(80) and hard(92). No reclaim frees anything.
        let world = FakeMemoryWorld(usage: 88, trimFrees: 0, evictFrees: 0)
        let guardActor = watchdog(world: world)

        let level = await guardActor.poll()

        XCTAssertEqual(level, .soft)
        let blocked = await guardActor.admissionsBlocked
        XCTAssertTrue(blocked)
        let events = await world.recordedEvents()
        XCTAssertEqual(events, ["trim(8)"] + evictStep(8))
    }

    func testPollStaysHardWhenReclaimInsufficient() async throws {
        let world = FakeMemoryWorld(usage: 99, trimFrees: 0, evictFrees: 0)
        let guardActor = watchdog(world: world)

        let level = await guardActor.poll()

        XCTAssertEqual(level, .hard)
        let blocked = await guardActor.admissionsBlocked
        XCTAssertTrue(blocked)
    }

    func testCheckAdmissionAllowsWhenUnderHard() async throws {
        let world = FakeMemoryWorld(usage: 50)
        let guardActor = watchdog(world: world)

        try await guardActor.checkAdmission(additionalBytes: 30) // 80 <= hard 92

        let events = await world.recordedEvents()
        XCTAssertEqual(events, [])
    }

    func testCheckAdmissionReclaimsThenAllows() async throws {
        // usage 90 + 10 = 100 > hard 92. Trim frees 0, evict frees 20 -> usage 70,
        // 70 + 10 = 80 <= 92 => admitted.
        let world = FakeMemoryWorld(usage: 90, trimFrees: 0, evictFrees: 20)
        let guardActor = watchdog(world: world)

        try await guardActor.checkAdmission(additionalBytes: 10)

        let events = await world.recordedEvents()
        XCTAssertEqual(events, ["trim(8)"] + evictStep(8))
    }

    func testCheckAdmissionDeniesWhenReclaimInsufficient() async throws {
        let world = FakeMemoryWorld(usage: 90, trimFrees: 0, evictFrees: 0)
        let guardActor = watchdog(world: world)

        do {
            try await guardActor.checkAdmission(additionalBytes: 10)
            XCTFail("expected admissionDenied")
        } catch let error as MemoryWatchdogError {
            guard case .admissionDenied(let required, let current, let ceiling) = error else {
                return XCTFail("wrong error \(error)")
            }
            XCTAssertEqual(required, 10)
            XCTAssertEqual(current, 90)
            XCTAssertEqual(ceiling, 92)
        }
        let events = await world.recordedEvents()
        XCTAssertEqual(events, ["trim(8)"] + evictStep(8))
    }

    func testRecoveryUnblocksAfterUsageDrops() async throws {
        let world = FakeMemoryWorld(usage: 99, trimFrees: 0, evictFrees: 0)
        let guardActor = watchdog(world: world)
        _ = await guardActor.poll()
        var blocked = await guardActor.admissionsBlocked
        XCTAssertTrue(blocked)

        await world.setUsage(40)
        let level = await guardActor.poll()

        XCTAssertEqual(level, .ok)
        blocked = await guardActor.admissionsBlocked
        XCTAssertFalse(blocked)
    }
}

final class MemoryWatchdogLeverOffTests: MemoryWatchdogTests {
    override var trimsAfterEviction: Bool { false }
}

final class MemoryWatchdogLeverOnTests: MemoryWatchdogTests {
    override var trimsAfterEviction: Bool { true }
}

/// MLX's accounting as the live sampler sees it: active + free-list cache.
/// Evicting a model moves its bytes from active to the cache (the weights are
/// released after the loader's own clear); only a trim returns cache bytes.
private actor FreeListMemoryWorld: MemoryWatchdogReclaimer {
    private var active: Int64
    private var cache: Int64
    private let evictableModelBytes: Int64
    private(set) var events: [String] = []

    init(active: Int64, cache: Int64, evictableModelBytes: Int64) {
        self.active = active
        self.cache = cache
        self.evictableModelBytes = evictableModelBytes
    }

    func sample() -> Int64 { active + cache }

    func trimReclaimableCaches(targetBytes: Int64) async -> Int64 {
        events.append("trim")
        let freed = cache
        cache = 0
        return freed
    }

    func evictIdleModels(targetBytes: Int64) async -> Int64 {
        events.append("evict")
        let moved = min(evictableModelBytes, active)
        active -= moved
        cache += moved
        return moved
    }

    func recordedEvents() -> [String] { events }
}

final class MemoryWatchdogTrimAfterEvictionTests: XCTestCase {
    // 100-byte ceiling => hard 92. 70 active (40 of it an idle model) + 10 cache;
    // a 30-byte load needs the idle model gone AND its bytes off the free list.
    private func world() -> FreeListMemoryWorld {
        FreeListMemoryWorld(active: 70, cache: 10, evictableModelBytes: 40)
    }

    private func watchdog(_ world: FreeListMemoryWorld, trimsAfterEviction: Bool) -> MemoryWatchdog {
        MemoryWatchdog(
            configuration: MemoryWatchdogConfiguration(
                ceilingBytes: 100, trimsAfterEviction: trimsAfterEviction),
            sampler: { await world.sample() },
            reclaimer: world
        )
    }

    /// The initializer and an unset (or unrecognised) variable both follow the
    /// one declared default; explicit values override it either way.
    func testDefaultComesFromOneConstant() {
        let shipped = MemoryWatchdogConfiguration.defaultTrimsAfterEviction
        XCTAssertEqual(MemoryWatchdogConfiguration(ceilingBytes: 100).trimsAfterEviction, shipped)
        XCTAssertEqual(MemoryWatchdogConfiguration.trimsAfterEvictionFromEnvironment([:]), shipped)
        XCTAssertEqual(
            MemoryWatchdogConfiguration.trimsAfterEvictionFromEnvironment(
                ["MLXCAT_WATCHDOG_TRIM_AFTER_EVICT": "maybe"]),
            shipped)
        for off in ["0", "false", "never", "FALSE"] {
            XCTAssertFalse(
                MemoryWatchdogConfiguration.trimsAfterEvictionFromEnvironment(
                    ["MLXCAT_WATCHDOG_TRIM_AFTER_EVICT": off]), off)
        }
        for on in ["1", "true", "always", "True"] {
            XCTAssertTrue(
                MemoryWatchdogConfiguration.trimsAfterEvictionFromEnvironment(
                    ["MLXCAT_WATCHDOG_TRIM_AFTER_EVICT": on]), on)
        }
    }

    /// Today's behaviour, pinned: the evicted bytes are still counted (as cache)
    /// when the ladder re-samples, so a load that would now fit is denied.
    func testDefaultLadderDeniesALoadThatFitsAfterEviction() async {
        let world = world()
        let guardActor = watchdog(world, trimsAfterEviction: false)
        do {
            try await guardActor.checkAdmission(additionalBytes: 30)
            XCTFail("expected the default ladder to deny: evicted bytes still sit in the cache")
        } catch let error as MemoryWatchdogError {
            guard case .admissionDenied(_, let current, _) = error else {
                return XCTFail("unexpected \(error)")
            }
            XCTAssertEqual(current, 70, "30 resident + 40 evicted bytes still on the free list")
        } catch {
            XCTFail("unexpected \(error)")
        }
        let events = await world.recordedEvents()
        XCTAssertEqual(events, ["trim", "evict"])
    }

    func testLeverAdmitsOnceTheEvictedBytesLeaveTheFreeList() async throws {
        let world = world()
        let guardActor = watchdog(world, trimsAfterEviction: true)
        try await guardActor.checkAdmission(additionalBytes: 30)
        let events = await world.recordedEvents()
        XCTAssertEqual(events, ["trim", "evict", "trim"])
        let usage = await world.sample()
        XCTAssertEqual(usage, 30)
    }

    func testLeverAlsoAppliesToThePollLadder() async {
        // 80 active (40 evictable) + 10 cache: over soft (80) even after the first trim.
        let world = FreeListMemoryWorld(active: 80, cache: 10, evictableModelBytes: 40)
        let off = await watchdog(world, trimsAfterEviction: false).poll()
        XCTAssertEqual(off, .soft, "default: 40 resident + 40 evicted-but-cached")

        let world2 = FreeListMemoryWorld(active: 80, cache: 10, evictableModelBytes: 40)
        let on = await watchdog(world2, trimsAfterEviction: true).poll()
        XCTAssertEqual(on, .ok)
        let events = await world2.recordedEvents()
        XCTAssertEqual(events, ["trim", "evict", "trim"])
    }
}
