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

final class MemoryWatchdogTests: XCTestCase {
    // 100-byte ceiling => soft 80, hard 92 with the conservative defaults.
    private func config(ceiling: Int64 = 100) -> MemoryWatchdogConfiguration {
        MemoryWatchdogConfiguration(ceilingBytes: ceiling)
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
        // so evict runs and frees 10 -> 75 (< soft) => recovers to ok.
        let world = FakeMemoryWorld(usage: 95, trimFrees: 10, evictFrees: 10)
        let guardActor = watchdog(world: world)

        let level = await guardActor.poll()

        XCTAssertEqual(level, .ok)
        let events = await world.recordedEvents()
        XCTAssertEqual(events, ["trim(15)", "evict(5)"])
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
        XCTAssertEqual(events, ["trim(8)", "evict(8)"])
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
        XCTAssertEqual(events, ["trim(8)", "evict(8)"])
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
