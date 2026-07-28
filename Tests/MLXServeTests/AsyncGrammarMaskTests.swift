import Foundation
@testable import MLXServe
import XCTest

final class AsyncGrammarMaskTests: XCTestCase {
    func testPreparedMaskBecomesReadyOffThread() {
        let mask = AsyncGrammarMask<TestMaskSnapshot>()

        mask.prepareCurrentState(from: TestMaskSnapshot(tokenIDs: [1, 2, 3]))

        XCTAssertEqual(waitForReadyMask(mask), [1, 2, 3])
        XCTAssertEqual(mask.completedCount, 1)
    }

    func testStalePreparedMaskIsDiscardedAfterStateAdvance() {
        let mask = AsyncGrammarMask<TestMaskSnapshot>()

        mask.prepareCurrentState(from: TestMaskSnapshot(tokenIDs: [1], delay: 0.05))
        mask.prepareAdvancedState(from: TestMaskSnapshot(tokenIDs: [2]))

        XCTAssertEqual(waitForReadyMask(mask), [2])
        XCTAssertTrue(waitForCompletedCount(mask, atLeast: 2))
        XCTAssertEqual(mask.readyTokenIDs, [2])
    }

    func testInFlightComputeIsCancelledByStateAdvance() {
        let mask = AsyncGrammarMask<BlockingMaskSnapshot>()
        let slow = BlockingMaskSnapshot(tokenIDs: [1])

        let start = Date()
        mask.prepareCurrentState(from: slow)
        XCTAssertEqual(slow.started.wait(timeout: .now() + 1), .success)
        mask.prepareAdvancedState(from: BlockingMaskSnapshot(tokenIDs: [2]))

        // The second snapshot also "computes" for up to 2 s unless cancelled,
        // but nothing cancels it — it must win. The first must abort well
        // before its natural 2 s deadline for the total to stay under ~2.5 s.
        XCTAssertEqual(waitForReadyMask(mask, timeout: 3), [2])
        XCTAssertLessThan(Date().timeIntervalSince(start), 2.5,
                          "stale in-flight compute ran to completion instead of aborting")
    }

    func testRapidAdvancesCoalesceToLatestSnapshot() {
        let mask = AsyncGrammarMask<TestMaskSnapshot>()

        mask.prepareCurrentState(from: TestMaskSnapshot(tokenIDs: [0], delay: 0.1))
        for i in 1 ... 40 {
            mask.prepareAdvancedState(from: TestMaskSnapshot(tokenIDs: [i]))
        }

        XCTAssertEqual(waitForReadyMask(mask), [40])
        // First (stale) compute + the single coalesced latest — not 41 jobs.
        XCTAssertLessThanOrEqual(mask.completedCount, 3)
    }

    func testInvalidateDropsPendingAndCancelsInFlight() {
        let mask = AsyncGrammarMask<BlockingMaskSnapshot>()
        let slow = BlockingMaskSnapshot(tokenIDs: [1])

        mask.prepareCurrentState(from: slow)
        XCTAssertEqual(slow.started.wait(timeout: .now() + 1), .success)
        mask.prepareAdvancedState(from: BlockingMaskSnapshot(tokenIDs: [2]))
        mask.invalidate()

        XCTAssertTrue(waitForCompletedCount(mask, atLeast: 1, timeout: 1),
                      "cancelled in-flight compute did not finish promptly after invalidate()")
        Thread.sleep(forTimeInterval: 0.05)
        XCTAssertNil(mask.readyTokenIDs)
        XCTAssertLessThanOrEqual(mask.completedCount, 1, "invalidate() must drop the pending compute")
    }

    private func waitForReadyMask<Snapshot: GrammarMaskSnapshot>(
        _ mask: AsyncGrammarMask<Snapshot>,
        timeout: TimeInterval = 1
    ) -> [Int]? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let tokenIDs = mask.readyTokenIDs {
                return tokenIDs
            }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return mask.readyTokenIDs
    }

    private func waitForCompletedCount<Snapshot: GrammarMaskSnapshot>(
        _ mask: AsyncGrammarMask<Snapshot>,
        atLeast expectedCount: Int,
        timeout: TimeInterval = 1
    ) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if mask.completedCount >= expectedCount {
                return true
            }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return mask.completedCount >= expectedCount
    }
}

private struct TestMaskSnapshot: GrammarMaskSnapshot {
    let tokenIDs: [Int]
    var delay: TimeInterval = 0

    func allowedTokenIDs(isCancelled: @Sendable () -> Bool) -> [Int]? {
        if delay > 0 {
            Thread.sleep(forTimeInterval: delay)
        }
        return tokenIDs
    }
}

/// Computes for up to 2 s unless the mask cancels it — long enough that a test
/// finishing fast proves cancellation did the work, not luck.
private final class BlockingMaskSnapshot: GrammarMaskSnapshot, @unchecked Sendable {
    let tokenIDs: [Int]
    let started = DispatchSemaphore(value: 0)

    init(tokenIDs: [Int]) {
        self.tokenIDs = tokenIDs
    }

    func allowedTokenIDs(isCancelled: @Sendable () -> Bool) -> [Int]? {
        started.signal()
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline {
            if isCancelled() {
                return nil
            }
            Thread.sleep(forTimeInterval: 0.002)
        }
        return tokenIDs
    }
}
