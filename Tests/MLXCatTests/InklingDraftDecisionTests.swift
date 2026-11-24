import Testing

@testable import MLXCatNative

struct InklingDraftDecisionTests {
    @Test func stopsAtFirstRejectionEvenWhenLaterRowsAgree() {
        let decision = InklingDraftDecision(
            proposals: [10, 21, 30], target: [10, 20, 30, 40], remaining: 5, endToken: 99)
        #expect(Array(decision.accepted) == [10])
        #expect(decision.rowsToKeep == 2)
        #expect(decision.next == 20)
        #expect(!decision.isFinished)
    }

    @Test func allAcceptedKeepsAnchorAndEveryProposedRow() {
        let decision = InklingDraftDecision(
            proposals: [10, 20], target: [10, 20, 30], remaining: 5, endToken: 99)
        #expect(Array(decision.accepted) == [10, 20])
        #expect(decision.rowsToKeep == 3)
        #expect(decision.next == 30)
        #expect(!decision.isFinished)
    }

    @Test func stopTokenAndOutputLimitBothDiscardTheSuffix() {
        let stopped = InklingDraftDecision(
            proposals: [99, 20], target: [99, 20, 30], remaining: 5, endToken: 99)
        #expect(stopped.isFinished)
        #expect(stopped.rowsToKeep == 2)
        let limited = InklingDraftDecision(
            proposals: [10, 20], target: [10, 20, 30], remaining: 1, endToken: 99)
        #expect(limited.isFinished)
        #expect(limited.rowsToKeep == 2)
    }
}
