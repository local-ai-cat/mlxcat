struct InklingDraftDecision {
    /// The target rows predict the tokens following the anchor and each proposal.
    let accepted: ArraySlice<Int>
    let next: Int
    let isFinished: Bool

    var rowsToKeep: Int { 1 + accepted.count }

    init(proposals: [Int], target: [Int], remaining: Int, endToken: Int) {
        precondition(target.count == proposals.count + 1 && remaining > 0)
        var count = 0
        var ended = false
        for (proposal, exactToken) in zip(proposals.prefix(remaining), target) {
            guard proposal == exactToken else { break }
            count += 1
            if proposal == endToken {
                ended = true
                break
            }
        }
        accepted = proposals.prefix(count)
        next = target[count]
        isFinished = ended || count == remaining
    }
}
