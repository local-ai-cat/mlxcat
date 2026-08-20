final class OutputCollector {
    private var responsesByUID: [String: [Response]] = [:]
    private var finishReasonsByUID: [String: FinishReason] = [:]

    public init() {}

    public func record(_ response: Response) {
        responsesByUID[response.uid, default: []].append(response)
        if let finishReason = response.finishReason {
            finishReasonsByUID[response.uid] = finishReason
        }
    }

    public func record(_ responses: [Response]) {
        for response in responses {
            record(response)
        }
    }

    public func tokens(for uid: String) -> [Int] {
        responsesByUID[uid, default: []]
            .filter { response in
                if case .cancelled? = response.finishReason { return false }
                if case .failed? = response.finishReason { return false }
                return true
            }
            .map(\.token)
    }

    public func responses(for uid: String) -> [Response] {
        responsesByUID[uid, default: []]
    }

    public func finishReason(for uid: String) -> FinishReason? {
        finishReasonsByUID[uid]
    }

    public func consumeTokens(for uid: String) -> [Int] {
        let result = tokens(for: uid)
        remove(uid: uid)
        return result
    }

    public func remove(uid: String) {
        responsesByUID.removeValue(forKey: uid)
        finishReasonsByUID.removeValue(forKey: uid)
    }

    public var allTokens: [String: [Int]] {
        responsesByUID.mapValues { responses in
            responses
                .filter { response in
                    if case .cancelled? = response.finishReason { return false }
                    if case .failed? = response.finishReason { return false }
                    return true
                }
                .map(\.token)
        }
    }
}
