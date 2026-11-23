/// The visit index owns request state. The physical layer index owns weights.
struct InklingLoop: Equatable {
    struct Visit: Equatable {
        let layer: Int
        let pass: Int?
    }

    enum ConfigurationError: Error {
        case invalidLoop
    }

    let visits: [Visit]
    private let coreStart: Int

    init(layerCount: Int, core: Range<Int>, passes: Int, residualBlockSize: Int) throws {
        guard residualBlockSize > 0, core.lowerBound > 0, !core.isEmpty,
            core.upperBound <= layerCount, passes > 0,
            core.lowerBound.isMultiple(of: residualBlockSize)
        else {
            throw ConfigurationError.invalidLoop
        }

        coreStart = core.lowerBound
        var visits = (0..<core.lowerBound).map { Visit(layer: $0, pass: nil) }
        for pass in 0..<passes {
            visits.append(contentsOf: core.map { Visit(layer: $0, pass: pass) })
        }
        visits.append(contentsOf: (core.upperBound..<layerCount).map { Visit(layer: $0, pass: nil) })
        self.visits = visits
    }

    struct ResidualStream<Value> {
        var completed: [Value] = []
        var partial: Value
    }

    /// Follows mlx-vlm language.py:740–778 at 466069f. A core restart sends
    /// only the preceding pass's completed core blocks and its last partial.
    func forward<Value, Bank>(
        embedding: Value,
        makeBank: ([Value]) -> Bank,
        visitLayer: (Int, Visit, Bank?, inout ResidualStream<Value>) -> Void
    ) -> ResidualStream<Value> {
        var stream = ResidualStream(partial: embedding)
        var prelude = stream
        var bank: Bank?
        for (slot, visit) in visits.enumerated() {
            if visit.layer == coreStart {
                if visit.pass == 0 {
                    prelude = stream
                } else {
                    let coreBlocks = stream.completed.dropFirst(prelude.completed.count + 1)
                    bank = makeBank(Array(coreBlocks) + [stream.partial])
                    stream = prelude
                }
            }
            visitLayer(slot, visit, visit.pass == nil ? nil : bank, &stream)
        }
        return stream
    }
}
