import Testing

@testable import MLXCatNative

struct InklingLoopTests {
    @Test func miniVisitsEveryCoreLayerThreeTimesWithIndependentSlots() throws {
        let loop = try InklingLoop(layerCount: 24, core: 6..<18, passes: 3, residualBlockSize: 6)
        #expect(loop.visits.count == 48)
        #expect(loop.visits.enumerated().filter { $0.element.layer == 6 }.map(\.offset) == [6, 18, 30])
        #expect(loop.visits.enumerated().filter { $0.element.layer == 6 }.map(\.element.pass) == [0, 1, 2])
        #expect(loop.visits.suffix(6).map(\.layer) == Array(18..<24))
    }

    @Test func eachRestartUsesOnlyThePreviousCorePassAndRestoresThePrelude() throws {
        let loop = try InklingLoop(layerCount: 4, core: 1..<3, passes: 3, residualBlockSize: 1)
        var banks: [[Int]] = []
        let output = loop.forward(embedding: -1) { blocks in
            banks.append(blocks)
            return blocks
        } visitLayer: { slot, visit, bank, stream in
            if visit.layer == 1 {
                #expect(stream.partial == 0)
                #expect(stream.completed == [-1])
            }
            if visit.pass == nil || visit.pass == 0 {
                #expect(bank == nil)
            }
            stream.completed.append(stream.partial)
            stream.partial = slot
        }
        #expect(banks == [[1, 2], [3, 4]])
        #expect(output.completed == [-1, 0, 5, 6])
        #expect(output.partial == 7)
    }

    @Test func rejectsCoreStartsWithoutAResidualBoundary() {
        #expect(throws: InklingLoop.ConfigurationError.self) {
            try InklingLoop(layerCount: 24, core: 5..<18, passes: 3, residualBlockSize: 6)
        }
    }
}
