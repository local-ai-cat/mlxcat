public struct FreeBlockQueue: Sendable {
    private var storage: [Int]
    private var head: Int

    public init(_ blockIDs: [Int] = []) {
        self.storage = blockIDs
        self.head = 0
    }

    public var isEmpty: Bool {
        head >= storage.count
    }

    public var count: Int {
        storage.count - head
    }

    public mutating func push(_ blockID: Int) {
        storage.append(blockID)
    }

    public mutating func pop() -> Int? {
        guard head < storage.count else { return nil }
        let blockID = storage[head]
        head += 1

        if head > 64, head * 2 > storage.count {
            storage.removeFirst(head)
            head = 0
        }
        return blockID
    }
}
