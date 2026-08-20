import Foundation

public struct BlockHashIndex: Sendable {
    private var blocksByHash: [Data: Set<Int>] = [:]

    public init() {}

    public func blockID(for hash: Data) -> Int? {
        blocksByHash[hash]?.min()
    }

    public func blockIDs(for hash: Data) -> Set<Int> {
        blocksByHash[hash, default: []]
    }

    public mutating func insert(blockID: Int, hash: Data) {
        blocksByHash[hash, default: []].insert(blockID)
    }

    public mutating func remove(blockID: Int, hash: Data) {
        blocksByHash[hash]?.remove(blockID)
        if blocksByHash[hash]?.isEmpty == true {
            blocksByHash.removeValue(forKey: hash)
        }
    }
}
