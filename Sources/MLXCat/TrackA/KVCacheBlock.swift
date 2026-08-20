import Foundation

public final class KVCacheBlock: @unchecked Sendable {
    public let blockID: Int
    public var refCount: Int
    public var blockHash: Data?
    public var tokenCount: Int
    public var lastAccessTick: UInt64

    public init(
        blockID: Int,
        refCount: Int = 0,
        blockHash: Data? = nil,
        tokenCount: Int = 0,
        lastAccessTick: UInt64 = 0
    ) {
        self.blockID = blockID
        self.refCount = refCount
        self.blockHash = blockHash
        self.tokenCount = tokenCount
        self.lastAccessTick = lastAccessTick
    }

    public var isAllocated: Bool {
        blockID != 0 && blockHash != nil
    }
}

public struct BlockTable: Sendable, Equatable {
    public var blockIDs: [Int]

    public init(blockIDs: [Int] = []) {
        self.blockIDs = blockIDs
    }

    public var count: Int {
        blockIDs.count
    }

    public var isEmpty: Bool {
        blockIDs.isEmpty
    }
}
