import Foundation

public class PagedCacheManager: @unchecked Sendable {
    public let blockSize: Int
    public let maxStoredBlocks: Int

    private let lock = NSRecursiveLock()
    private var blocks: [KVCacheBlock]
    private var freeBlocks = FreeBlockQueue()
    private var hashIndex = BlockHashIndex()
    private var payloadsByHash: [Data: KVCacheBlockPayload] = [:]
    private var tick: UInt64 = 0

    public init(blockSize: Int = 256, initialCapacity: Int = 0, maxStoredBlocks: Int = 4096) {
        self.blockSize = blockSize
        self.maxStoredBlocks = max(1, maxStoredBlocks)
        self.blocks = [KVCacheBlock(blockID: 0)]
        if initialCapacity > 0 {
            grow(by: initialCapacity)
        }
    }

    public var allocatedBlocks: Int {
        withLock {
            blocks.filter(\.isAllocated).count
        }
    }

    public var totalRefCount: Int {
        withLock {
            blocks.reduce(0) { $0 + $1.refCount }
        }
    }

    public func contains(hash: Data) -> Bool {
        withLock {
            hashIndex.blockID(for: hash) != nil
        }
    }

    public var hotPayloadCount: Int {
        withLock {
            payloadsByHash.count
        }
    }

    public var hotPayloadHashes: [Data] {
        withLock {
            Array(payloadsByHash.keys)
        }
    }

    public func blockHash(for blockID: Int) -> Data? {
        withLock {
            guard blocks.indices.contains(blockID) else { return nil }
            return blocks[blockID].blockHash
        }
    }

    public func blockHashes(for table: BlockTable) -> [Data] {
        withLock {
            table.blockIDs.compactMap { blockID in
                guard blocks.indices.contains(blockID) else { return nil }
                return blocks[blockID].blockHash
            }
        }
    }

    public func payload(for hash: Data) -> KVCacheBlockPayload? {
        withLock {
            payloadsByHash[hash]
        }
    }

    public func payload(for blockID: Int) -> KVCacheBlockPayload? {
        withLock {
            guard blocks.indices.contains(blockID), let hash = blocks[blockID].blockHash else { return nil }
            return payloadsByHash[hash]
        }
    }

    public func setHotPayload(_ payload: KVCacheBlockPayload, for hash: Data) {
        withLock {
            payloadsByHash[hash] = payload
        }
    }

    public func removeHotPayload(for hash: Data) {
        _ = withLock {
            payloadsByHash.removeValue(forKey: hash)
        }
    }

    public func tokenCount(for hash: Data) -> Int? {
        withLock {
            guard let blockID = hashIndex.blockID(for: hash) else { return nil }
            return blocks[blockID].tokenCount
        }
    }

    @discardableResult
    public func storeBlock(
        hash: Data,
        tokenCount: Int,
        payload: KVCacheBlockPayload
    ) -> Int {
        withLock {
            if let blockID = hashIndex.blockID(for: hash) {
                payloadsByHash[hash] = payloadsByHash[hash] ?? payload
                touch(blockID: blockID)
                return blockID
            }

            let blockID = allocateBlockID()
            let block = blocks[blockID]
            block.refCount = 1
            block.blockHash = hash
            block.tokenCount = tokenCount
            block.lastAccessTick = nextTick()
            hashIndex.insert(blockID: blockID, hash: hash)
            payloadsByHash[hash] = payload
            evictStoredBlocksIfNeeded(protectedBlockID: blockID)
            return blockID
        }
    }

    @discardableResult
    public func registerBlockMetadata(
        hash: Data,
        tokenCount: Int,
        refCount: Int = 1
    ) -> Int {
        withLock {
            if let blockID = hashIndex.blockID(for: hash) {
                return blockID
            }

            let blockID = allocateBlockID()
            let block = blocks[blockID]
            block.refCount = refCount
            block.blockHash = hash
            block.tokenCount = tokenCount
            block.lastAccessTick = nextTick()
            hashIndex.insert(blockID: blockID, hash: hash)
            evictStoredBlocksIfNeeded(protectedBlockID: blockID)
            return blockID
        }
    }

    public func retain(hash: Data) -> Int? {
        withLock {
            guard let blockID = hashIndex.blockID(for: hash) else { return nil }
            retain(blockID: blockID)
            return blockID
        }
    }

    public func retain(blockID: Int) {
        withLock {
            guard blocks.indices.contains(blockID), blocks[blockID].isAllocated else { return }
            blocks[blockID].refCount += 1
            touch(blockID: blockID)
        }
    }

    public func release(_ table: BlockTable) {
        withLock {
            for blockID in table.blockIDs {
                release(blockID: blockID)
            }
            evictStoredBlocksIfNeeded(protectedBlockID: nil)
        }
    }

    public func forkBlockTable(_ table: BlockTable) -> BlockTable {
        withLock {
            for blockID in table.blockIDs {
                retain(blockID: blockID)
            }
            return table
        }
    }

    public func getBlocksForGeneration(_ table: inout BlockTable) {
        withLock {
            for (index, blockID) in table.blockIDs.enumerated() {
                guard blocks.indices.contains(blockID), blocks[blockID].refCount > 1,
                    let hash = blocks[blockID].blockHash,
                    let payload = payloadsByHash[hash]
                else {
                    continue
                }

                blocks[blockID].refCount -= 1
                let cloneID = allocateBlockID()
                let clone = blocks[cloneID]
                clone.refCount = 1
                clone.blockHash = hash
                clone.tokenCount = blocks[blockID].tokenCount
                clone.lastAccessTick = nextTick()
                hashIndex.insert(blockID: cloneID, hash: hash)
                payloadsByHash[hash] = payload
                table.blockIDs[index] = cloneID
            }
            evictStoredBlocksIfNeeded(protectedBlockID: nil)
        }
    }

    public func clearAll() {
        withLock {
            blocks = [KVCacheBlock(blockID: 0)]
            freeBlocks = FreeBlockQueue()
            hashIndex = BlockHashIndex()
            payloadsByHash.removeAll()
            tick = 0
        }
    }

    public func enforceCapacityLimit() {
        withLock {
            evictStoredBlocksIfNeeded(protectedBlockID: nil)
        }
    }

    private func release(blockID: Int) {
        guard blocks.indices.contains(blockID), blocks[blockID].isAllocated else { return }
        blocks[blockID].refCount -= 1
        guard blocks[blockID].refCount <= 0 else { return }

        if let hash = blocks[blockID].blockHash {
            hashIndex.remove(blockID: blockID, hash: hash)
            if hashIndex.blockIDs(for: hash).isEmpty {
                payloadsByHash.removeValue(forKey: hash)
            }
        }

        blocks[blockID].refCount = 0
        blocks[blockID].blockHash = nil
        blocks[blockID].tokenCount = 0
        blocks[blockID].lastAccessTick = 0
        freeBlocks.push(blockID)
    }

    private func evictStoredBlocksIfNeeded(protectedBlockID: Int?) {
        while blocks.filter(\.isAllocated).count > maxStoredBlocks {
            let candidate = blocks
                .filter { block in
                    guard let hash = block.blockHash else { return false }
                    return block.isAllocated
                        && block.blockID != protectedBlockID
                        && block.refCount <= 1
                        && canEvictBlock(hash: hash)
                }
                .min { lhs, rhs in
                    lhs.lastAccessTick < rhs.lastAccessTick
                }
            guard let candidate else { return }
            evict(blockID: candidate.blockID)
        }
    }

    private func evict(blockID: Int) {
        guard blocks.indices.contains(blockID), blocks[blockID].isAllocated,
            let hash = blocks[blockID].blockHash
        else {
            return
        }

        hashIndex.remove(blockID: blockID, hash: hash)
        if hashIndex.blockIDs(for: hash).isEmpty {
            payloadsByHash.removeValue(forKey: hash)
        }

        blocks[blockID].refCount = 0
        blocks[blockID].blockHash = nil
        blocks[blockID].tokenCount = 0
        blocks[blockID].lastAccessTick = 0
        freeBlocks.push(blockID)
        didEvictBlock(hash: hash)
    }

    func canEvictBlock(hash: Data) -> Bool {
        true
    }

    func didEvictBlock(hash: Data) {}

    private func touch(blockID: Int) {
        guard blocks.indices.contains(blockID) else { return }
        blocks[blockID].lastAccessTick = nextTick()
    }

    private func allocateBlockID() -> Int {
        if let blockID = freeBlocks.pop() {
            return blockID
        }

        let blockID = blocks.count
        blocks.append(KVCacheBlock(blockID: blockID))
        return blockID
    }

    private func grow(by count: Int) {
        guard count > 0 else { return }
        for _ in 0 ..< count {
            let blockID = blocks.count
            blocks.append(KVCacheBlock(blockID: blockID))
            freeBlocks.push(blockID)
        }
    }

    private func nextTick() -> UInt64 {
        tick += 1
        return tick
    }

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }
}
