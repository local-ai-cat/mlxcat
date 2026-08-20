import Foundation
import MLX
import MLXLMCommon

public struct PrefixCacheHit: Sendable, Equatable {
    public let matchedTokenCount: Int
    public let table: BlockTable
    public let blockHashes: [Data]

    public var blockCount: Int {
        table.count
    }
}

public final class BlockAwarePrefixCache: @unchecked Sendable {
    public let modelName: String
    public let blockSize: Int
    public let manager: PagedCacheManager
    private let lock = NSRecursiveLock()

    public init(
        modelName: String,
        blockSize: Int = 256,
        manager: PagedCacheManager? = nil
    ) {
        self.modelName = modelName
        self.blockSize = blockSize
        self.manager = manager ?? PagedCacheManager(blockSize: blockSize)
    }

    public func fetchCache(tokens: [Int]) -> PrefixCacheHit? {
        withLock {
            let candidateHashes = BlockHashing.chainHashes(
                modelName: modelName,
                tokens: tokens,
                blockSize: blockSize
            )
            guard !candidateHashes.isEmpty else { return nil }

            var retainedBlockIDs: [Int] = []
            var retainedHashes: [Data] = []
            for hash in candidateHashes {
                guard let blockID = manager.retain(hash: hash) else { break }
                retainedBlockIDs.append(blockID)
                retainedHashes.append(hash)
            }

            guard !retainedBlockIDs.isEmpty else { return nil }
            return PrefixCacheHit(
                matchedTokenCount: retainedBlockIDs.count * blockSize,
                table: BlockTable(blockIDs: retainedBlockIDs),
                blockHashes: retainedHashes
            )
        }
    }

    public func reconstructCache(from hit: PrefixCacheHit) throws -> [KVCacheSimple] {
        try withLock {
            let payloads = try hit.table.blockIDs.map { blockID in
                guard let payload = manager.payload(for: blockID) else {
                    throw PrefixCacheError.missingBlockPayload(
                        blockID: blockID,
                        hash: manager.blockHash(for: blockID)
                    )
                }
                return payload
            }
            guard let firstPayload = payloads.first else { return [] }

            return try (0 ..< firstPayload.layers.count).map { layerIndex in
                let firstLayer = firstPayload.layers[layerIndex]
                for payload in payloads where payload.layers.indices.contains(layerIndex) == false {
                    throw PrefixCacheError.inconsistentLayerCount
                }
                let sequenceAxis = try Self.sequenceAxis(
                    cacheType: firstLayer.className,
                    state: [firstLayer.keys, firstLayer.values]
                )
                let keys = concatenated(payloads.map { $0.layers[layerIndex].keys }, axis: sequenceAxis)
                let values = concatenated(payloads.map { $0.layers[layerIndex].values }, axis: sequenceAxis)
                let cache = KVCacheSimple()
                cache.state = [keys, values]
                return cache
            }
        }
    }

    @discardableResult
    public func storeCache(tokens: [Int], cache: [any KVCache]) throws -> BlockTable {
        try withLock {
            let fullBlockCount = tokens.count / blockSize
            guard fullBlockCount > 0 else { return BlockTable() }
            try preflightCacheForStorage(cache)

            let hashes = BlockHashing.chainHashes(
                modelName: modelName,
                tokens: tokens,
                blockSize: blockSize
            )
            precondition(hashes.count == fullBlockCount)

            var blockIDs: [Int] = []
            for blockIndex in 0 ..< fullBlockCount {
                let hash = hashes[blockIndex]
                if manager.contains(hash: hash), let blockID = manager.retain(hash: hash) {
                    manager.release(BlockTable(blockIDs: [blockID]))
                    blockIDs.append(blockID)
                    continue
                }

                let payload = try payloadForBlock(
                    cache: cache,
                    tokenStart: blockIndex * blockSize,
                    tokenEnd: (blockIndex + 1) * blockSize
                )
                let blockID = manager.storeBlock(
                    hash: hash,
                    tokenCount: blockSize,
                    payload: payload
                )
                blockIDs.append(blockID)
            }

            return BlockTable(blockIDs: blockIDs)
        }
    }

    public func release(_ hit: PrefixCacheHit) {
        withLock {
            manager.release(hit.table)
        }
    }

    public func release(_ table: BlockTable) {
        withLock {
            manager.release(table)
        }
    }

    private func payloadForBlock(
        cache: [any KVCache],
        tokenStart: Int,
        tokenEnd: Int
    ) throws -> KVCacheBlockPayload {
        let layers = try cache.enumerated().map { _, layerCache in
            let state = layerCache.state
            let sequenceAxis = try Self.sequenceAxis(
                cacheType: String(describing: type(of: layerCache)),
                state: state
            )
            let keys = state[0][Self.indices(sequenceAxis: sequenceAxis, range: tokenStart ..< tokenEnd, rank: state[0].shape.count)]
            let values = state[1][Self.indices(sequenceAxis: sequenceAxis, range: tokenStart ..< tokenEnd, rank: state[1].shape.count)]
            return CacheLayerBlockPayload(
                keys: keys,
                values: values,
                metaState: [
                    String(blockSize),
                    CacheTypeHandlers.encodeBool(false),
                ]
            )
        }
        return KVCacheBlockPayload(layers: layers)
    }

    private func preflightCacheForStorage(_ cache: [any KVCache]) throws {
        guard !cache.isEmpty else {
            throw PrefixCacheError.unsupportedCacheLayout(cacheType: "empty", stateShapes: [])
        }
        for layerCache in cache {
            _ = try Self.sequenceAxis(
                cacheType: String(describing: type(of: layerCache)),
                state: layerCache.state
            )
        }
    }

    private static func sequenceAxis(cacheType: String, state: [MLXArray]) throws -> Int {
        guard state.count == 2 else {
            throw PrefixCacheError.unsupportedCacheLayout(
                cacheType: cacheType,
                stateShapes: state.map(\.shape)
            )
        }
        let ranks = state.map { $0.shape.count }
        guard ranks[0] == ranks[1], ranks[0] >= 3 else {
            throw PrefixCacheError.unsupportedCacheLayout(
                cacheType: cacheType,
                stateShapes: state.map(\.shape)
            )
        }
        let sequenceAxis = ranks[0] - 2
        guard state[0].dim(sequenceAxis) == state[1].dim(sequenceAxis) else {
            throw PrefixCacheError.unsupportedCacheLayout(
                cacheType: cacheType,
                stateShapes: state.map(\.shape)
            )
        }
        return sequenceAxis
    }

    private static func indices(sequenceAxis: Int, range: Range<Int>, rank: Int) -> [any MLXArrayIndex] {
        var result: [any MLXArrayIndex] = Array(repeating: 0..., count: rank)
        result[sequenceAxis] = range
        return result
    }

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }
}

public enum PrefixCacheError: Error, Equatable, CustomStringConvertible {
    case unsupportedCacheLayout(cacheType: String, stateShapes: [[Int]])
    case missingBlockPayload(blockID: Int, hash: Data?)
    case inconsistentLayerCount

    public var description: String {
        switch self {
        case .unsupportedCacheLayout(let cacheType, let stateShapes):
            return "Prefix cache supports sequence KV state only; \(cacheType) has state shapes \(stateShapes)."
        case .missingBlockPayload(let blockID, let hash):
            let hashDescription = hash.map(BlockHashing.hex) ?? "nil"
            return "Prefix cache block \(blockID) with hash \(hashDescription) has metadata but no loaded payload."
        case .inconsistentLayerCount:
            return "Prefix cache block payloads have inconsistent layer counts."
        }
    }
}
