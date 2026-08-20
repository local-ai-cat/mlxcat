import MLX
import MLXLMCommon

final class BatchLayerCache {
    var kvCache: any KVCache

    private enum Layout: Equatable {
        case single
        case sequence
        case batchState
    }

    private var layout: Layout
    private var rowMetaStates: [[String]]

    private init(kvCache: any KVCache, layout: Layout, rowMetaStates: [[String]]) {
        self.kvCache = kvCache
        self.layout = layout
        self.rowMetaStates = rowMetaStates
    }

    var batchSize: Int {
        if layout == .single {
            return rowMetaStates.isEmpty ? 0 : 1
        }
        return kvCache.state.first?.dim(0) ?? 0
    }

    static func adoptSingle(_ cache: any KVCache) throws -> BatchLayerCache {
        guard !cache.state.isEmpty else {
            throw BatchKVCacheError.emptyState(cacheType: String(describing: type(of: cache)))
        }
        return BatchLayerCache(
            kvCache: cache,
            layout: .single,
            rowMetaStates: [cache.metaState]
        )
    }

    static func merge(_ caches: [any KVCache]) throws -> BatchLayerCache {
        guard let firstCache = caches.first else {
            throw BatchKVCacheError.emptyMerge
        }
        let firstState = firstCache.state
        guard !firstState.isEmpty else {
            throw BatchKVCacheError.emptyState(cacheType: String(describing: type(of: firstCache)))
        }

        if isSequenceState(firstState) {
            let batchCache = try BatchKVCache.merge(caches)
            return BatchLayerCache(
                kvCache: batchCache,
                layout: .sequence,
                rowMetaStates: caches.map(\.metaState)
            )
        }

        return try mergeBatchState(caches)
    }

    func insert(_ cache: any KVCache) throws {
        try extend(Self.merge([cache]))
    }

    func extend(_ other: BatchLayerCache) throws {
        guard layout == other.layout else {
            if layout == .single {
                try replaceWithMergedRows(from: [self, other])
                return
            }
            if other.layout == .single {
                try extend(Self.merge(other.rowCaches()))
                return
            }
            throw BatchKVCacheError.incompatibleLayout(
                expected: String(describing: layout),
                actual: String(describing: other.layout)
            )
        }

        switch layout {
        case .single:
            try replaceWithMergedRows(from: [self, other])
        case .sequence:
            guard let cache = kvCache as? BatchKVCache,
                let otherCache = other.kvCache as? BatchKVCache
            else {
                throw BatchKVCacheError.incompatibleLayout(expected: "BatchKVCache", actual: String(describing: type(of: other.kvCache)))
            }
            try cache.extend(otherCache)
        case .batchState:
            if let arraysCache = kvCache as? ArraysCache,
                let otherArraysCache = other.kvCache as? ArraysCache
            {
                arraysCache.extend(other: otherArraysCache)
            } else {
                let currentState = kvCache.state
                let otherState = other.kvCache.state
                guard currentState.count == otherState.count else {
                    throw BatchKVCacheError.inconsistentStateCount(
                        expected: currentState.count,
                        actual: otherState.count,
                        cacheType: String(describing: type(of: other.kvCache))
                    )
                }
                kvCache.state = zip(currentState, otherState).map { concatenated([$0, $1], axis: 0) }
            }
        }

        rowMetaStates.append(contentsOf: other.rowMetaStates)
        refreshBatchedMetaState()
    }

    func filter(keeping rows: [Int]) {
        guard !rows.isEmpty else {
            kvCache.state = []
            rowMetaStates.removeAll()
            return
        }

        switch layout {
        case .single:
            guard rows == [0] else {
                kvCache.state = []
                rowMetaStates.removeAll()
                return
            }
        case .sequence:
            (kvCache as? BatchKVCache)?.filter(keeping: rows)
        case .batchState:
            let rowIndices = MLXArray(rows.map(Int32.init))
            if let arraysCache = kvCache as? ArraysCache {
                arraysCache.filter(batchIndices: rowIndices)
            } else {
                kvCache.state = kvCache.state.map { $0.take(rowIndices, axis: 0) }
            }
        }
        rowMetaStates = rows.map { row in
            row < rowMetaStates.count ? rowMetaStates[row] : []
        }
        refreshBatchedMetaState()
    }

    func extract(_ row: Int) -> any KVCache {
        if layout == .single {
            guard row == 0, !rowMetaStates.isEmpty else { return KVCacheSimple() }
            return kvCache.copy()
        }

        if layout == .sequence, let cache = kvCache as? BatchKVCache {
            return cache.extract(row)
        }

        let cache = KVCacheSimple()
        guard row >= 0, row < batchSize else { return cache }
        cache.state = kvCache.state.map { Self.sliceRow($0, row: row) }
        return cache
    }

    func copyLayer() -> BatchLayerCache {
        BatchLayerCache(
            kvCache: kvCache.copy(),
            layout: layout,
            rowMetaStates: rowMetaStates
        )
    }

    private func rowCaches() -> [any KVCache] {
        guard batchSize > 0 else { return [] }
        if layout == .single {
            return [kvCache]
        }
        return (0 ..< batchSize).map { extract($0) }
    }

    private func replaceWithMergedRows(from layers: [BatchLayerCache]) throws {
        let merged = try Self.merge(layers.flatMap { $0.rowCaches() })
        kvCache = merged.kvCache
        layout = merged.layout
        rowMetaStates = merged.rowMetaStates
    }

    private static func mergeBatchState(_ caches: [any KVCache]) throws -> BatchLayerCache {
        guard let firstCache = caches.first else {
            throw BatchKVCacheError.emptyMerge
        }
        let states = caches.map(\.state)
        let cacheTypes = caches.map { String(describing: type(of: $0)) }
        let firstState = states[0]

        for (state, cacheType) in zip(states, cacheTypes) where state.count != firstState.count {
            throw BatchKVCacheError.inconsistentStateCount(
                expected: firstState.count,
                actual: state.count,
                cacheType: cacheType
            )
        }

        var batchState: [MLXArray] = []
        for stateIndex in firstState.indices {
            let firstShape = firstState[stateIndex].shape
            for row in states.indices {
                try validateBatchStateShape(
                    cacheType: cacheTypes[row],
                    stateIndex: stateIndex,
                    expected: firstShape,
                    actual: states[row][stateIndex].shape
                )
            }
            batchState.append(concatenated(states.map { $0[stateIndex] }, axis: 0))
        }

        if let firstArraysCache = firstCache as? ArraysCache {
            let cache = firstArraysCache.copy()
            guard let mergedCache = cache as? ArraysCache else {
                throw BatchKVCacheError.incompatibleLayout(
                    expected: "ArraysCache",
                    actual: String(describing: type(of: cache))
                )
            }
            for cache in caches.dropFirst() {
                guard let arraysCache = cache as? ArraysCache else {
                    throw BatchKVCacheError.incompatibleLayout(
                        expected: "ArraysCache",
                        actual: String(describing: type(of: cache))
                    )
                }
                mergedCache.extend(other: arraysCache)
            }
            return BatchLayerCache(
                kvCache: mergedCache,
                layout: .batchState,
                rowMetaStates: caches.map(\.metaState)
            )
        }

        var cache = firstCache.copy()
        cache.state = batchState
        let rowMetaStates = caches.map(\.metaState)
        cache.metaState = mergedMetaState(from: rowMetaStates)
        return BatchLayerCache(
            kvCache: cache,
            layout: .batchState,
            rowMetaStates: rowMetaStates
        )
    }

    private func refreshBatchedMetaState() {
        guard layout == .batchState else { return }
        guard !(kvCache is ArraysCache) else { return }
        kvCache.metaState = Self.mergedMetaState(from: rowMetaStates)
    }

    private static func mergedMetaState(from rows: [[String]]) -> [String] {
        guard rows.contains(where: { !$0.isEmpty }) else { return [] }
        let normalizedRows = rows.map { row in row.isEmpty ? ["0", ""] : row }
        guard normalizedRows.allSatisfy({ $0.count >= 2 }),
            normalizedRows.allSatisfy({ Int($0[0]) != nil })
        else {
            return normalizedRows.first ?? []
        }

        var slotBase = 0
        var presentSlots: [String] = []
        for row in normalizedRows {
            let slotCount = Int(row[0]) ?? 0
            let slots = row[1].split(separator: ",").compactMap { Int($0) }
            presentSlots.append(contentsOf: slots.map { String($0 + slotBase) })
            slotBase += slotCount
        }

        var merged = normalizedRows[0]
        merged[0] = String(slotBase)
        merged[1] = presentSlots.joined(separator: ",")
        return merged
    }

    private static func isSequenceState(_ state: [MLXArray]) -> Bool {
        guard state.count == 2 else { return false }
        let ranks = state.map { $0.shape.count }
        return ranks[0] == ranks[1] && ranks[0] >= 3
    }

    private static func validateBatchStateShape(
        cacheType: String,
        stateIndex: Int,
        expected: [Int],
        actual: [Int]
    ) throws {
        guard expected.count == actual.count else {
            throw BatchKVCacheError.incompatibleStateShape(
                cacheType: cacheType,
                stateIndex: stateIndex,
                expected: expected,
                actual: actual
            )
        }
        for axis in expected.indices where axis != 0 && expected[axis] != actual[axis] {
            throw BatchKVCacheError.incompatibleStateShape(
                cacheType: cacheType,
                stateIndex: stateIndex,
                expected: expected,
                actual: actual
            )
        }
    }

    private static func sliceRow(_ array: MLXArray, row: Int) -> MLXArray {
        var indices: [any MLXArrayIndex] = Array(repeating: 0..., count: array.shape.count)
        indices[0] = row ..< row + 1
        return array[indices]
    }
}
