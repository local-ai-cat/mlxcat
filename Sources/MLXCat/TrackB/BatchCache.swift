import Foundation
import MLX
import MLXLMCommon

public enum BatchKVCacheError: Error, Equatable, CustomStringConvertible {
    case emptyMerge
    case emptyState(cacheType: String)
    case inconsistentStateCount(expected: Int, actual: Int, cacheType: String)
    case unsupportedSequenceState(cacheType: String, stateShapes: [[Int]])
    case incompatibleStateShape(cacheType: String, stateIndex: Int, expected: [Int], actual: [Int])
    case incompatibleLayout(expected: String, actual: String)
    case unsupportedStateMutation(layout: String, stateCount: Int)
    case invalidMetadata([String])
    case unsupportedRightPadding(layout: String)
    case rightPaddingWidthMismatch(expected: Int, actual: Int)
    case rightPaddingAlreadyPrepared

    public var description: String {
        switch self {
        case .emptyMerge:
            return "BatchKVCache.merge requires at least one cache."
        case .emptyState(let cacheType):
            return "BatchKVCache cannot merge empty cache state for \(cacheType)."
        case .inconsistentStateCount(let expected, let actual, let cacheType):
            return "BatchKVCache cannot merge \(cacheType): expected \(expected) state arrays, found \(actual)."
        case .unsupportedSequenceState(let cacheType, let stateShapes):
            return "BatchKVCache cannot treat \(cacheType) as sequence KV cache; state shapes are \(stateShapes)."
        case .incompatibleStateShape(let cacheType, let stateIndex, let expected, let actual):
            return "BatchKVCache cannot merge \(cacheType) state \(stateIndex): expected shape compatible with \(expected), found \(actual)."
        case .incompatibleLayout(let expected, let actual):
            return "BatchKVCache cannot combine \(actual) cache layout with existing \(expected) layout."
        case .unsupportedStateMutation(let layout, let stateCount):
            return "BatchKVCache cannot set \(stateCount) state arrays on \(layout) layout."
        case .invalidMetadata(let metadata):
            return "BatchKVCache metadata is invalid: \(metadata)."
        case .unsupportedRightPadding(let layout):
            return "BatchKVCache cannot right-pad a \(layout) layout; ragged prefill is sequence-layout only."
        case .rightPaddingWidthMismatch(let expected, let actual):
            return "BatchKVCache right padding must name every row: expected \(expected) values, found \(actual)."
        case .rightPaddingAlreadyPrepared:
            return "BatchKVCache already holds unfinalized right padding; finalize() before preparing another ragged write."
        }
    }
}

public final class BatchKVCache: KVCache, BatchPositionedKVCache {
    public private(set) var leftPadding: MLXArray {
        didSet { paddingGeneration &+= 1 }
    }
    public private(set) var idx: Int
    public var offset: Int { idx }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool {
        if case .sequence = layout { return true }
        return false
    }

    private enum Layout: Equatable {
        case sequence(axis: Int)
        case batchState

        var description: String {
            switch self {
            case .sequence(let axis):
                return "sequence(axis: \(axis))"
            case .batchState:
                return "batchState"
            }
        }
    }

    private struct RotatingCacheMetadata {
        let keep: Int
        let offset: Int
        let idx: Int
    }

    private var buffers: [MLXArray] = []
    private var rowMetaStates: [[String]] = []
    private var positionOffsets: MLXArray
    private var layout: Layout
    private let step: Int

    public var batchSize: Int {
        leftPadding.dim(0)
    }

    public init(leftPadding: [Int], idx: Int = 0, step: Int = 256) {
        self.leftPadding = MLXArray(leftPadding.map(Int32.init))
        self.positionOffsets = MLXArray(leftPadding.map { Int32(idx - $0) })
        self.idx = idx
        self.step = step
        self.layout = .sequence(axis: 2)
    }

    public var batchOffset: MLXArray {
        positionOffsets
    }

    // MARK: Single-row scalar RoPE offset

    /// Bumped by `leftPadding.didSet`, so every one of its write sites —
    /// current and future — invalidates the cached scalar below without any
    /// per-site bookkeeping. A counter bump is free; reading the offset value
    /// costs a host sync, which is why it happens lazily in `ropeOffset` and
    /// at most once per mutation epoch. `leftPadding` is a valid epoch marker
    /// for `positionOffsets` too: every site that rebases positionOffsets
    /// independently of the per-step lockstep advance (init, filter, extend,
    /// state/metaState set, the rotating temporal restore) also assigns
    /// leftPadding, while `update()` — which does NOT bump the epoch —
    /// advances positionOffsets and `idx` together, so the cached DELTA below
    /// stays true across decode steps.
    private var paddingGeneration: UInt64 = 0
    /// `positionOffsets[0] - idx` for the sole row. NOT derivable from
    /// leftPadding: a rotating cache restored by merge reports its ABSOLUTE
    /// position (e.g. offset 6 over a 4-token kept window), which
    /// `idx - leftPadding` cannot reconstruct — BatchCacheShapeTests
    /// testMergeRestoresTemporalOrder holds that line.
    private var scalarDeltaCache: (generation: UInt64, delta: Int)?

    /// Whether a one-row cache reports its RoPE position as `.scalar`.
    /// Escape hatch, not a tuning knob: `.scalar(p)` and `.batch([p])` are the
    /// same number for a single row — but `.scalar` is eligible for MLX's RoPE
    /// fast path (single-token, row-contiguous) where a length-1 array offset
    /// forces the general kernel on every layer of every decode step. That gap
    /// showed up as gemma-4-E2B short/c1 e2e dropping to 0.87x vs best rival
    /// the day the per-row anchor landed. Set MLXCAT_SINGLE_ROW_SCALAR_ROPE=0
    /// to restore the unconditional `.batch` for A/B or fault isolation.
    private static let scalarSingleRowEnabled: Bool =
        ProcessInfo.processInfo.environment["MLXCAT_SINGLE_ROW_SCALAR_ROPE"] != "0"

    public var ropeOffset: RoPEOffset {
        // Sequence layout only: batchState layers (hybrid Mamba conv/ssm)
        // reassign leftPadding on every update(), so the lazy read below would
        // put a host sync on their hot path. Sequence-layout padding changes
        // only at admission boundaries (init/filter/extend/state/metaState).
        if Self.scalarSingleRowEnabled, case .sequence = layout, batchSize == 1 {
            if scalarDeltaCache?.generation != paddingGeneration {
                scalarDeltaCache = (paddingGeneration, positionOffsets.item(Int.self) - idx)
            }
            if let cached = scalarDeltaCache {
                return .scalar(idx + cached.delta)
            }
        }
        // Snapshot (+ 0) before cache.update(...) advances the offsets — same
        // contract as the BatchPositionedKVCache default this replaces.
        return .batch(positionOffsets + 0)
    }

    public func innerState() -> [MLXArray] {
        state
    }

    public static func merge(_ caches: [any KVCache]) throws -> BatchKVCache {
        guard !caches.isEmpty else {
            throw BatchKVCacheError.emptyMerge
        }

        let rawStates = caches.map(\.state)
        let cacheTypes = caches.map { String(describing: type(of: $0)) }
        guard let firstState = rawStates.first, !firstState.isEmpty else {
            throw BatchKVCacheError.emptyState(cacheType: cacheTypes.first ?? "unknown")
        }

        for (state, cacheType) in zip(rawStates, cacheTypes) where state.count != firstState.count {
            throw BatchKVCacheError.inconsistentStateCount(
                expected: firstState.count,
                actual: state.count,
                cacheType: cacheType
            )
        }

        let layout = try inferLayout(cacheType: cacheTypes[0], state: firstState)
        switch layout {
        case .sequence(let sequenceAxis):
            let states = rawStates.enumerated().map { row, state in
                normalizedSequenceState(
                    cache: caches[row],
                    cacheType: cacheTypes[row],
                    state: state,
                    sequenceAxis: sequenceAxis
                )
            }
            let lengths = try states.enumerated().map { row, state in
                try sequenceLength(cacheType: cacheTypes[row], state: state, sequenceAxis: sequenceAxis)
            }
            let positionOffsets = zip(caches, lengths).map { cache, length in
                max(cache.offset, length)
            }
            return try mergeSequenceCaches(
                states: states,
                cacheTypes: cacheTypes,
                sequenceAxis: sequenceAxis,
                rowMetaStates: caches.map(\.metaState),
                positionOffsets: positionOffsets
            )
        case .batchState:
            throw BatchKVCacheError.unsupportedSequenceState(
                cacheType: cacheTypes[0],
                stateShapes: firstState.map(\.shape)
            )
        }
    }

    public func extract(_ row: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        guard row >= 0, row < batchSize else { return cache }

        switch layout {
        case .sequence(let sequenceAxis):
            let padding = leftPadding.asArray(Int.self)[row]
            cache.state = buffers.map {
                Self.slice($0, row: row, sequenceAxis: sequenceAxis, range: padding ..< idx)
            }
        case .batchState:
            cache.state = buffers.map { Self.sliceRow($0, row: row) }
        }
        if row < rowMetaStates.count,
            let metaState = Self.kvCacheSimpleCompatibleMetaState(rowMetaStates[row])
        {
            cache.metaState = metaState
        }
        return cache
    }

    public func filter(keeping rows: [Int]) {
        guard !rows.isEmpty else {
            buffers.removeAll()
            rowMetaStates.removeAll()
            leftPadding = MLXArray([Int32]())
            positionOffsets = MLXArray([Int32]())
            idx = 0
            return
        }

        let rowIndices = MLXArray(rows.map(Int32.init))
        buffers = buffers.map { $0.take(rowIndices, axis: 0) }
        leftPadding = leftPadding.take(rowIndices, axis: 0)
        positionOffsets = positionOffsets.take(rowIndices, axis: 0)
        rowMetaStates = rows.map { row in
            row < rowMetaStates.count ? rowMetaStates[row] : []
        }

        guard case .sequence(let sequenceAxis) = layout else { return }

        let minLeftPadding = leftPadding.min().item(Int.self)
        guard minLeftPadding > 0 else { return }

        buffers = buffers.map {
            Self.slice($0, sequenceAxis: sequenceAxis, range: minLeftPadding ..< idx)
        }
        idx -= minLeftPadding
        leftPadding = leftPadding - MLXArray(Int32(minLeftPadding))
    }

    public func insert(_ cache: any KVCache) throws {
        try extend([cache])
    }

    public func extend(_ caches: [any KVCache]) throws {
        guard !caches.isEmpty else { return }
        try extend(BatchKVCache.merge(caches))
    }

    public func extend(_ other: BatchKVCache) throws {
        guard other.batchSize > 0 else { return }
        guard layout == other.layout else {
            throw BatchKVCacheError.incompatibleLayout(
                expected: layout.description,
                actual: other.layout.description
            )
        }
        guard batchSize > 0, !buffers.isEmpty else {
            state = other.state
            leftPadding = other.leftPadding
            positionOffsets = other.positionOffsets
            rowMetaStates = other.rowMetaStates
            idx = other.idx
            layout = other.layout
            return
        }

        switch layout {
        case .sequence(let sequenceAxis):
            let targetIdx = max(idx, other.idx)
            let normalizedCurrent = normalize(
                buffers: buffers.map { Self.slice($0, sequenceAxis: sequenceAxis, range: ..<idx) },
                leftPadding: leftPadding,
                from: idx,
                to: targetIdx,
                sequenceAxis: sequenceAxis
            )
            let normalizedOther = normalize(
                buffers: other.buffers.map { Self.slice($0, sequenceAxis: sequenceAxis, range: ..<other.idx) },
                leftPadding: other.leftPadding,
                from: other.idx,
                to: targetIdx,
                sequenceAxis: sequenceAxis
            )

            buffers = zip(normalizedCurrent.buffers, normalizedOther.buffers).map {
                concatenated([$0, $1], axis: 0)
            }
            leftPadding = concatenated([normalizedCurrent.leftPadding, normalizedOther.leftPadding], axis: 0)
            positionOffsets = concatenated([positionOffsets, other.positionOffsets], axis: 0)
            idx = targetIdx
        case .batchState:
            buffers = zip(buffers, other.buffers).map { concatenated([$0, $1], axis: 0) }
            leftPadding = concatenated([leftPadding, other.leftPadding], axis: 0)
            positionOffsets = concatenated([positionOffsets, other.positionOffsets], axis: 0)
        }

        rowMetaStates.append(contentsOf: other.rowMetaStates)
    }

    // MARK: Right-padded ragged prefill

    /// Set while a ragged multi-row prefill write is outstanding, cleared by
    /// ``finalize()``. Only rows that FINISH their prompt in the current write
    /// ever carry padding — a row still prefilling gets 0 and is untouched by
    /// the roll below — so padding can never end up interleaved between two of
    /// a row's own chunks.
    private var rightPadding: MLXArray?

    public var hasPendingRightPadding: Bool { rightPadding != nil }

    /// Declare that the next write is right-padded: row `i` carries
    /// `padding[i]` junk positions after its real tokens.
    ///
    /// Right padding needs no mask. Padding sits AFTER a row's real tokens, and
    /// under causal attention a real token at position p attends only to <= p,
    /// so it can never see its own padding; the padded rows' logits are
    /// discarded. That is why upstream pads prompts on the right and decode
    /// batches on the left (`guest/mlx-lm/mlx_lm/models/cache.py:967-988`) —
    /// the two paddings buy different things and only the left one costs a mask.
    public func prepare(rightPadding padding: [Int]) throws {
        guard case .sequence = layout else {
            throw BatchKVCacheError.unsupportedRightPadding(layout: layout.description)
        }
        guard padding.count == batchSize else {
            throw BatchKVCacheError.rightPaddingWidthMismatch(
                expected: batchSize,
                actual: padding.count
            )
        }
        guard let maxPadding = padding.max(), maxPadding > 0 else { return }
        guard rightPadding == nil else {
            throw BatchKVCacheError.rightPaddingAlreadyPrepared
        }
        rightPadding = MLXArray(padding.map(Int32.init))
    }

    /// Convert an outstanding right-padded write into the left-padded layout
    /// decode expects, by rolling each row right by its own padding: the junk
    /// moves from the tail of the written window to its head, every row's tip
    /// lands on the same column, and `positionOffsets`/`leftPadding` absorb the
    /// shift. Mirrors `BatchKVCache.finalize` upstream
    /// (`guest/mlx-lm/mlx_lm/models/cache.py:980-987`).
    ///
    /// The roll covers exactly the WRITTEN window `0 ..< idx`, never the
    /// over-allocated capacity beyond it. Rolling the whole buffer would wrap
    /// capacity zeros into the valid window and push the last `padding` real
    /// entries out past `idx` where nothing reads them — silent truncation of
    /// the tip, which is the token the row is about to decode from.
    public func finalize() {
        guard let padding = rightPadding else { return }
        rightPadding = nil
        guard case .sequence(let sequenceAxis) = layout, idx > 0 else { return }

        for bufferIndex in buffers.indices {
            let written = Self.slice(buffers[bufferIndex], sequenceAxis: sequenceAxis, range: 0 ..< idx)
            let rolled = Self.dynamicRoll(written, shifts: padding, axis: sequenceAxis)
            buffers[bufferIndex][
                Self.indices(
                    sequenceAxis: sequenceAxis,
                    range: 0 ..< idx,
                    rank: buffers[bufferIndex].shape.count
                )
            ] = rolled
        }

        positionOffsets = positionOffsets - padding
        leftPadding = leftPadding + padding
    }

    /// Per-row circular shift along `axis`, shifting row `i` right by
    /// `shifts[i]`. Port of upstream's `dynamic_roll`
    /// (`guest/mlx-lm/mlx_lm/models/cache.py:903-909`) built from `takeAlong`,
    /// since mlx-swift exposes no roll op.
    private static func dynamicRoll(_ array: MLXArray, shifts: MLXArray, axis: Int) -> MLXArray {
        let rank = array.shape.count
        let length = array.dim(axis)

        var shiftShape = Array(repeating: 1, count: rank)
        shiftShape[0] = shifts.dim(0)

        var positionShape = Array(repeating: 1, count: rank)
        positionShape[axis] = length

        let positions = MLXArray((0 ..< length).map(Int32.init)).reshaped(positionShape)
        // Positive modulo: MLX's remainder keeps the dividend's sign, and a
        // negative index would read from the wrong end of the row.
        let raw = positions - shifts.reshaped(shiftShape)
        let wrapped = remainder(remainder(raw, Int32(length)) + Int32(length), Int32(length))
        return takeAlong(array, broadcast(wrapped, to: array.shape), axis: axis)
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        switch layout {
        case .sequence(let sequenceAxis):
            let previous = idx
            ensureCapacity(keys: newKeys, values: newValues, sequenceAxis: sequenceAxis)
            idx += newKeys.dim(sequenceAxis)
            positionOffsets = positionOffsets + MLXArray(Int32(newKeys.dim(sequenceAxis)))

            buffers[0][Self.indices(sequenceAxis: sequenceAxis, range: previous ..< idx, rank: buffers[0].shape.count)] = newKeys
            buffers[1][Self.indices(sequenceAxis: sequenceAxis, range: previous ..< idx, rank: buffers[1].shape.count)] = newValues

            return (
                Self.slice(buffers[0], sequenceAxis: sequenceAxis, range: ..<idx),
                Self.slice(buffers[1], sequenceAxis: sequenceAxis, range: ..<idx)
            )
        case .batchState:
            buffers = [newKeys, newValues]
            leftPadding = MLXArray(Array(repeating: Int32(0), count: newKeys.dim(0)))
            positionOffsets = MLXArray(Array(repeating: Int32(0), count: newKeys.dim(0)))
            return (newKeys, newValues)
        }
    }

    public var state: [MLXArray] {
        get {
            switch layout {
            case .sequence(let sequenceAxis):
                return buffers.map { Self.slice($0, sequenceAxis: sequenceAxis, range: ..<idx) }
            case .batchState:
                return buffers
            }
        }
        set {
            if case .sequence(let sequenceAxis) = layout {
                guard newValue.count == 2 else {
                    return
                }
                buffers = newValue
                idx = newValue[0].dim(sequenceAxis)
                leftPadding = MLXArray(Array(repeating: Int32(0), count: newValue[0].dim(0)))
                positionOffsets = MLXArray(Array(repeating: Int32(idx), count: newValue[0].dim(0)))
            } else {
                buffers = newValue
                leftPadding = MLXArray(Array(repeating: Int32(0), count: newValue.first?.dim(0) ?? 0))
                positionOffsets = MLXArray(Array(repeating: Int32(0), count: newValue.first?.dim(0) ?? 0))
            }
        }
    }

    public var metaState: [String] {
        get {
            [
                String(idx),
                leftPadding.asArray(Int.self).map(String.init).joined(separator: ","),
                layout.description,
                positionOffsets.asArray(Int.self).map(String.init).joined(separator: ","),
            ]
        }
        set {
            guard newValue.count >= 2 else { return }
            idx = Int(newValue[0]) ?? 0
            let padding = newValue[1].split(separator: ",").compactMap { Int($0) }
            leftPadding = MLXArray(padding.map(Int32.init))
            if newValue.count >= 4 {
                let offsets = newValue[3].split(separator: ",").compactMap { Int($0) }
                positionOffsets = MLXArray(offsets.map(Int32.init))
            } else {
                positionOffsets = MLXArray(padding.map { Int32(idx - $0) })
            }
        }
    }

    public func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard case .sequence = layout else {
            return .none
        }
        guard let mask = CausalMask.create(
            n: n,
            offset: idx,
            leftPadding: leftPadding,
            windowSize: windowSize
        ) else {
            return .none
        }
        var additiveMask = MLX.where(mask, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        if let dtype = buffers.first?.dtype {
            additiveMask = additiveMask.asType(dtype)
        }
        return .array(additiveMask)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        guard case .sequence = layout else { return 0 }
        let trimmed = min(idx, n)
        guard trimmed > 0 else { return 0 }

        let currentIdx = idx
        let currentLeftPadding = leftPadding.asArray(Int.self)
        let currentPositionOffsets = positionOffsets.asArray(Int.self)
        let rowTrimmed = currentLeftPadding.map { padding in
            min(trimmed, max(0, currentIdx - padding))
        }

        idx -= trimmed
        leftPadding = MLXArray(zip(currentLeftPadding, rowTrimmed).map { padding, rowTrimmed in
            let remainingLength = max(0, currentIdx - padding - rowTrimmed)
            return Int32(idx - remainingLength)
        })
        positionOffsets = MLXArray(zip(currentPositionOffsets, rowTrimmed).map { offset, rowTrimmed in
            Int32(offset - rowTrimmed)
        })
        return trimmed
    }

    public func copy() -> any KVCache {
        let copy = BatchKVCache(leftPadding: leftPadding.asArray(Int.self), idx: idx, step: step)
        copy.layout = layout
        copy.buffers = buffers.map { $0[.ellipsis] }
        copy.rowMetaStates = rowMetaStates
        copy.positionOffsets = positionOffsets
        return copy
    }

    private static func mergeSequenceCaches(
        states: [[MLXArray]],
        cacheTypes: [String],
        sequenceAxis: Int,
        rowMetaStates: [[String]],
        positionOffsets: [Int]
    ) throws -> BatchKVCache {
        let lengths = try states.enumerated().map { row, state in
            try sequenceLength(cacheType: cacheTypes[row], state: state, sequenceAxis: sequenceAxis)
        }
        let maxLength = lengths.max() ?? 0
        let padding = lengths.map { maxLength - $0 }
        let batchSize = states.count
        let firstState = states[0]

        var batchBuffers: [MLXArray] = []
        for (stateIndex, firstArray) in firstState.enumerated() {
            var batchShape = firstArray.shape
            batchShape[0] = batchSize
            batchShape[sequenceAxis] = maxLength
            batchBuffers.append(MLXArray.zeros(batchShape, dtype: firstArray.dtype))

            for (row, state) in states.enumerated() {
                let array = state[stateIndex]
                try validateCompatibleShape(
                    cacheType: cacheTypes[row],
                    stateIndex: stateIndex,
                    expected: firstArray.shape,
                    actual: array.shape,
                    flexibleAxes: [0, sequenceAxis]
                )
                batchBuffers[stateIndex][indices(row: row, sequenceAxis: sequenceAxis, range: padding[row] ..< padding[row] + lengths[row], rank: array.shape.count)] = array
            }
        }

        let cache = BatchKVCache(leftPadding: padding, idx: maxLength)
        cache.layout = .sequence(axis: sequenceAxis)
        cache.buffers = batchBuffers
        cache.rowMetaStates = rowMetaStates
        cache.positionOffsets = MLXArray(positionOffsets.map(Int32.init))
        return cache
    }

    private static func mergeBatchStateCaches(
        states: [[MLXArray]],
        cacheTypes: [String],
        rowMetaStates: [[String]]
    ) throws -> BatchKVCache {
        let firstState = states[0]
        var batchBuffers: [MLXArray] = []
        for stateIndex in firstState.indices {
            let firstArray = firstState[stateIndex]
            for row in states.indices {
                try validateCompatibleShape(
                    cacheType: cacheTypes[row],
                    stateIndex: stateIndex,
                    expected: firstArray.shape,
                    actual: states[row][stateIndex].shape,
                    flexibleAxes: [0]
                )
            }
            batchBuffers.append(concatenated(states.map { $0[stateIndex] }, axis: 0))
        }

        let cache = BatchKVCache(leftPadding: Array(repeating: 0, count: states.count), idx: 0)
        cache.layout = .batchState
        cache.buffers = batchBuffers
        cache.rowMetaStates = rowMetaStates
        return cache
    }

    private static func inferLayout(cacheType: String, state: [MLXArray]) throws -> Layout {
        guard state.count == 2 else {
            throw BatchKVCacheError.unsupportedSequenceState(
                cacheType: cacheType,
                stateShapes: state.map(\.shape)
            )
        }

        let ranks = state.map { $0.shape.count }
        if ranks[0] == ranks[1], ranks[0] >= 3 {
            return .sequence(axis: ranks[0] - 2)
        }
        return .batchState
    }

    private static func normalizedSequenceState(
        cache: any KVCache,
        cacheType: String,
        state: [MLXArray],
        sequenceAxis: Int
    ) -> [MLXArray] {
        guard isKnownRotatingCache(cacheType) else { return state }
        guard let metadata = rotatingMetadata(from: cache.metaState) else { return state }
        guard state.count == 2 else { return state }
        return state.map {
            temporalRotatingArray($0, metadata: metadata, sequenceAxis: sequenceAxis)
        }
    }

    private static func rotatingMetadata(from metaState: [String]) -> RotatingCacheMetadata? {
        guard metaState.count == 5,
            let keep = Int(metaState[0]),
            Int(metaState[1]) != nil,
            Int(metaState[2]) != nil,
            let offset = Int(metaState[3]),
            let idx = Int(metaState[4])
        else {
            return nil
        }
        return RotatingCacheMetadata(keep: keep, offset: offset, idx: idx)
    }

    private static func temporalRotatingArray(
        _ array: MLXArray,
        metadata: RotatingCacheMetadata,
        sequenceAxis: Int
    ) -> MLXArray {
        let length = array.dim(sequenceAxis)
        let idx = min(metadata.idx, length)
        let keep = min(metadata.keep, idx)

        if idx == length {
            return array
        }
        if idx < metadata.offset {
            var parts: [MLXArray] = []
            if keep > 0 {
                parts.append(slice(array, sequenceAxis: sequenceAxis, range: 0 ..< keep))
            }
            if idx < length {
                parts.append(slice(array, sequenceAxis: sequenceAxis, range: idx ..< length))
            }
            if keep < idx {
                parts.append(slice(array, sequenceAxis: sequenceAxis, range: keep ..< idx))
            }
            return concatenated(parts, axis: sequenceAxis)
        }
        return slice(array, sequenceAxis: sequenceAxis, range: ..<idx)
    }

    private static func kvCacheSimpleCompatibleMetaState(_ metaState: [String]) -> [String]? {
        guard metaState.count == 1, metaState[0].isEmpty else { return nil }
        return metaState
    }

    private static func isKnownRotatingCache(_ cacheType: String) -> Bool {
        cacheType.localizedCaseInsensitiveContains("rotating")
            || cacheType.localizedCaseInsensitiveContains("circular")
    }

    private static func sequenceLength(
        cacheType: String,
        state: [MLXArray],
        sequenceAxis: Int
    ) throws -> Int {
        let lengths = state.map { $0.dim(sequenceAxis) }
        guard let firstLength = lengths.first, lengths.allSatisfy({ $0 == firstLength }) else {
            throw BatchKVCacheError.unsupportedSequenceState(
                cacheType: cacheType,
                stateShapes: state.map(\.shape)
            )
        }
        return firstLength
    }

    private static func validateCompatibleShape(
        cacheType: String,
        stateIndex: Int,
        expected: [Int],
        actual: [Int],
        flexibleAxes: Set<Int>
    ) throws {
        guard expected.count == actual.count else {
            throw BatchKVCacheError.incompatibleStateShape(
                cacheType: cacheType,
                stateIndex: stateIndex,
                expected: expected,
                actual: actual
            )
        }
        for axis in expected.indices where !flexibleAxes.contains(axis) && expected[axis] != actual[axis] {
            throw BatchKVCacheError.incompatibleStateShape(
                cacheType: cacheType,
                stateIndex: stateIndex,
                expected: expected,
                actual: actual
            )
        }
    }

    private func ensureCapacity(keys newKeys: MLXArray, values newValues: MLXArray, sequenceAxis: Int) {
        guard !buffers.isEmpty else {
            buffers = [
                Self.capacityBuffer(for: newKeys, sequenceAxis: sequenceAxis, step: step),
                Self.capacityBuffer(for: newValues, sequenceAxis: sequenceAxis, step: step),
            ]
            return
        }

        guard idx + newKeys.dim(sequenceAxis) > buffers[0].dim(sequenceAxis) else { return }

        let keyExtension = Self.capacityBuffer(for: newKeys, sequenceAxis: sequenceAxis, step: step)
        let valueExtension = Self.capacityBuffer(for: newValues, sequenceAxis: sequenceAxis, step: step)
        buffers[0] = concatenated([buffers[0], keyExtension], axis: sequenceAxis)
        buffers[1] = concatenated([buffers[1], valueExtension], axis: sequenceAxis)
    }

    private static func capacityBuffer(for array: MLXArray, sequenceAxis: Int, step: Int) -> MLXArray {
        let nSteps = (step + array.dim(sequenceAxis) - 1) / step
        var shape = array.shape
        shape[sequenceAxis] = nSteps * step
        return MLXArray.zeros(shape, dtype: array.dtype)
    }

    private func normalize(
        buffers inputBuffers: [MLXArray],
        leftPadding inputLeftPadding: MLXArray,
        from currentIdx: Int,
        to targetIdx: Int,
        sequenceAxis: Int
    ) -> (buffers: [MLXArray], leftPadding: MLXArray) {
        let delta = targetIdx - currentIdx
        guard delta > 0 else {
            return (inputBuffers, inputLeftPadding)
        }

        let normalizedBuffers = inputBuffers.map { input in
            var shape = input.shape
            shape[sequenceAxis] = delta
            let padding = MLXArray.zeros(shape, dtype: input.dtype)
            return concatenated([padding, input], axis: sequenceAxis)
        }
        return (
            normalizedBuffers,
            inputLeftPadding + MLXArray(Int32(delta))
        )
    }

    private static func sliceRow(_ array: MLXArray, row: Int) -> MLXArray {
        array[indices(row: row, rank: array.shape.count)]
    }

    private static func slice(_ array: MLXArray, row: Int, sequenceAxis: Int, range: Range<Int>) -> MLXArray {
        array[indices(row: row, sequenceAxis: sequenceAxis, range: range, rank: array.shape.count)]
    }

    private static func slice(_ array: MLXArray, sequenceAxis: Int, range: Range<Int>) -> MLXArray {
        array[indices(sequenceAxis: sequenceAxis, range: range, rank: array.shape.count)]
    }

    private static func slice(_ array: MLXArray, sequenceAxis: Int, range: PartialRangeUpTo<Int>) -> MLXArray {
        array[indices(sequenceAxis: sequenceAxis, range: range, rank: array.shape.count)]
    }

    private static func indices(row: Int, rank: Int) -> [any MLXArrayIndex] {
        var result: [any MLXArrayIndex] = Array(repeating: 0..., count: rank)
        result[0] = row ..< row + 1
        return result
    }

    private static func indices(sequenceAxis: Int, range: Range<Int>, rank: Int) -> [any MLXArrayIndex] {
        var result: [any MLXArrayIndex] = Array(repeating: 0..., count: rank)
        result[sequenceAxis] = range
        return result
    }

    private static func indices(sequenceAxis: Int, range: PartialRangeUpTo<Int>, rank: Int) -> [any MLXArrayIndex] {
        var result: [any MLXArrayIndex] = Array(repeating: 0..., count: rank)
        result[sequenceAxis] = range
        return result
    }

    private static func indices(row: Int, sequenceAxis: Int, range: Range<Int>, rank: Int) -> [any MLXArrayIndex] {
        var result: [any MLXArrayIndex] = Array(repeating: 0..., count: rank)
        result[0] = row ..< row + 1
        result[sequenceAxis] = range
        return result
    }
}
