import Foundation
import MLX
import MLXLMCommon

public struct SessionPrefixKVStoreStats: Equatable, Sendable {
    public let fetchHitCount: Int
    public let storeCount: Int
    public let releaseCount: Int
    public let clearCount: Int
    public let evictionCount: Int
    /// Anonymous slots dropped because a newer slot held everything they could serve.
    public let supersededCount: Int
    public let currentBytes: Int64
    public let slotCount: Int
}

public final class SessionPrefixKVStore: PrefixKVStore, @unchecked Sendable {
    private final class Slot {
        let id: UUID
        let sessionKey: String?
        var tokens: [Int]
        var layers: [SerializedKVLayer]
        var byteCount: Int64
        var lastAccess: UInt64
        var leaseCount: Int
        /// Recurrent-state checkpoints strictly inside `tokens`, keyed by position.
        var checkpoints: [Int: [SerializedKVLayer?]] = [:]
        /// Whether any layer cannot trim (a hybrid model's recurrent state). Such a
        /// slot is exact only at `tokens.count` and at its checkpoints.
        var hasRecurrentLayers = false
        /// How many of `tokens` are the request's prompt. The scheduler passes it; when a
        /// caller does not, an anonymous slot that extends another takes that slot's (a
        /// request publishes its bare prompt first, then longer prefixes of the same row).
        /// Nil: unknown, read as `tokens.count`.
        var promptLength: Int?

        init(
            id: UUID = UUID(),
            sessionKey: String?,
            tokens: [Int],
            layers: [SerializedKVLayer],
            byteCount: Int64,
            lastAccess: UInt64
        ) {
            self.id = id
            self.sessionKey = sessionKey
            self.tokens = tokens
            self.layers = layers
            self.byteCount = byteCount
            self.lastAccess = lastAccess
            self.leaseCount = 0
        }
    }

    private struct HitStorage {
        let slotID: UUID
        let trimCount: Int
        /// Set when the hit resumes at a recurrent checkpoint rather than at a
        /// trim of the slot's own end state.
        let checkpointPosition: Int?
    }

    private let lock = NSRecursiveLock()
    private let layerDecoder: @Sendable (SerializedKVLayer) throws -> any KVCache
    let supportsWindowedCheckpoints: Bool
    private let maxBytes: Int64
    private let maxAnonymousSlots: Int
    private let disk: SessionPrefixDiskStore?
    private var sessionSlots: [String: Slot] = [:]
    private var anonymousSlots: [Slot] = []
    private var sequence: UInt64 = 0
    private var currentBytes: Int64 = 0
    private var _fetchHitCount = 0
    private var _storeCount = 0
    private var _releaseCount = 0
    private var _clearCount = 0
    private var _evictionCount = 0
    private var _supersededCount = 0

    /// How many tokens at the end of an anonymous slot's prompt the next turn of the same chat may
    /// render differently. The generation prompt (`<|im_start|>assistant\n`, an empty think block,
    /// a channel header) is re-rendered in the next turn's history: measured with Qwen3-0.6B, a
    /// 7,263-token prompt and its next turn share 7,261 tokens. Dropping a slot this rule matches
    /// costs at most this many prompt tokens of prefill.
    static let promptTailTolerance = 32

    public convenience init(maxBytes: Int64 = 0, maxAnonymousSlots: Int = 8) {
        self.init(maxBytes: maxBytes, maxAnonymousSlots: maxAnonymousSlots, windowedLayerDecoder: nil)
    }

    package init(
        maxBytes: Int64, maxAnonymousSlots: Int = 8,
        windowedLayerDecoder: (@Sendable (SerializedKVLayer) throws -> any KVCache)?,
        disk: SessionPrefixDiskStore? = nil
    ) {
        self.layerDecoder = windowedLayerDecoder ?? BlockAwarePrefixKVStore.cache(from:)
        self.supportsWindowedCheckpoints = windowedLayerDecoder != nil
        self.maxBytes = max(0, maxBytes)
        self.maxAnonymousSlots = max(1, maxAnonymousSlots)
        self.disk = disk
    }

    func restoreLayer(_ layer: SerializedKVLayer) throws -> any KVCache {
        try layerDecoder(layer)
    }

    public var stats: SessionPrefixKVStoreStats {
        withLock {
            SessionPrefixKVStoreStats(
                fetchHitCount: _fetchHitCount,
                storeCount: _storeCount,
                releaseCount: _releaseCount,
                clearCount: _clearCount,
                evictionCount: _evictionCount,
                supersededCount: _supersededCount,
                currentBytes: currentBytes,
                slotCount: sessionSlots.count + anonymousSlots.count
            )
        }
    }

    public func fetch(tokens: [Int], sessionKey: String?) -> PrefixKVStoreHit? {
        withLock {
            guard tokens.count > 1 else { return nil }
            if let sessionKey, !sessionKey.isEmpty, sessionSlots[sessionKey] == nil,
                let saved = disk?.read(session: sessionKey)
            {
                try? store(tokens: saved.tokens, sessionKey: sessionKey, cache: saved.layers,
                           checkpoints: saved.checkpoints, persist: false)
            }
            let candidates = candidateSlots(sessionKey: sessionKey)
                .filter { $0.leaseCount == 0 }
                .compactMap { slot -> (slot: Slot, matched: Int, checkpoint: Int?)? in
                    let matched = commonPrefixLength(slot.tokens, tokens)
                    guard matched > 0 else { return nil }
                    Scheduler.prefixDebug(
                        "fetch slot tokens=\(slot.tokens.count) matched=\(matched) recurrent=\(slot.hasRecurrentLayers) "
                            + "checkpoints=\(slot.checkpoints.keys.sorted()) classes=\(Set(slot.layers.map(\.className)))")
                    // A hybrid slot's end state is exact only at `tokens.count`, so
                    // a shorter match resumes at the deepest checkpoint it covers,
                    // or is not a candidate at all: handing it out would only fail
                    // the recurrent trim later AND shadow a slot that could serve.
                    if matched < slot.tokens.count, slot.hasRecurrentLayers {
                        guard let position = slot.checkpoints.keys.filter({ $0 <= matched }).max(),
                            position > 0
                        else { return nil }
                        return (slot, position, position)
                    }
                    return (slot, matched, nil)
                }
                .sorted { lhs, rhs in
                    if lhs.matched == rhs.matched {
                        return lhs.slot.lastAccess > rhs.slot.lastAccess
                    }
                    return lhs.matched > rhs.matched
                }
            guard let best = candidates.first else { return nil }
            best.slot.leaseCount += 1
            best.slot.lastAccess = nextSequence()
            _fetchHitCount += 1
            return PrefixKVStoreHit(
                matchedTokenCount: best.matched,
                blockCount: 1,
                storage: HitStorage(
                    slotID: best.slot.id,
                    trimCount: max(0, best.slot.tokens.count - best.matched),
                    checkpointPosition: best.checkpoint
                )
            )
        }
    }

    public func preload(_ hit: PrefixKVStoreHit) throws {
        _ = try reconstructCache(from: hit)
    }

    public func reconstructCache(from hit: PrefixKVStoreHit) throws -> [SerializedKVLayer] {
        try withLock {
            guard let storage = hit.storage as? HitStorage,
                let slot = slot(id: storage.slotID)
            else {
                throw PrefixKVStoreError.invalidHit
            }
            let checkpoint = storage.checkpointPosition.flatMap { slot.checkpoints[$0] }
            if storage.checkpointPosition != nil, checkpoint == nil {
                throw PrefixKVStoreError.invalidHit
            }
            return try slot.layers.enumerated().map { index, layer in
                if let recurrent = checkpoint?[index] ?? nil {
                    return recurrent
                }
                // At the exact boundary all arrays, including any rolling state, are valid.
                if supportsWindowedCheckpoints, storage.trimCount == 0 { return layer }
                let cache = try restoreLayer(layer)
                if storage.trimCount > 0, cache.trim(storage.trimCount) != storage.trimCount {
                    throw PrefixKVStoreError.unsupportedCacheTrim(layer.className)
                }
                return SerializedKVLayer(
                    state: cache.state,
                    metaState: cache.metaState,
                    className: layer.className
                )
            }
        }
    }

    public func store(tokens: [Int], sessionKey: String?, cache: [SerializedKVLayer]) throws {
        try store(tokens: tokens, sessionKey: sessionKey, cache: cache, checkpoints: [])
    }

    public func store(
        tokens: [Int],
        sessionKey: String?,
        cache: [SerializedKVLayer],
        checkpoints: [PrefixRecurrentCheckpoint]
    ) throws {
        try store(tokens: tokens, sessionKey: sessionKey, cache: cache, checkpoints: checkpoints, persist: true)
    }

    public func store(
        tokens: [Int],
        sessionKey: String?,
        cache: [SerializedKVLayer],
        checkpoints: [PrefixRecurrentCheckpoint],
        promptLength: Int
    ) throws {
        try store(
            tokens: tokens, sessionKey: sessionKey, cache: cache, checkpoints: checkpoints, persist: true,
            promptLength: promptLength)
    }

    /// `promptLength` nil (a caller that does not know it): an anonymous slot that extends
    /// another takes that slot's prompt length, which holds within one request's publishes.
    private func store(
        tokens: [Int], sessionKey: String?, cache: [SerializedKVLayer],
        checkpoints: [PrefixRecurrentCheckpoint], persist: Bool, promptLength: Int? = nil
    ) throws {
        try withLock {
            guard tokens.count > 1, !cache.isEmpty else { return }
            let copied = try cache.map { layer -> SerializedKVLayer in
                _ = try restoreLayer(layer)
                return SerializedKVLayer(
                    state: layer.state,
                    metaState: layer.metaState,
                    className: layer.className
                )
            }
            let byteCount = copied.reduce(Int64(0)) { total, layer in
                total + layer.state.reduce(Int64(0)) { $0 + estimatedBytes($1) }
            }
            let slot = Slot(
                sessionKey: sessionKey,
                tokens: tokens,
                layers: copied,
                byteCount: byteCount,
                lastAccess: nextSequence()
            )
            slot.hasRecurrentLayers = recurrentLayers(of: copied).contains { $0 != nil }
            if let promptLength, promptLength > 0 {
                slot.promptLength = min(promptLength, tokens.count)
            }
            Scheduler.prefixDebug(
                "store tokens=\(tokens.count) layers=\(copied.count) recurrent=\(slot.hasRecurrentLayers) "
                    + "offered=\(checkpoints.map { "\($0.position)/\($0.layers.count)" })")
            for checkpoint in checkpoints
            where checkpoint.position > 0 && checkpoint.position < tokens.count
                && checkpoint.layers.count == copied.count
            {
                slot.checkpoints[checkpoint.position] = checkpoint.layers
            }
            if let sessionKey, !sessionKey.isEmpty {
                if let old = sessionSlots[sessionKey] {
                    guard old.leaseCount == 0 else {
                        return
                    }
                    currentBytes -= old.byteCount
                    // A session slot is replaced by its own continuation (prompt
                    // → prompt+reply → next turn). When the new slot extends the
                    // old one, the old one's checkpoints and its exact end state
                    // stay valid positions of the new slot; keep them, or the
                    // next turn — which re-renders the reply and diverges inside
                    // it — has nothing to resume from.
                    if !old.checkpoints.isEmpty || !slot.checkpoints.isEmpty,
                        old.tokens.count < tokens.count,
                        commonPrefixLength(old.tokens, tokens) == old.tokens.count
                    {
                        for (position, layers) in old.checkpoints where slot.checkpoints[position] == nil {
                            slot.checkpoints[position] = layers
                        }
                        if slot.checkpoints[old.tokens.count] == nil {
                            slot.checkpoints[old.tokens.count] = recurrentLayers(of: old.layers)
                        }
                    }
                }
                sessionSlots[sessionKey] = slot
            } else {
                supersedeAnonymousSlots(by: slot)
                anonymousSlots.append(slot)
            }
            slot.byteCount += slot.checkpoints.values.reduce(Int64(0)) { total, layers in
                total + layers.compactMap { $0 }.reduce(Int64(0)) { sum, layer in
                    sum + layer.state.reduce(Int64(0)) { $0 + estimatedBytes($1) }
                }
            }
            currentBytes += slot.byteCount
            _storeCount += 1
            evictIfNeeded()
            if persist, let disk, let sessionKey, !sessionKey.isEmpty {
                disk.store(session: sessionKey, snapshot: .init(
                    tokens: slot.tokens, layers: slot.layers,
                    checkpoints: slot.checkpoints.map { PrefixRecurrentCheckpoint(position: $0.key, layers: $0.value) }),
                    byteCount: slot.byteCount)
            }
        }
    }

    public func release(_ hit: PrefixKVStoreHit) {
        withLock {
            guard let storage = hit.storage as? HitStorage,
                let slot = slot(id: storage.slotID)
            else {
                return
            }
            slot.leaseCount = max(0, slot.leaseCount - 1)
            slot.lastAccess = nextSequence()
            _releaseCount += 1
        }
    }

    public func clearEntry(_ hit: PrefixKVStoreHit) {
        withLock {
            guard let storage = hit.storage as? HitStorage else { return }
            if let session = slot(id: storage.slotID)?.sessionKey { disk?.invalidate(session: session) }
            removeSlot(id: storage.slotID)
            _clearCount += 1
        }
    }

    private func candidateSlots(sessionKey: String?) -> [Slot] {
        if let sessionKey, !sessionKey.isEmpty {
            return sessionSlots[sessionKey].map { [$0] } ?? []
        }
        return anonymousSlots
    }

    private func slot(id: UUID) -> Slot? {
        if let slot = sessionSlots.values.first(where: { $0.id == id }) {
            return slot
        }
        return anonymousSlots.first { $0.id == id }
    }

    private func removeSlot(id: UUID) {
        if let key = sessionSlots.first(where: { $0.value.id == id })?.key,
            let slot = sessionSlots.removeValue(forKey: key)
        {
            currentBytes -= slot.byteCount
            return
        }
        if let index = anonymousSlots.firstIndex(where: { $0.id == id }) {
            let slot = anonymousSlots.remove(at: index)
            currentBytes -= slot.byteCount
        }
    }

    /// Drops every unleased anonymous slot `newSlot` makes redundant, as a session slot is
    /// replaced by its own continuation.
    ///
    /// Without this a sessionless caller's every publish was a new full copy of its KV: the
    /// prompt, every 256 decoded tokens and the finish of each request, then each next turn
    /// again. Measured in the Cat (a328, M4 Pro 48 GB, Qwen3-0.6B, a 7,274-token history, no
    /// session header): two replies held 4,169 MB against 875 MB with the store off, and a
    /// 10-turn chat filled the 4 GB cap and evicted the slot its 10th turn needed.
    ///
    /// A slot is redundant when the new one holds its prompt, all but the last
    /// `promptTailTolerance` tokens, and is not a shorter prefix of it:
    /// - it is a prefix of the new slot (a later publish of the same request, or an exact retry);
    /// - or the two diverge only near or after the end of the old slot's prompt: the next turn of
    ///   the same chat, which re-renders the generation prompt and the reply (however long the
    ///   old reply was).
    /// Two chats that share a system prompt diverge earlier in the prompt, and both are kept.
    /// Whatever the new slot does not hold is passed on: the old slot's recurrent checkpoints up
    /// to the shared prefix, and its end state when it is a prefix of the new slot.
    private func supersedeAnonymousSlots(by newSlot: Slot) {
        for old in anonymousSlots where old.leaseCount == 0 {
            let shared = commonPrefixLength(old.tokens, newSlot.tokens)
            let oldPrompt = old.promptLength ?? old.tokens.count
            // A slot with recurrent or windowed layers resumes only at its end and its
            // checkpoints, so a split inside its prompt can lose its only resume point:
            // no tolerance there.
            let tolerance = old.hasRecurrentLayers ? 0 : Self.promptTailTolerance
            guard shared > 0, shared >= oldPrompt - tolerance else { continue }
            // The new slot is a prefix of the old one (a retry's first publish): the old one
            // still holds more. The retry's later publishes extend past it or diverge.
            if shared == newSlot.tokens.count, shared < old.tokens.count { continue }
            if shared == old.tokens.count {
                newSlot.promptLength = newSlot.promptLength ?? oldPrompt
            }
            for (position, layers) in old.checkpoints
            where position <= shared && position < newSlot.tokens.count && newSlot.checkpoints[position] == nil {
                newSlot.checkpoints[position] = layers
            }
            if old.hasRecurrentLayers, shared == old.tokens.count, shared < newSlot.tokens.count,
                newSlot.checkpoints[shared] == nil
            {
                newSlot.checkpoints[shared] = recurrentLayers(of: old.layers)
            }
            removeSlot(id: old.id)
            _supersededCount += 1
        }
    }

    private func evictIfNeeded() {
        while anonymousSlots.count > maxAnonymousSlots {
            guard evictOneSlot(anonymousOnly: true) else { break }
        }
        guard maxBytes > 0 else { return }
        while currentBytes > maxBytes {
            guard evictOneSlot(anonymousOnly: false) else { break }
        }
    }

    private func evictOneSlot(anonymousOnly: Bool) -> Bool {
        let candidates: [Slot]
        if anonymousOnly {
            candidates = anonymousSlots
        } else {
            candidates = anonymousSlots + sessionSlots.values
        }
        guard let victim = candidates
            .filter({ $0.leaseCount == 0 })
            .sorted(by: { $0.lastAccess < $1.lastAccess })
            .first
        else {
            return false
        }
        removeSlot(id: victim.id)
        _evictionCount += 1
        return true
    }

    /// The layers that cannot trim (recurrent state), index-aligned, `nil` elsewhere.
    private func recurrentLayers(of layers: [SerializedKVLayer]) -> [SerializedKVLayer?] {
        layers.map { layer in
            guard let cache = try? restoreLayer(layer), Scheduler.requiresExactPrefixBoundary(cache)
            else { return nil }
            return layer
        }
    }

    private func commonPrefixLength(_ lhs: [Int], _ rhs: [Int]) -> Int {
        var count = 0
        while count < lhs.count, count < rhs.count, lhs[count] == rhs[count] {
            count += 1
        }
        return count
    }

    private func nextSequence() -> UInt64 {
        sequence &+= 1
        return sequence
    }

    private func estimatedBytes(_ array: MLXArray) -> Int64 {
        let elements = array.shape.reduce(1) { $0 * max(1, $1) }
        return Int64(elements * byteWidth(array.dtype))
    }

    private func byteWidth(_ dtype: DType) -> Int {
        switch dtype {
        case .bool, .uint8, .int8:
            return 1
        case .bfloat16, .float16, .uint16, .int16:
            return 2
        case .float32, .int32, .uint32:
            return 4
        case .float64, .int64, .uint64:
            return 8
        default:
            return 4
        }
    }

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }
}
