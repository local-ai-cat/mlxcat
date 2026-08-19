import Foundation
import MLX
import MLXLMCommon

public struct SessionPrefixKVStoreStats: Equatable, Sendable {
    public let fetchHitCount: Int
    public let storeCount: Int
    public let releaseCount: Int
    public let clearCount: Int
    public let evictionCount: Int
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
    }

    private let lock = NSRecursiveLock()
    private let maxBytes: Int64
    private let maxAnonymousSlots: Int
    private var sessionSlots: [String: Slot] = [:]
    private var anonymousSlots: [Slot] = []
    private var sequence: UInt64 = 0
    private var currentBytes: Int64 = 0
    private var _fetchHitCount = 0
    private var _storeCount = 0
    private var _releaseCount = 0
    private var _clearCount = 0
    private var _evictionCount = 0

    public init(maxBytes: Int64 = 0, maxAnonymousSlots: Int = 8) {
        self.maxBytes = max(0, maxBytes)
        self.maxAnonymousSlots = max(1, maxAnonymousSlots)
    }

    public var stats: SessionPrefixKVStoreStats {
        withLock {
            SessionPrefixKVStoreStats(
                fetchHitCount: _fetchHitCount,
                storeCount: _storeCount,
                releaseCount: _releaseCount,
                clearCount: _clearCount,
                evictionCount: _evictionCount,
                currentBytes: currentBytes,
                slotCount: sessionSlots.count + anonymousSlots.count
            )
        }
    }

    public func fetch(tokens: [Int], sessionKey: String?) -> PrefixKVStoreHit? {
        withLock {
            guard tokens.count > 1 else { return nil }
            let candidates = candidateSlots(sessionKey: sessionKey)
                .filter { $0.leaseCount == 0 }
                .compactMap { slot -> (slot: Slot, matched: Int)? in
                    let matched = commonPrefixLength(slot.tokens, tokens)
                    return matched > 0 ? (slot, matched) : nil
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
                    trimCount: max(0, best.slot.tokens.count - best.matched)
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
            return try slot.layers.map { layer in
                let cache = try BlockAwarePrefixKVStore.cache(from: layer)
                if storage.trimCount > 0 {
                    guard cache.trim(storage.trimCount) == storage.trimCount else {
                        throw PrefixKVStoreError.unsupportedCacheTrim(layer.className)
                    }
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
        try withLock {
            guard tokens.count > 1, !cache.isEmpty else { return }
            let copied = try cache.map { layer -> SerializedKVLayer in
                _ = try BlockAwarePrefixKVStore.cache(from: layer)
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
            if let sessionKey, !sessionKey.isEmpty {
                if let old = sessionSlots[sessionKey] {
                    guard old.leaseCount == 0 else {
                        return
                    }
                    currentBytes -= old.byteCount
                }
                sessionSlots[sessionKey] = slot
            } else {
                anonymousSlots.append(slot)
            }
            currentBytes += byteCount
            _storeCount += 1
            evictIfNeeded()
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
