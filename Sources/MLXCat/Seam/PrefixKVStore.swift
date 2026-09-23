import Foundation
import MLX
import MLXLMCommon

public struct SerializedKVLayer: @unchecked Sendable {
    public let state: [MLXArray]
    public let metaState: [String]
    public let className: String

    public init(state: [MLXArray], metaState: [String], className: String) {
        self.state = state
        self.metaState = metaState
        self.className = className
    }
}

public final class PrefixKVStoreHit: @unchecked Sendable {
    public let matchedTokenCount: Int
    public let blockCount: Int
    let storage: Any

    init(matchedTokenCount: Int, blockCount: Int, storage: Any) {
        self.matchedTokenCount = matchedTokenCount
        self.blockCount = blockCount
        self.storage = storage
    }
}

public protocol PrefixKVStore: AnyObject, Sendable {
    func fetch(tokens: [Int], sessionKey: String?) -> PrefixKVStoreHit?
    func preload(_ hit: PrefixKVStoreHit) throws
    func reconstructCache(from hit: PrefixKVStoreHit) throws -> [SerializedKVLayer]
    func store(tokens: [Int], sessionKey: String?, cache: [SerializedKVLayer]) throws
    /// A requirement, not only an extension method: the scheduler holds the
    /// store as `any PrefixKVStore`, and an extension-only method would dispatch
    /// statically to the default and silently drop the checkpoints.
    func store(
        tokens: [Int],
        sessionKey: String?,
        cache: [SerializedKVLayer],
        checkpoints: [PrefixRecurrentCheckpoint]
    ) throws
    func release(_ hit: PrefixKVStoreHit)
    func clearEntry(_ hit: PrefixKVStoreHit)
}

/// Recurrent-layer state captured at one prompt position during prefill.
///
/// A hybrid model (Qwen3.5/3.8: gated-delta-net layers beside attention) cannot
/// rewind its recurrent layers the way attention KV trims, so a stored slot is
/// only reusable at the exact position its recurrent state describes. A
/// checkpoint adds another such position: `layers` is index-aligned with the
/// slot's cache and holds the recurrent layers' state at `position` (attention
/// layers are `nil` — they trim from the slot's KV).
public struct PrefixRecurrentCheckpoint: @unchecked Sendable {
    public let position: Int
    public let layers: [SerializedKVLayer?]

    public init(position: Int, layers: [SerializedKVLayer?]) {
        self.position = position
        self.layers = layers
    }
}

public extension PrefixKVStore {
    /// Stores that cannot use recurrent checkpoints ignore them.
    func store(
        tokens: [Int],
        sessionKey: String?,
        cache: [SerializedKVLayer],
        checkpoints: [PrefixRecurrentCheckpoint]
    ) throws {
        try store(tokens: tokens, sessionKey: sessionKey, cache: cache)
    }

    func fetch(tokens: [Int]) -> PrefixKVStoreHit? {
        fetch(tokens: tokens, sessionKey: nil)
    }

    func store(tokens: [Int], cache: [SerializedKVLayer]) throws {
        try store(tokens: tokens, sessionKey: nil, cache: cache)
    }
}

public final class BlockAwarePrefixKVStore: PrefixKVStore, @unchecked Sendable {
    public let prefixCache: BlockAwarePrefixCache
    private let lock = NSRecursiveLock()
    private var _fetchHitCount = 0
    private var _storeCount = 0
    private var _releaseCount = 0
    private var _clearCount = 0

    public var fetchHitCount: Int {
        withLock { _fetchHitCount }
    }

    public var storeCount: Int {
        withLock { _storeCount }
    }

    public var releaseCount: Int {
        withLock { _releaseCount }
    }

    public var clearCount: Int {
        withLock { _clearCount }
    }

    public init(prefixCache: BlockAwarePrefixCache) {
        self.prefixCache = prefixCache
    }

    public func fetch(tokens: [Int], sessionKey: String?) -> PrefixKVStoreHit? {
        withLock {
            guard let hit = prefixCache.fetchCache(tokens: tokens) else { return nil }
            _fetchHitCount += 1
            return PrefixKVStoreHit(
                matchedTokenCount: hit.matchedTokenCount,
                blockCount: hit.blockCount,
                storage: hit
            )
        }
    }

    public func preload(_ hit: PrefixKVStoreHit) throws {
        try withLock {
            _ = try reconstructCache(from: hit)
        }
    }

    public func reconstructCache(from hit: PrefixKVStoreHit) throws -> [SerializedKVLayer] {
        try withLock {
            guard let rawHit = hit.storage as? PrefixCacheHit else {
                throw PrefixKVStoreError.invalidHit
            }

            return try prefixCache.reconstructCache(from: rawHit).map { layerCache in
                SerializedKVLayer(
                    state: layerCache.state,
                    metaState: layerCache.metaState,
                    className: "KVCacheSimple"
                )
            }
        }
    }

    public func store(tokens: [Int], sessionKey: String?, cache: [SerializedKVLayer]) throws {
        try withLock {
            let layerCaches = try cache.map { layer in
                try Self.cache(from: layer)
            }
            try prefixCache.storeCache(tokens: tokens, cache: layerCaches)
            _storeCount += 1
        }
    }

    public func release(_ hit: PrefixKVStoreHit) {
        withLock {
            guard let rawHit = hit.storage as? PrefixCacheHit else { return }
            prefixCache.release(rawHit)
            _releaseCount += 1
        }
    }

    public func clearEntry(_ hit: PrefixKVStoreHit) {
        withLock {
            release(hit)
            _clearCount += 1
        }
    }

    public static func cache(from layer: SerializedKVLayer) throws -> any KVCache {
        switch layer.className {
        case "MambaCache":
            guard layer.state.count <= 2 else {
                throw PrefixKVStoreError.unsupportedLayerState
            }
            let cache = MambaCache(leftPadding: metadataValues(layer.metaState, at: 2))
            cache.state = layer.state
            cache.prepare(lengths: metadataValues(layer.metaState, at: 3))
            return cache
        case "KVCache", "KVCacheSimple":
            guard layer.state.count == 2 else {
                throw PrefixKVStoreError.unsupportedLayerState
            }
            let cache = KVCacheSimple()
            cache.state = layer.state
            return cache
        default:
            guard layer.state.count == 2, layer.metaState.isEmpty else {
                throw PrefixKVStoreError.unsupportedCacheClass(layer.className)
            }
            let cache = KVCacheSimple()
            cache.state = layer.state
            return cache
        }
    }

    private static func metadataValues(_ metaState: [String], at index: Int) -> [Int]? {
        guard metaState.indices.contains(index), !metaState[index].isEmpty else { return nil }
        return metaState[index].split(separator: ",").compactMap { Int($0) }
    }

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }
}

public enum PrefixKVStoreError: Error, Equatable {
    case invalidHit
    case unsupportedLayerState
    case unsupportedCacheClass(String)
    case unsupportedCacheTrim(String)
}
