public struct ModelKVCacheProfile: Sendable, Equatable {
    public let hiddenLayerCount: Int
    public let keyValueHeadCount: Int
    public let headDimension: Int
    public let scalarByteCount: Int

    public init?(
        hiddenLayerCount: Int?,
        attentionHeadCount: Int?,
        keyValueHeadCount: Int?,
        headDimension: Int?,
        hiddenSize: Int?,
        scalarByteCount: Int = 2
    ) {
        guard let hiddenLayerCount, hiddenLayerCount > 0 else { return nil }
        guard let attentionHeadCount, attentionHeadCount > 0 else { return nil }
        let resolvedKeyValueHeadCount = keyValueHeadCount ?? attentionHeadCount
        guard resolvedKeyValueHeadCount > 0 else { return nil }

        let resolvedHeadDimension: Int?
        if let headDimension {
            resolvedHeadDimension = headDimension
        } else if let hiddenSize, hiddenSize > 0, hiddenSize.isMultiple(of: attentionHeadCount) {
            resolvedHeadDimension = hiddenSize / attentionHeadCount
        } else {
            resolvedHeadDimension = nil
        }
        guard let resolvedHeadDimension, resolvedHeadDimension > 0 else { return nil }
        guard scalarByteCount > 0 else { return nil }

        self.hiddenLayerCount = hiddenLayerCount
        self.keyValueHeadCount = resolvedKeyValueHeadCount
        self.headDimension = resolvedHeadDimension
        self.scalarByteCount = scalarByteCount
    }

    public var bytesPerToken: Int64 {
        Int64(hiddenLayerCount)
            * 2
            * Int64(keyValueHeadCount)
            * Int64(headDimension)
            * Int64(scalarByteCount)
    }

    public func estimatedBytes(promptTokens: Int, maxGeneratedTokens: Int) -> Int64 {
        let tokenCount = max(0, promptTokens) + max(0, maxGeneratedTokens)
        return Int64(tokenCount) * bytesPerToken
    }

    public static func scalarByteCount(forDType dtype: String?) -> Int {
        guard let dtype = dtype?.lowercased() else { return 2 }
        if dtype.contains("64") { return 8 }
        if dtype.contains("32") { return 4 }
        if dtype.contains("16") || dtype.contains("half") { return 2 }
        return 2
    }
}

public struct ModelCacheCapabilities: Sendable, Equatable {
    public let usesWindowedKVCache: Bool
    public let kvCacheProfile: ModelKVCacheProfile?

    public init(
        usesWindowedKVCache: Bool = false,
        kvCacheProfile: ModelKVCacheProfile? = nil
    ) {
        self.usesWindowedKVCache = usesWindowedKVCache
        self.kvCacheProfile = kvCacheProfile
    }

    public static let `default` = ModelCacheCapabilities()
}
