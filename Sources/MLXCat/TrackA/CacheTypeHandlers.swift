import MLX

public struct CacheLayerBlockPayload: @unchecked Sendable {
    public let keys: MLXArray
    public let values: MLXArray
    public let className: String
    public let metaState: [String]

    public init(
        keys: MLXArray,
        values: MLXArray,
        className: String = "KVCacheSimple",
        metaState: [String]
    ) {
        self.keys = keys
        self.values = values
        self.className = className
        self.metaState = metaState
    }
}

public struct KVCacheBlockPayload: @unchecked Sendable {
    public let layers: [CacheLayerBlockPayload]

    public init(layers: [CacheLayerBlockPayload]) {
        self.layers = layers
    }
}

public enum CacheTypeHandlers {
    public static func encodeBool(_ value: Bool) -> String {
        value ? "1" : "0"
    }

    public static func decodeBool(_ value: String) -> Bool? {
        switch value {
        case "1":
            return true
        case "0":
            return false
        default:
            return nil
        }
    }
}
