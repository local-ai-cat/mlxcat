import CryptoKit
import Foundation

public enum BlockHashing {
    public static let formatVersion: UInt8 = 1

    public static func computeBlockHash(
        modelName: String,
        previousHash: Data?,
        tokens: ArraySlice<Int>,
        blockSize: Int
    ) -> Data {
        var canonical = Data()
        canonical.append(formatVersion)
        appendString("mlxserve-prefix-cache", to: &canonical)
        appendString(modelName, to: &canonical)
        appendUInt32(UInt32(blockSize), to: &canonical)

        if let previousHash {
            appendUInt32(UInt32(previousHash.count), to: &canonical)
            canonical.append(previousHash)
        } else {
            appendUInt32(0, to: &canonical)
        }

        appendUInt32(UInt32(tokens.count), to: &canonical)
        for token in tokens {
            appendInt64(Int64(token), to: &canonical)
        }

        return Data(SHA256.hash(data: canonical))
    }

    public static func chainHashes(
        modelName: String,
        tokens: [Int],
        blockSize: Int
    ) -> [Data] {
        let fullBlockCount = tokens.count / blockSize
        var previousHash: Data?
        var hashes: [Data] = []

        for blockIndex in 0 ..< fullBlockCount {
            let start = blockIndex * blockSize
            let end = start + blockSize
            let hash = computeBlockHash(
                modelName: modelName,
                previousHash: previousHash,
                tokens: tokens[start ..< end],
                blockSize: blockSize
            )
            hashes.append(hash)
            previousHash = hash
        }

        return hashes
    }

    public static func hex(_ hash: Data) -> String {
        hash.map { String(format: "%02x", $0) }.joined()
    }

    private static func appendString(_ string: String, to data: inout Data) {
        let bytes = Array(string.utf8)
        appendUInt32(UInt32(bytes.count), to: &data)
        data.append(contentsOf: bytes)
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendInt64(_ value: Int64, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
}
