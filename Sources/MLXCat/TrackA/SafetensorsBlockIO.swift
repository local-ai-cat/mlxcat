import Foundation
import MLX

public struct SafetensorsTensorSnapshot: Sendable {
    public let name: String
    public let dtype: DType
    public let shape: [Int]
    public let data: Data

    public init(name: String, dtype: DType, shape: [Int], data: Data) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.data = data
    }
}

public struct SafetensorsBlockFile: Sendable {
    public let tensors: [SafetensorsTensorSnapshot]
    public let metadata: [String: String]

    public init(tensors: [SafetensorsTensorSnapshot], metadata: [String: String]) {
        self.tensors = tensors
        self.metadata = metadata
    }
}

public struct SafetensorsLoadedBlock {
    public let arrays: [String: MLXArray]
    public let metadata: [String: String]
    public let rawTensorBytes: [String: Data]
}

public enum SafetensorsBlockIO {
    public static let formatVersion = "4"
    private static let maxHeaderBytes = 16 * 1024 * 1024

    public static func snapshot(
        hash: Data,
        payload: KVCacheBlockPayload,
        tokenCount: Int,
        modelName: String,
        blockSize: Int
    ) -> SafetensorsBlockFile {
        var tensors: [SafetensorsTensorSnapshot] = []
        for (layerIndex, layer) in payload.layers.enumerated() {
            tensors.append(snapshotTensor(name: "layer.\(layerIndex).keys", array: layer.keys))
            tensors.append(snapshotTensor(name: "layer.\(layerIndex).values", array: layer.values))
        }

        return SafetensorsBlockFile(
            tensors: tensors,
            metadata: [
                "formatVersion": formatVersion,
                "blockHash": BlockHashing.hex(hash),
                "tokenCount": String(tokenCount),
                "numLayers": String(payload.layers.count),
                "modelName": modelName,
                "blockSize": String(blockSize),
                "isRotating": CacheTypeHandlers.encodeBool(false),
            ]
        )
    }

    public static func write(_ blockFile: SafetensorsBlockFile, to url: URL) throws {
        let encoded = try encode(blockFile)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let tmpURL = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        try encoded.write(to: tmpURL, options: .atomic)
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmpURL)
        } else {
            try FileManager.default.moveItem(at: tmpURL, to: url)
        }
    }

    public static func read(from url: URL) throws -> SafetensorsLoadedBlock {
        let data = try Data(contentsOf: url)
        return try decode(data)
    }

    public static func readMetadata(from url: URL) throws -> [String: String] {
        let data = try Data(contentsOf: url)
        let header = try parseHeader(data)
        return header.metadata
    }

    private static func snapshotTensor(name: String, array: MLXArray) -> SafetensorsTensorSnapshot {
        let dtype = array.dtype
        let dataArray = dtype == .bfloat16 ? array.view(dtype: .uint16) : array
        eval(dataArray)
        let bytes = dataArray.asData(access: .copy).data
        return SafetensorsTensorSnapshot(
            name: name,
            dtype: dtype,
            shape: array.shape,
            data: bytes
        )
    }

    private static func encode(_ blockFile: SafetensorsBlockFile) throws -> Data {
        let tensors = blockFile.tensors.sorted { $0.name < $1.name }
        var header: [String: Any] = [
            "__metadata__": blockFile.metadata
        ]
        var offset = 0
        for tensor in tensors {
            let end = offset + tensor.data.count
            header[tensor.name] = [
                "dtype": safetensorsDType(tensor.dtype),
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            ]
            offset = end
        }

        var headerBytes = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys]
        )
        let padding = (8 - ((8 + headerBytes.count) % 8)) % 8
        if padding > 0 {
            headerBytes.append(contentsOf: repeatElement(UInt8(ascii: " "), count: padding))
        }

        var result = Data()
        appendUInt64(UInt64(headerBytes.count), to: &result)
        result.append(headerBytes)
        for tensor in tensors {
            result.append(tensor.data)
        }
        return result
    }

    private static func decode(_ data: Data) throws -> SafetensorsLoadedBlock {
        let parsed = try parseHeader(data)
        var arrays: [String: MLXArray] = [:]
        var rawTensorBytes: [String: Data] = [:]

        for tensor in parsed.tensors {
            let absoluteStart = parsed.dataStart + tensor.start
            let absoluteEnd = parsed.dataStart + tensor.end
            guard absoluteStart <= absoluteEnd, absoluteEnd <= data.count else {
                throw SafetensorsBlockIOError.invalidTensorOffsets(tensor.name)
            }

            let bytes = data[absoluteStart ..< absoluteEnd]
            let tensorData = Data(bytes)
            let expectedBytes = try expectedByteCount(shape: tensor.shape, dtype: tensor.dtype)
            guard tensorData.count == expectedBytes else {
                throw SafetensorsBlockIOError.invalidTensorByteCount(
                    tensor.name,
                    expected: expectedBytes,
                    actual: tensorData.count
                )
            }
            rawTensorBytes[tensor.name] = tensorData

            let array: MLXArray
            if tensor.dtype == .bfloat16 {
                array = MLXArray(tensorData, tensor.shape, dtype: .uint16).view(dtype: .bfloat16)
            } else {
                array = MLXArray(tensorData, tensor.shape, dtype: tensor.dtype)
            }
            arrays[tensor.name] = array
        }

        return SafetensorsLoadedBlock(
            arrays: arrays,
            metadata: parsed.metadata,
            rawTensorBytes: rawTensorBytes
        )
    }

    private static func parseHeader(_ data: Data) throws -> ParsedHeader {
        guard data.count >= 8 else {
            throw SafetensorsBlockIOError.invalidHeader
        }

        let headerLength = data.prefix(8).withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        guard headerLength <= UInt64(maxHeaderBytes),
            headerLength <= UInt64(Int.max - 8)
        else {
            throw SafetensorsBlockIOError.invalidHeader
        }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerLength)
        guard headerEnd <= data.count else {
            throw SafetensorsBlockIOError.invalidHeader
        }

        let headerData = data[headerStart ..< headerEnd]
        guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw SafetensorsBlockIOError.invalidHeader
        }

        let metadata = header["__metadata__"] as? [String: String] ?? [:]
        var tensors: [ParsedTensor] = []
        for (name, value) in header where name != "__metadata__" {
            guard let tensor = value as? [String: Any],
                let dtypeString = tensor["dtype"] as? String,
                let dtype = dtype(from: dtypeString),
                let shape = tensor["shape"] as? [Int],
                let offsets = tensor["data_offsets"] as? [Int],
                offsets.count == 2,
                shape.allSatisfy({ $0 >= 0 }),
                offsets[0] >= 0,
                offsets[1] >= offsets[0]
            else {
                throw SafetensorsBlockIOError.invalidTensorHeader(name)
            }

            tensors.append(
                ParsedTensor(
                    name: name,
                    dtype: dtype,
                    shape: shape,
                    start: offsets[0],
                    end: offsets[1]
                )
            )
        }

        return ParsedHeader(
            metadata: metadata,
            tensors: tensors,
            dataStart: headerEnd
        )
    }

    private static func safetensorsDType(_ dtype: DType) -> String {
        switch dtype {
        case .bool: "BOOL"
        case .uint8: "U8"
        case .uint16: "U16"
        case .uint32: "U32"
        case .uint64: "U64"
        case .int8: "I8"
        case .int16: "I16"
        case .int32: "I32"
        case .int64: "I64"
        case .float16: "F16"
        case .float32: "F32"
        case .bfloat16: "BF16"
        case .float64: "F64"
        case .complex64: "C64"
        }
    }

    private static func dtype(from value: String) -> DType? {
        switch value {
        case "BOOL": .bool
        case "U8": .uint8
        case "U16": .uint16
        case "U32": .uint32
        case "U64": .uint64
        case "I8": .int8
        case "I16": .int16
        case "I32": .int32
        case "I64": .int64
        case "F16": .float16
        case "F32": .float32
        case "BF16": .bfloat16
        case "F64": .float64
        case "C64": .complex64
        default: nil
        }
    }

    private static func expectedByteCount(shape: [Int], dtype: DType) throws -> Int {
        let elementCount = try shape.reduce(1) { partial, dimension in
            let multiplied = partial.multipliedReportingOverflow(by: dimension)
            guard !multiplied.overflow else {
                throw SafetensorsBlockIOError.invalidHeader
            }
            return multiplied.partialValue
        }
        let multiplied = elementCount.multipliedReportingOverflow(by: byteWidth(dtype))
        guard !multiplied.overflow else {
            throw SafetensorsBlockIOError.invalidHeader
        }
        return multiplied.partialValue
    }

    private static func byteWidth(_ dtype: DType) -> Int {
        switch dtype {
        case .bool, .uint8, .int8:
            return 1
        case .uint16, .int16, .float16, .bfloat16:
            return 2
        case .uint32, .int32, .float32:
            return 4
        case .uint64, .int64, .float64, .complex64:
            return 8
        }
    }

    private static func appendUInt64(_ value: UInt64, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
}

public enum SafetensorsBlockIOError: Error, Equatable {
    case invalidHeader
    case invalidTensorHeader(String)
    case invalidTensorOffsets(String)
    case invalidTensorByteCount(String, expected: Int, actual: Int)
    case missingTensor(String)
    case incompatibleMetadata(String)
}

private struct ParsedHeader {
    let metadata: [String: String]
    let tensors: [ParsedTensor]
    let dataStart: Int
}

private struct ParsedTensor {
    let name: String
    let dtype: DType
    let shape: [Int]
    let start: Int
    let end: Int
}
