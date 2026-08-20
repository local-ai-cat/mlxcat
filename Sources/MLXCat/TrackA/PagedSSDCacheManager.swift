import Foundation

public final class PagedSSDCacheManager: PagedCacheManager, @unchecked Sendable {
    public let cacheDirectory: URL
    public let modelName: String
    public let maxHotBlocks: Int

    private let writer = SSDWriter()
    private let ssdLock = NSRecursiveLock()
    private var pendingWrites: [Data: Task<Void, Error>] = [:]
    private var hotLRU: [Data] = []
    private var knownSSDHashes: Set<Data> = []

    public init(
        cacheDirectory: URL,
        modelName: String,
        blockSize: Int = 256,
        maxHotBlocks: Int = 1,
        maxStoredBlocks: Int = 4096
    ) throws {
        self.cacheDirectory = cacheDirectory
        self.modelName = modelName
        self.maxHotBlocks = maxHotBlocks
        super.init(blockSize: blockSize, maxStoredBlocks: maxStoredBlocks)
        try scanOnStart()
    }

    public override func storeBlock(
        hash: Data,
        tokenCount: Int,
        payload: KVCacheBlockPayload
    ) -> Int {
        let blockID = super.storeBlock(hash: hash, tokenCount: tokenCount, payload: payload)
        markHot(hash)
        enqueueWrite(hash: hash, tokenCount: tokenCount, payload: payload)
        enforceHotLimit(protectedHash: hash)
        return blockID
    }

    public override func payload(for hash: Data) -> KVCacheBlockPayload? {
        if let payload = super.payload(for: hash) {
            markHot(hash)
            return payload
        }

        guard isKnownOnSSD(hash) else {
            return nil
        }
        let loaded: KVCacheBlockPayload
        do {
            loaded = try loadPayload(hash: hash)
        } catch {
            logSSDCacheFailure("failed to load SSD payload for \(BlockHashing.hex(hash))", error)
            return nil
        }
        setHotPayload(loaded, for: hash)
        markHot(hash)
        enforceHotLimit(protectedHash: nil)
        return loaded
    }

    public override func payload(for blockID: Int) -> KVCacheBlockPayload? {
        guard let hash = blockHash(for: blockID) else { return nil }
        return payload(for: hash)
    }

    public func flushPendingWrites() async throws {
        let writes = withSSDLock {
            Array(pendingWrites.values)
        }
        for write in writes {
            try await write.value
        }
    }

    public func fileURL(for hash: Data) -> URL {
        let hex = BlockHashing.hex(hash)
        let subdirectory = String(hex.prefix(1))
        return cacheDirectory
            .appendingPathComponent(subdirectory, isDirectory: true)
            .appendingPathComponent("\(hex).safetensors")
    }

    private func enqueueWrite(hash: Data, tokenCount: Int, payload: KVCacheBlockPayload) {
        let snapshot = SafetensorsBlockIO.snapshot(
            hash: hash,
            payload: payload,
            tokenCount: tokenCount,
            modelName: modelName,
            blockSize: blockSize
        )
        let destination = fileURL(for: hash)
        let write = Task { [weak self, writer] in
            do {
                try await writer.write(snapshot, to: destination)
                self?.completeWrite(hash: hash, succeeded: true)
            } catch {
                self?.completeWrite(hash: hash, succeeded: false)
                throw error
            }
        }
        withSSDLock {
            pendingWrites[hash] = write
        }
    }

    private func loadPayload(hash: Data) throws -> KVCacheBlockPayload {
        let loaded = try SafetensorsBlockIO.read(from: fileURL(for: hash))
        try validate(metadata: loaded.metadata, expectedHash: hash)

        guard let layerCount = Int(loaded.metadata["numLayers"] ?? "") else {
            throw SafetensorsBlockIOError.incompatibleMetadata("numLayers")
        }

        var layers: [CacheLayerBlockPayload] = []
        for layerIndex in 0 ..< layerCount {
            let keysName = "layer.\(layerIndex).keys"
            let valuesName = "layer.\(layerIndex).values"
            guard let keys = loaded.arrays[keysName] else {
                throw SafetensorsBlockIOError.missingTensor(keysName)
            }
            guard let values = loaded.arrays[valuesName] else {
                throw SafetensorsBlockIOError.missingTensor(valuesName)
            }
            layers.append(
                CacheLayerBlockPayload(
                    keys: keys,
                    values: values,
                    metaState: [
                        loaded.metadata["tokenCount"] ?? String(blockSize),
                        CacheTypeHandlers.encodeBool(false),
                    ]
                )
            )
        }
        return KVCacheBlockPayload(layers: layers)
    }

    private func scanOnStart() throws {
        try FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        guard let enumerator = FileManager.default.enumerator(
            at: cacheDirectory,
            includingPropertiesForKeys: nil
        ) else {
            return
        }

        for case let fileURL as URL in enumerator where fileURL.pathExtension == "safetensors" {
            do {
                let metadata = try SafetensorsBlockIO.readMetadata(from: fileURL)
                guard let hashHex = metadata["blockHash"],
                    let hash = Data(hexString: hashHex)
                else {
                    continue
                }
                try validate(metadata: metadata, expectedHash: hash)
                let tokenCount = Int(metadata["tokenCount"] ?? "") ?? blockSize
                registerBlockMetadata(hash: hash, tokenCount: tokenCount)
                _ = withSSDLock {
                    knownSSDHashes.insert(hash)
                }
            } catch {
                logSSDCacheFailure("ignored invalid SSD cache file \(fileURL.path)", error)
                continue
            }
        }
    }

    private func validate(metadata: [String: String], expectedHash: Data) throws {
        guard metadata["formatVersion"] == SafetensorsBlockIO.formatVersion else {
            throw SafetensorsBlockIOError.incompatibleMetadata("formatVersion")
        }
        guard metadata["modelName"] == modelName else {
            throw SafetensorsBlockIOError.incompatibleMetadata("modelName")
        }
        guard metadata["blockSize"] == String(blockSize) else {
            throw SafetensorsBlockIOError.incompatibleMetadata("blockSize")
        }
        guard metadata["blockHash"] == BlockHashing.hex(expectedHash) else {
            throw SafetensorsBlockIOError.incompatibleMetadata("blockHash")
        }
        guard CacheTypeHandlers.decodeBool(metadata["isRotating"] ?? "") == false else {
            throw SafetensorsBlockIOError.incompatibleMetadata("isRotating")
        }
    }

    private func markHot(_ hash: Data) {
        withSSDLock {
            hotLRU.removeAll { $0 == hash }
            hotLRU.append(hash)
        }
    }

    private func enforceHotLimit(protectedHash: Data?) {
        while hotPayloadCount > maxHotBlocks {
            let victim = withSSDLock {
                hotLRU.first { hash in
                    hash != protectedHash && knownSSDHashes.contains(hash)
                }
            }
            guard let victim else { return }
            withSSDLock {
                hotLRU.removeAll { $0 == victim }
            }
            removeHotPayload(for: victim)
        }
    }

    private func isKnownOnSSD(_ hash: Data) -> Bool {
        withSSDLock {
            knownSSDHashes.contains(hash)
        }
    }

    private func completeWrite(hash: Data, succeeded: Bool) {
        withSSDLock {
            pendingWrites.removeValue(forKey: hash)
            if succeeded {
                knownSSDHashes.insert(hash)
            }
        }
        if succeeded {
            enforceCapacityLimit()
            enforceHotLimit(protectedHash: nil)
        }
    }

    override func canEvictBlock(hash: Data) -> Bool {
        isKnownOnSSD(hash)
    }

    override func didEvictBlock(hash: Data) {
        withSSDLock {
            hotLRU.removeAll { $0 == hash }
            knownSSDHashes.remove(hash)
            pendingWrites.removeValue(forKey: hash)?.cancel()
        }
    }

    private func withSSDLock<T>(_ body: () throws -> T) rethrows -> T {
        ssdLock.lock()
        defer { ssdLock.unlock() }
        return try body()
    }

    private func logSSDCacheFailure(_ message: String, _ error: Error) {
        let line = "MLXCat SSD cache warning: \(message): \(error)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}

private actor SSDWriter {
    func write(_ snapshot: SafetensorsBlockFile, to url: URL) throws {
        try SafetensorsBlockIO.write(snapshot, to: url)
    }
}

private extension Data {
    init?(hexString: String) {
        guard hexString.count.isMultiple(of: 2) else { return nil }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(hexString.count / 2)
        var index = hexString.startIndex
        while index < hexString.endIndex {
            let next = hexString.index(index, offsetBy: 2)
            guard let byte = UInt8(hexString[index ..< next], radix: 16) else {
                return nil
            }
            bytes.append(byte)
            index = next
        }
        self = Data(bytes)
    }
}
