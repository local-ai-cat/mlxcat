import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXServe
import Tokenizers
import XCTest

private struct PrefixCacheGateResult: Sendable {
    let maxLogitError: Float
    let checkedTokens: Int
    let mismatches: Int
    let sharedBlockCount: Int
    let allocatedBaseline: Int
    let allocatedAfterRequests: Int
    let totalRefBaseline: Int
    let totalRefAfterRequests: Int
}

private struct SSDCacheGateResult: Sendable {
    let byteRoundTripExact: Bool
    let maxLogitError: Float
    let checkedTokens: Int
    let mismatches: Int
    let scannedBlocks: Int
    let hotPayloadsAfterScan: Int
}

final class TrackAPrefixCacheTests: XCTestCase {
    func testPagedCacheManagerEvictsUnusedBlocksAtCapacity() throws {
        let manager = PagedCacheManager(blockSize: 1, maxStoredBlocks: 2)
        let hashes = (0 ..< 3).map { Data(repeating: UInt8($0), count: 32) }

        for hash in hashes {
            manager.storeBlock(hash: hash, tokenCount: 1, payload: Self.unitPayload())
        }

        XCTAssertEqual(manager.allocatedBlocks, 2)
        XCTAssertFalse(manager.contains(hash: hashes[0]))
        XCTAssertTrue(manager.contains(hash: hashes[1]))
        XCTAssertTrue(manager.contains(hash: hashes[2]))
    }

    func testPagedCacheManagerKeepsRetainedBlocksUntilReleased() throws {
        let manager = PagedCacheManager(blockSize: 1, maxStoredBlocks: 1)
        let retainedHash = Data(repeating: 0xA1, count: 32)
        let newerHash = Data(repeating: 0xB2, count: 32)

        let retainedBlockID = manager.storeBlock(
            hash: retainedHash,
            tokenCount: 1,
            payload: Self.unitPayload()
        )
        manager.retain(blockID: retainedBlockID)
        manager.storeBlock(hash: newerHash, tokenCount: 1, payload: Self.unitPayload())

        XCTAssertEqual(manager.allocatedBlocks, 2)
        XCTAssertTrue(manager.contains(hash: retainedHash))
        XCTAssertTrue(manager.contains(hash: newerHash))

        manager.release(BlockTable(blockIDs: [retainedBlockID]))

        XCTAssertEqual(manager.allocatedBlocks, 1)
    }

    func testReconstructThrowsWhenRetainedBlockPayloadIsMissing() throws {
        let prefixCache = BlockAwarePrefixCache(modelName: "unit-test", blockSize: 2)
        let tokens = [1, 2]
        let hash = try XCTUnwrap(BlockHashing.chainHashes(
            modelName: "unit-test",
            tokens: tokens,
            blockSize: 2
        ).first)
        let blockID = prefixCache.manager.registerBlockMetadata(hash: hash, tokenCount: 2)
        let hit = try XCTUnwrap(prefixCache.fetchCache(tokens: tokens))
        defer { prefixCache.release(hit) }

        XCTAssertThrowsError(try prefixCache.reconstructCache(from: hit)) { error in
            XCTAssertEqual(
                error as? PrefixCacheError,
                .missingBlockPayload(blockID: blockID, hash: hash)
            )
        }
    }

    func testStoreCacheThrowsForUnsupportedFixedStateCache() throws {
        let prefixCache = BlockAwarePrefixCache(modelName: "unit-test", blockSize: 2)
        let fixedStateCache = UnsupportedPrefixTestCache(
            state: [
                MLXArray.zeros([1, 3, 16], dtype: .bfloat16),
                MLXArray.zeros([1, 4, 8, 8], dtype: .float32),
            ]
        )

        XCTAssertThrowsError(
            try prefixCache.storeCache(tokens: [1, 2], cache: [fixedStateCache])
        ) { error in
            guard case .unsupportedCacheLayout = error as? PrefixCacheError else {
                XCTFail("expected unsupported cache layout, got \(error)")
                return
            }
        }
    }

    func testSafetensorsReadThrowsOnTruncatedTensorData() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlxserve-truncated-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let hash = Data(repeating: 0xCC, count: 32)
        let array = MLXArray.zeros([1, 1, 2, 2], dtype: .float32)
        let snapshot = SafetensorsBlockIO.snapshot(
            hash: hash,
            payload: KVCacheBlockPayload(
                layers: [
                    CacheLayerBlockPayload(
                        keys: array,
                        values: array,
                        metaState: ["2", CacheTypeHandlers.encodeBool(false)]
                    )
                ]
            ),
            tokenCount: 2,
            modelName: "truncated-test",
            blockSize: 2
        )
        let url = directory.appendingPathComponent("truncated.safetensors")
        try SafetensorsBlockIO.write(snapshot, to: url)
        var data = try Data(contentsOf: url)
        data.removeLast()
        try data.write(to: url)

        XCTAssertThrowsError(try SafetensorsBlockIO.read(from: url)) { error in
            guard case .invalidTensorOffsets? = error as? SafetensorsBlockIOError else {
                XCTFail("expected invalid tensor offsets, got \(error)")
                return
            }
        }
    }

    func testHotPrefixCacheReconstructsAndDoesNotLeakBlocks() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M3 prefix-cache gate.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M3 prefix-cache fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let result = try await container.perform { context in
            try await Self.evaluatePrefixCacheGate(context: context)
        }

        if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_PREFIX_GATE"] == "1" {
            print(
                "M3 prefix: maxLogitError=\(result.maxLogitError), checkedTokens=\(result.checkedTokens), mismatches=\(result.mismatches), sharedBlocks=\(result.sharedBlockCount), allocatedBaseline=\(result.allocatedBaseline), allocatedAfter=\(result.allocatedAfterRequests), totalRefBaseline=\(result.totalRefBaseline), totalRefAfter=\(result.totalRefAfterRequests)"
            )
        }

        XCTAssertLessThan(result.maxLogitError, 1.25)
        XCTAssertGreaterThan(result.checkedTokens, 0)
        XCTAssertEqual(result.mismatches, 0)
        XCTAssertEqual(result.sharedBlockCount, 1)
        XCTAssertEqual(result.allocatedAfterRequests, result.allocatedBaseline)
        XCTAssertEqual(result.totalRefAfterRequests, result.totalRefBaseline)
    }

    func testSSDColdTierRestartsAndRoundTripsBytesExactly() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M4 SSD prefix-cache gate.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M4 SSD prefix-cache fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let cacheDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlxserve-m4-\(UUID().uuidString)", isDirectory: true)
        let byteDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlxserve-m4-bytes-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: cacheDirectory) }
        defer { try? FileManager.default.removeItem(at: byteDirectory) }

        let byteRoundTripExact = try Self.exactBF16RoundTrip(in: byteDirectory)

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let result = try await container.perform { context in
            try await Self.evaluateSSDCacheGate(
                context: context,
                cacheDirectory: cacheDirectory,
                byteRoundTripExact: byteRoundTripExact
            )
        }

        if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_PREFIX_GATE"] == "1" {
            print(
                "M4 SSD: byteExact=\(result.byteRoundTripExact), maxLogitError=\(result.maxLogitError), checkedTokens=\(result.checkedTokens), mismatches=\(result.mismatches), scannedBlocks=\(result.scannedBlocks), hotPayloadsAfterScan=\(result.hotPayloadsAfterScan)"
            )
        }

        XCTAssertTrue(result.byteRoundTripExact)
        XCTAssertLessThan(result.maxLogitError, 1.25)
        XCTAssertGreaterThan(result.checkedTokens, 0)
        XCTAssertEqual(result.mismatches, 0)
        XCTAssertEqual(result.scannedBlocks, 1)
        XCTAssertEqual(result.hotPayloadsAfterScan, 0)
    }

    private static func evaluatePrefixCacheGate(
        context: ModelContext
    ) async throws -> PrefixCacheGateResult {
        let model = context.model
        let blockSize = 256
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        let seedText = String(
            repeating: "The capital of France is Paris. Swift concurrency protects shared state. GPU kernels execute matrix operations quickly. ",
            count: 128
        )
        let seedTokens = try await tokenIDs(for: seedText, context: context, parameters: parameters)
        var prefixTokens: [Int] = []
        while prefixTokens.count < blockSize {
            prefixTokens.append(contentsOf: seedTokens)
        }
        prefixTokens = Array(prefixTokens.prefix(blockSize))
        let candidateSuffixes = try await [
            " The answer is",
            " Therefore",
            " In summary",
            " Paris",
            " Swift",
            " The next",
            " GPU",
            " because",
            " once",
            " list",
        ].mapAsync { text in
            try await tokenIDs(for: text, context: context, parameters: parameters)
        }

        let prefix = try prefill(model: model, tokens: prefixTokens, parameters: parameters)
        let prefixCache = BlockAwarePrefixCache(modelName: "Qwen3-0.6B-4bit", blockSize: blockSize)
        let storedTable = try prefixCache.storeCache(tokens: prefixTokens, cache: prefix.cache)
        XCTAssertEqual(storedTable.count, 1)

        let allocatedBaseline = prefixCache.manager.allocatedBlocks
        let totalRefBaseline = prefixCache.manager.totalRefCount

        var selectedBranches: [(suffix: [Int], logitError: Float, checkedTokens: Int, mismatches: Int)] = []
        for suffix in candidateSuffixes {
            let branch = try compareBranch(
                model: model,
                prefixCache: prefixCache,
                prefixTokens: prefixTokens,
                suffixTokens: suffix,
                parameters: parameters
            )
            if branch.checkedTokens > 0 {
                selectedBranches.append((suffix, branch.logitError, branch.checkedTokens, branch.mismatches))
            }
            if selectedBranches.count == 2 {
                break
            }
        }
        guard selectedBranches.count == 2 else {
            XCTFail("expected at least two wide-margin suffixes for M3 prefix-cache gate")
            return PrefixCacheGateResult(
                maxLogitError: Float.infinity,
                checkedTokens: 0,
                mismatches: 1,
                sharedBlockCount: 0,
                allocatedBaseline: allocatedBaseline,
                allocatedAfterRequests: prefixCache.manager.allocatedBlocks,
                totalRefBaseline: totalRefBaseline,
                totalRefAfterRequests: prefixCache.manager.totalRefCount
            )
        }
        let suffixX = selectedBranches[0].suffix
        let suffixY = selectedBranches[1].suffix

        var maxLogitError = selectedBranches.map(\.logitError).max() ?? 0
        var checkedTokens = selectedBranches.reduce(0) { $0 + $1.checkedTokens }
        var mismatches = selectedBranches.reduce(0) { $0 + $1.mismatches }

        guard let hitX = prefixCache.fetchCache(tokens: prefixTokens + suffixX),
            let hitY = prefixCache.fetchCache(tokens: prefixTokens + suffixY)
        else {
            XCTFail("expected divergent branches to share the stored prefix block")
            return PrefixCacheGateResult(
                maxLogitError: Float.infinity,
                checkedTokens: 0,
                mismatches: 1,
                sharedBlockCount: 0,
                allocatedBaseline: allocatedBaseline,
                allocatedAfterRequests: prefixCache.manager.allocatedBlocks,
                totalRefBaseline: totalRefBaseline,
                totalRefAfterRequests: prefixCache.manager.totalRefCount
            )
        }
        let sharedBlockCount = zip(hitX.table.blockIDs, hitY.table.blockIDs)
            .filter { $0 == $1 }
            .count
        prefixCache.release(hitX)
        prefixCache.release(hitY)

        for index in 0 ..< 8 {
            let suffix = index.isMultiple(of: 2) ? suffixX : suffixY
            let branch = try compareBranch(
                model: model,
                prefixCache: prefixCache,
                prefixTokens: prefixTokens,
                suffixTokens: suffix,
                parameters: parameters
            )
            maxLogitError = max(maxLogitError, branch.logitError)
            checkedTokens += branch.checkedTokens
            mismatches += branch.mismatches
        }

        return PrefixCacheGateResult(
            maxLogitError: maxLogitError,
            checkedTokens: checkedTokens,
            mismatches: mismatches,
            sharedBlockCount: sharedBlockCount,
            allocatedBaseline: allocatedBaseline,
            allocatedAfterRequests: prefixCache.manager.allocatedBlocks,
            totalRefBaseline: totalRefBaseline,
            totalRefAfterRequests: prefixCache.manager.totalRefCount
        )
    }

    private static func evaluateSSDCacheGate(
        context: ModelContext,
        cacheDirectory: URL,
        byteRoundTripExact: Bool
    ) async throws -> SSDCacheGateResult {
        let model = context.model
        let blockSize = 256
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)
        let seedText = String(
            repeating: "The capital of France is Paris. Swift concurrency protects shared state. GPU kernels execute matrix operations quickly. ",
            count: 128
        )
        let seedTokens = try await tokenIDs(for: seedText, context: context, parameters: parameters)
        var prefixTokens: [Int] = []
        while prefixTokens.count < blockSize {
            prefixTokens.append(contentsOf: seedTokens)
        }
        prefixTokens = Array(prefixTokens.prefix(blockSize))

        let candidateSuffixes = try await [
            " The answer is",
            " Therefore",
            " In summary",
            " Paris",
            " Swift",
            " The next",
        ].mapAsync { text in
            try await tokenIDs(for: text, context: context, parameters: parameters)
        }

        let warmManager = try PagedSSDCacheManager(
            cacheDirectory: cacheDirectory,
            modelName: "Qwen3-0.6B-4bit",
            blockSize: blockSize,
            maxHotBlocks: 0
        )
        let warmCache = BlockAwarePrefixCache(
            modelName: "Qwen3-0.6B-4bit",
            blockSize: blockSize,
            manager: warmManager
        )
        let prefix = try prefill(model: model, tokens: prefixTokens, parameters: parameters)
        let storedTable = try warmCache.storeCache(tokens: prefixTokens, cache: prefix.cache)
        XCTAssertEqual(storedTable.count, 1)
        try await warmManager.flushPendingWrites()

        let restartedManager = try PagedSSDCacheManager(
            cacheDirectory: cacheDirectory,
            modelName: "Qwen3-0.6B-4bit",
            blockSize: blockSize,
            maxHotBlocks: 0
        )
        let restartedCache = BlockAwarePrefixCache(
            modelName: "Qwen3-0.6B-4bit",
            blockSize: blockSize,
            manager: restartedManager
        )

        var maxLogitError: Float = 0
        var checkedTokens = 0
        var mismatches = 0
        for suffix in candidateSuffixes {
            let branch = try compareBranch(
                model: model,
                prefixCache: restartedCache,
                prefixTokens: prefixTokens,
                suffixTokens: suffix,
                parameters: parameters
            )
            maxLogitError = max(maxLogitError, branch.logitError)
            checkedTokens += branch.checkedTokens
            mismatches += branch.mismatches
            if checkedTokens > 0 {
                break
            }
        }

        return SSDCacheGateResult(
            byteRoundTripExact: byteRoundTripExact,
            maxLogitError: maxLogitError,
            checkedTokens: checkedTokens,
            mismatches: mismatches,
            scannedBlocks: restartedManager.allocatedBlocks,
            hotPayloadsAfterScan: restartedManager.hotPayloadCount
        )
    }

    private static func compareBranch(
        model: any LanguageModel,
        prefixCache: BlockAwarePrefixCache,
        prefixTokens: [Int],
        suffixTokens: [Int],
        parameters: GenerateParameters
    ) throws -> (logitError: Float, checkedTokens: Int, mismatches: Int) {
        guard let hit = prefixCache.fetchCache(tokens: prefixTokens + suffixTokens) else {
            throw PrefixCacheTestError.missingPrefixHit
        }
        defer { prefixCache.release(hit) }

        let reconstructedCache = try prefixCache.reconstructCache(from: hit)
        let reconstructedOutput = model(
            LMInput.Text(tokens: MLXArray(suffixTokens.map(Int32.init)))[text: .newAxis],
            cache: reconstructedCache,
            state: nil
        )
        let reconstructedLogits = reconstructedOutput.logits[0..., -1, 0...]
        eval(reconstructedLogits, reconstructedCache)

        let fresh = try prefill(
            model: model,
            tokens: prefixTokens + suffixTokens,
            parameters: parameters
        )

        let logitError = maxAbsoluteDifference(reconstructedLogits, fresh.logits)
        let margin = topOneTopTwoMargin(fresh.logits)
        let reconstructedToken = argMax(reconstructedLogits, axis: -1).item(Int.self)
        let freshToken = argMax(fresh.logits, axis: -1).item(Int.self)

        if margin > logitError * 4 + 1e-3 {
            return (logitError, 1, reconstructedToken == freshToken ? 0 : 1)
        }
        return (logitError, 0, 0)
    }

    private static func prefill(
        model: any LanguageModel,
        tokens: [Int],
        parameters: GenerateParameters
    ) throws -> (cache: [any KVCache], logits: MLXArray) {
        let cache = try model.newCache(parameters: parameters)
        let input = LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)))
        let output = model(input[text: .newAxis], cache: cache, state: nil)
        let logits = output.logits[0..., -1, 0...]
        eval(logits, cache)
        return (cache, logits)
    }

    private static func maxAbsoluteDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }

    private static func topOneTopTwoMargin(_ logits: MLXArray) -> Float {
        let topValues = top(logits.asType(.float32), k: 2, axis: -1).asArray(Float.self)
        let sorted = topValues.sorted(by: >)
        return sorted[0] - sorted[1]
    }

    private static func tokenIDs(
        for text: String,
        context: ModelContext,
        parameters: GenerateParameters
    ) async throws -> [Int] {
        let input = try await context.processor.prepare(input: UserInput(prompt: text))
        let cache = try context.model.newCache(parameters: parameters)
        switch try context.model.prepare(input, cache: cache, state: nil, windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            return tokens.tokens.asArray(Int.self)
        case .logits:
            throw PrefixCacheTestError.missingPrefixHit
        }
    }

    private static func exactBF16RoundTrip(in cacheDirectory: URL) throws -> Bool {
        let hash = Data(repeating: 0xAB, count: 32)
        let raw = MLXArray([UInt16(0x3f80), UInt16(0x4000), UInt16(0xbf80)], [1, 1, 3, 1])
        let bf16 = raw.view(dtype: .bfloat16)
        let payload = KVCacheBlockPayload(
            layers: [
                CacheLayerBlockPayload(
                    keys: bf16,
                    values: bf16,
                    metaState: ["3", CacheTypeHandlers.encodeBool(false)]
                )
            ]
        )
        let snapshot = SafetensorsBlockIO.snapshot(
            hash: hash,
            payload: payload,
            tokenCount: 3,
            modelName: "byte-test",
            blockSize: 3
        )
        let url = cacheDirectory
            .appendingPathComponent("byte", isDirectory: true)
            .appendingPathComponent("roundtrip.safetensors")
        try SafetensorsBlockIO.write(snapshot, to: url)
        let loaded = try SafetensorsBlockIO.read(from: url)
        let (mlxLoaded, mlxMetadata) = try loadArraysAndMetadata(url: url, stream: .cpu)

        let originalBytes = raw.asData(access: .copy).data
        guard loaded.rawTensorBytes["layer.0.keys"] == originalBytes,
            loaded.rawTensorBytes["layer.0.values"] == originalBytes,
            loaded.metadata["isRotating"] == CacheTypeHandlers.encodeBool(false),
            mlxMetadata["isRotating"] == CacheTypeHandlers.encodeBool(false),
            let loadedKeys = loaded.arrays["layer.0.keys"],
            let loadedValues = loaded.arrays["layer.0.values"],
            let mlxLoadedKeys = mlxLoaded["layer.0.keys"]
        else {
            return false
        }

        return loadedKeys.view(dtype: .uint16).asArray(UInt16.self) == raw.asArray(UInt16.self)
            && loadedValues.view(dtype: .uint16).asArray(UInt16.self) == raw.asArray(UInt16.self)
            && mlxLoadedKeys.view(dtype: .uint16).asArray(UInt16.self) == raw.asArray(UInt16.self)
    }

    private static func unitPayload() -> KVCacheBlockPayload {
        KVCacheBlockPayload(
            layers: [
                CacheLayerBlockPayload(
                    keys: MLXArray.zeros([1, 1, 1], dtype: .float32),
                    values: MLXArray.zeros([1, 1, 1], dtype: .float32),
                    metaState: ["1", CacheTypeHandlers.encodeBool(false)]
                )
            ]
        )
    }
}

private enum PrefixCacheTestError: Error {
    case missingPrefixHit
}

private final class UnsupportedPrefixTestCache: KVCache {
    var state: [MLXArray]
    var metaState: [String] = []
    var offset: Int { 0 }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    init(state: [MLXArray]) {
        self.state = state
    }

    func innerState() -> [MLXArray] {
        state
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        state = [newKeys, newValues]
        return (newKeys, newValues)
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        0
    }

    func copy() -> any KVCache {
        UnsupportedPrefixTestCache(state: state.map { $0[.ellipsis] })
    }
}

private extension Array {
    func mapAsync<T>(_ transform: (Element) async throws -> T) async throws -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self {
            result.append(try await transform(element))
        }
        return result
    }
}
