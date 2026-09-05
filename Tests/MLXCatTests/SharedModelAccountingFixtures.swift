import Foundation
import MLXCat
import MLXCatHTTP
import MLXCatSpeech
@testable import MLXCatHTTPServer

struct AccountingChatLoader: EnginePoolModelLoader {
    var unloadGate: AccountingBarrier?
    func loadModel(id: String, modelURL: URL) async throws -> String { id }
    func unloadModel(_ engine: String, id: String) async { await unloadGate?.wait() }
}

enum AccountingFailure: Error { case injected }

actor AccountingBarrier {
    private(set) var entries = 0
    private var entered = false
    private var opened = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        entries += 1
        entered = true
        entryWaiters.forEach { $0.resume() }
        entryWaiters.removeAll()
        guard !opened else { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { entryWaiters.append($0) }
    }

    func open() {
        opened = true
        waiters.forEach { $0.resume() }
        waiters.removeAll()
    }
}

actor AccountingResourceState {
    let loadGate: AccountingBarrier?
    let unloadGate: AccountingBarrier?
    let failFirstLoad: Bool
    private(set) var loads = 0
    private(set) var unloads = 0
    private(set) var loaderWasCancelled = false

    init(loadGate: AccountingBarrier? = nil, unloadGate: AccountingBarrier? = nil, failFirstLoad: Bool = false) {
        self.loadGate = loadGate
        self.unloadGate = unloadGate
        self.failFirstLoad = failFirstLoad
    }

    func load() async throws {
        loads += 1
        await loadGate?.wait()
        loaderWasCancelled = Task.isCancelled
        if failFirstLoad && loads == 1 { throw AccountingFailure.injected }
    }

    func unload() async {
        unloads += 1
        await unloadGate?.wait()
    }
}

actor AccountingRerankCache: NativeRerankModelServing {
    private(set) var scores = 0
    let work: AccountingBarrier?
    private(set) var loads = 0
    private(set) var unloads = 0
    init(work: AccountingBarrier? = nil) { self.work = work }
    func load(id: String, modelURL: URL) async throws { loads += 1 }
    func unload(id: String) async { unloads += 1 }
    func score(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult {
        scores += 1
        await work?.wait()
        try Task.checkCancellation()
        return OpenAIRerankResult(scores: [1], indices: [0], totalTokens: 1)
    }
}

actor AccountingSpeechAdapter: SpeechEngineAdapter {
    private(set) var loads = 0
    private(set) var transcriptions = 0
    nonisolated let engineID = "fixture"
    nonisolated let displayName = "CPU fixture"
    nonisolated let capabilities = SpeechEngineCapabilities(
        supportsFileTranscription: true, supportsStreaming: true, silicon: .cpu
    )
    let bytes: Int64
    let work: AccountingBarrier
    private(set) var unloads = 0
    init(bytes: Int64, work: AccountingBarrier) { self.bytes = bytes; self.work = work }
    func availableModels() -> [SpeechModelInfo] {
        [SpeechModelInfo(id: "speech", engineID: engineID, displayName: "Fixture", footprint: bytes)]
    }
    func loadModel(_ modelID: String) async throws { loads += 1 }
    func unloadModel(_ modelID: String) async { unloads += 1 }
    func transcribeFile(_ request: SpeechFileTranscriptionRequest) async throws -> SpeechTranscriptionResult {
        transcriptions += 1
        await work.wait()
        return SpeechTranscriptionResult(text: "fixture")
    }
    func makeStreamSession(modelID: String, language: String?) async throws -> any SpeechStreamSession {
        AccountingSpeechSession(work: work)
    }
}

actor AccountingSpeechSession: SpeechStreamSession {
    nonisolated let segments = AsyncThrowingStream<SpeechStreamSegment, Error> { $0.finish() }
    let work: AccountingBarrier
    init(work: AccountingBarrier) { self.work = work }
    func push(_ buffer: SpeechPCMBuffer) async throws { await work.wait() }
    func finish() async throws {}
    func cancel() async {}
    func latencyStats() async -> SpeechSessionLatencyStats { .empty }
}

final class AccountingCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func increment() { lock.withLock { count += 1 } }
    var value: Int { lock.withLock { count } }
}
