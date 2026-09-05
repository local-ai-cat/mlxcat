import Foundation
import MLXCat
@testable import MLXCatHTTP
@testable import MLXCatNative
import Testing

extension SharedModelAccountingTests {
    enum HTTPExit: CaseIterable, Sendable {
        case anthropicStop, anthropicHeaderFailure, anthropicDeltaFailure
        case responsesHeaderFailure, responsesDeltaFailure

        var body: [String: Any] {
            if isAnthropic {
                return ["model": "chat", "max_tokens": 20, "stream": self != .anthropicStop,
                        "messages": [["role": "user", "content": "hello"]], "stop_sequences": ["STOP"]]
            }
            return ["model": "chat", "input": "hello", "stream": true]
        }

        var isAnthropic: Bool {
            switch self {
            case .anthropicStop, .anthropicHeaderFailure, .anthropicDeltaFailure: return true
            default: return false
            }
        }
    }

    @Test(arguments: HTTPExit.allCases)
    func actualDialectHandlersAwaitProducerTeardown(exit: HTTPExit) async throws {
        let pool = EnginePool(models: ["chat": model("chat", 40)], loader: AccountingChatLoader(), finalCeiling: 100)
        let lease = try await pool.acquire("chat")
        let reservation = try await pool.reserveRequest(additionalBytes: 20)
        let ticket = try await pool.admitWaitingRequest()
        let teardown = AccountingBarrier()
        let completed = AccountingCounter()
        let (chunks, continuation) = AsyncThrowingStream<OpenAIChatChunk, Error>.makeStream()
        let producer = OpenAIChatStream(promptTokens: 1, chunks: chunks, cancelAndWait: {
            await teardown.wait()
            continuation.finish()
        })
        let stream = PoolLeasedChatStream.make(producer, pool: pool, lease: lease,
                                               ticket: ticket, reservation: reservation)
        let backend = DialectAccountingBackend(stream: stream)
        let writer = DialectAccountingWriter(exit: exit)
        let request = HTTPRequest(method: "POST", path: "/fixture", headers: [:],
                                  body: try JSONSerialization.data(withJSONObject: exit.body))
        let handler = Task {
            defer { completed.increment() }
            if exit.isAnthropic {
                try await AnthropicMessagesHandler(backend: backend).handleMessages(request, connection: writer)
            } else {
                try await ResponsesHandler(backend: backend, store: ResponsesStore())
                    .handleCreate(request, connection: writer)
            }
        }
        continuation.yield(OpenAIChatChunk(text: "hello STOP", tokenID: 1))
        await teardown.waitUntilEntered()
        #expect(completed.value == 0)
        #expect(await pool.status().models.first?.inUse == 1)
        #expect(await pool.reservedRequestBytes == 20)
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 0)
        await teardown.open()
        do {
            try await handler.value
            #expect(exit == .anthropicStop)
        } catch is AccountingFailure { #expect(exit != .anthropicStop) }
        #expect(completed.value == 1)
        #expect(await teardown.entries == 1)
        #expect(await pool.status().models.first?.inUse == 0)
        #expect(await pool.reservedRequestBytes == 0)
        if exit == .anthropicStop {
            #expect(await writer.output.contains("stop_sequence"))
            #expect(await writer.output.contains("hello "))
        } else {
            #expect(await writer.failed)
        }
    }
}

private struct DialectAccountingBackend: OpenAIChatBackend {
    let models = [OpenAIModelInfo(id: "chat")]
    let stream: OpenAIChatStream
    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream { stream }
}

private actor DialectAccountingWriter: HTTPResponseWriting {
    let exit: SharedModelAccountingTests.HTTPExit
    private(set) var output = ""
    private(set) var failed = false
    init(exit: SharedModelAccountingTests.HTTPExit) { self.exit = exit }
    func sendFinal(data: Data) async throws { try await send(data: data) }
    func send(data: Data) async throws {
        let text = try #require(String(data: data, encoding: .utf8))
        switch exit {
        case .anthropicStop: break
        case .anthropicHeaderFailure, .responsesHeaderFailure: failed = true
        case .anthropicDeltaFailure:
            if text.contains("content_block_delta") { failed = true }
        case .responsesDeltaFailure:
            if text.contains("response.output_text.delta") { failed = true }
        }
        if failed { throw AccountingFailure.injected }
        output += text
    }
}
