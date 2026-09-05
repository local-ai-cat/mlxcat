import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXVLM
import Tokenizers

public final class NativeModelEngine: @unchecked Sendable {
    let modelID: String

    let tokenizer: any MLXLMCommon.Tokenizer
    let resident: AsyncResourceOwner<NativeModelResident>
    let parameters: GenerateParameters
    private let prefixStore: SessionPrefixKVStore
    private let cacheCapabilities: ModelCacheCapabilities
    let eosTokenIds: Set<Int>
    private let maxContextTokens: Int?
    let grammarVocabularyLock = NSLock()
    var cachedGrammarVocabulary: JSONGrammarVocabulary?
    var cachedToolGrammarVocabulary: JSONGrammarVocabulary?
    var cachedToolGrammarDenyIndex: QwenXMLToolBodyDenyIndex?

    init(
        context: ModelContext,
        modelID: String,
        maxContextTokens: Int?,
        maxConcurrentRequests: Int,
        cacheCapabilities: ModelCacheCapabilities,
        serializationPolicy: SerializationPolicy,
        schedulerManagedTextPrefill: Bool,
        chunkIdlePrefill: Bool,
        kvQuantization: KVQuantizationPolicy = .off
    ) {
        self.tokenizer = context.tokenizer
        self.modelID = modelID
        self.maxContextTokens = maxContextTokens
        self.parameters = GenerateParameters(maxTokens: 16, temperature: 0)
        self.cacheCapabilities = cacheCapabilities
        self.prefixStore = SessionPrefixKVStore(maxBytes: Self.prefixCacheMaxBytes())
        let engine = MLXCatEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: maxConcurrentRequests,
            prefixStore: prefixStore,
            cacheCapabilities: cacheCapabilities,
            serializationPolicy: serializationPolicy,
            schedulerManagedTextPrefill: schedulerManagedTextPrefill,
            chunkIdlePrefill: chunkIdlePrefill,
            kvQuantization: kvQuantization
        )
        self.resident = AsyncResourceOwner(NativeModelResident(context: context, engine: engine))
        var eosTokenIds = context.configuration.eosTokenIds
        if let tokenizerEosTokenId = tokenizer.eosTokenId {
            eosTokenIds.insert(tokenizerEosTokenId)
        }
        self.eosTokenIds = eosTokenIds
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        try await prepareChatCompletion(request).start(nil)
    }

    typealias PreparedChat = PreparedNativeChat

    func prepareChatCompletion(_ request: OpenAIChatRequest) async throws -> PreparedChat {
        let input = try await prepareChatInput(request)
        let promptTokens = try countPromptTokens(input)
        try Self.validateContextWindow(
            promptTokens: promptTokens,
            maxTokens: request.maxTokens,
            contextWindow: maxContextTokens
        )
        let uid = "chat-\(UUID().uuidString)"
        let mlxRequest = Request(
            uid: uid,
            input: input,
            maxTokens: request.maxTokens,
            sampling: try samplingParameters(from: request),
            eosTokenIds: eosTokenIds,
            cacheSession: request.cacheSession
        )

        return PreparedChat(
            start: { allocation in
                try await self.makeStream(from: mlxRequest, promptTokens: promptTokens, allocation: allocation)
            },
            promptTokens: promptTokens
        )
    }

    func startCompletion(_ request: OpenAICompletionRequest) async throws -> OpenAIChatStream {
        try await prepareCompletion(request).start(nil)
    }

    func prepareCompletion(_ request: OpenAICompletionRequest) async throws -> PreparedChat {
        guard case .string(let prompt) = request.prompt else {
            throw NativeModelEngineError.invalidPrompt
        }
        let tokenIDs = tokenizer.encode(text: prompt, addSpecialTokens: true)
        try Self.validateContextWindow(
            promptTokens: tokenIDs.count,
            maxTokens: request.maxTokens,
            contextWindow: maxContextTokens
        )
        let input = LMInput(tokens: MLXArray(tokenIDs))
        let uid = "cmpl-\(UUID().uuidString)"
        let mlxRequest = Request(
            uid: uid,
            input: input,
            maxTokens: request.maxTokens,
            sampling: try samplingParameters(from: request),
            eosTokenIds: eosTokenIds
        )
        return PreparedChat(
            start: { allocation in
                try await self.makeStream(from: mlxRequest, promptTokens: tokenIDs.count, allocation: allocation)
            },
            promptTokens: tokenIDs.count
        )
    }

    func prefixCacheStats() -> SessionPrefixKVStoreStats {
        prefixStore.stats
    }

    func countPromptTokens(for request: OpenAIChatRequest) async throws -> Int {
        let input = try await prepareChatInput(request)
        return try countPromptTokens(input)
    }

    func countPromptTokens(for request: OpenAICompletionRequest) async throws -> Int {
        guard case .string(let prompt) = request.prompt else {
            throw NativeModelEngineError.invalidPrompt
        }
        return tokenizer.encode(text: prompt, addSpecialTokens: true).count
    }

    static func validateContextWindow(
        promptTokens: Int,
        maxTokens: Int,
        contextWindow: Int?
    ) throws {
        guard let contextWindow, promptTokens + maxTokens > contextWindow else { return }
        throw OpenAIHTTPError(
            status: 400,
            message: "This model's maximum context length is \(contextWindow) tokens, "
                + "but the request requires \(promptTokens + maxTokens) tokens "
                + "(\(promptTokens) prompt + \(maxTokens) completion).",
            code: "context_length_exceeded"
        )
    }

    func estimatedKVCacheBytes(promptTokens: Int, maxGeneratedTokens: Int) -> Int64? {
        cacheCapabilities.kvCacheProfile?.estimatedBytes(
            promptTokens: promptTokens,
            maxGeneratedTokens: maxGeneratedTokens
        )
    }

    private func makeStream(from mlxRequest: Request, promptTokens: Int,
                            allocation: RetainedAllocation?) async throws -> OpenAIChatStream {
        let request = mlxRequest.retaining(allocation)
        let session = try await resident.withValue { $0.engine.streamSession(request) }
        let responseStream = session.responses
        let (chunks, continuation) = AsyncThrowingStream<OpenAIChatChunk, Error>.makeStream()
        let task = Task {
            do {
                for try await response in responseStream {
                    if case .failed(let message)? = response.finishReason {
                        throw NativeModelEngineError.generationFailed(message)
                    }
                    guard response.token >= 0 else {
                        if response.finishReason != nil {
                            break
                        }
                        continue
                    }
                    let text = self.tokenizer.decode(
                        tokenIds: [response.token],
                        skipSpecialTokens: response.finishReason != nil
                            && self.eosTokenIds.contains(response.token)
                    )
                    continuation.yield(
                        OpenAIChatChunk(
                            text: text,
                            tokenID: response.token,
                            finishReason: openAIFinishReason(response.finishReason),
                            cachedPromptTokens: response.cachedPromptTokens
                        )
                    )
                    if response.finishReason != nil {
                        break
                    }
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return OpenAIChatStream(promptTokens: promptTokens, chunks: chunks, cancelAndWait: {
            await session.cancelAndWait()
            task.cancel()
            await task.value
        })
    }

    public func shutdown() async {
        let prefixStore = self.prefixStore
        await resident.invalidate { state in
            await state.engine.shutdown()
            Stream.gpu.synchronize()
            await prefixStore.clearAndWait()
        }
    }

    private func countPromptTokens(_ input: LMInput) throws -> Int {
        input.text.tokens.size
    }

    /// Byte budget for retained prefix-KV snapshots. Unset, the store used to
    /// run uncapped (`maxBytes: 0`) with up to 8 anonymous slots pinned — full
    /// KV snapshots that reach GBs on large models. Default: 1/12th of
    /// physical memory, clamped to [512 MB, 4 GB]. The env var still wins,
    /// including an explicit `0` for the old uncapped behavior.
    static func prefixCacheMaxBytes() -> Int64 {
        if let raw = ProcessInfo.processInfo.environment["MLXSERVE_PREFIX_CACHE_MAX_BYTES"],
           let explicit = Int64(raw) {
            return explicit
        }
        let physical = Int64(ProcessInfo.processInfo.physicalMemory)
        let candidate = physical / 12
        let minimumBytes: Int64 = 512 << 20
        let maximumBytes: Int64 = 4 << 30
        return min(max(candidate, minimumBytes), maximumBytes)
    }

    func xtcSpecialTokens() -> [Int] {
        let newlineTokens = tokenizer.encode(text: "\n", addSpecialTokens: false)
        return Array(Set(newlineTokens).union(eosTokenIds)).sorted()
    }

}

enum NativeModelEngineError: Error, CustomStringConvertible {
    case generationFailed(String)
    case invalidPrompt
    case invalidImageReference(String)

    var description: String {
        switch self {
        case .generationFailed(let message):
            return message
        case .invalidPrompt:
            return "completion backend requires a single string prompt"
        case .invalidImageReference(let message):
            return "invalid image reference: \(message)"
        }
    }
}

private func openAIFinishReason(_ finishReason: FinishReason?) -> String? {
    switch finishReason {
    case .stop:
        return "stop"
    case .length:
        return "length"
    case .cancelled:
        return "stop"
    case .failed:
        return "stop"
    case nil:
        return nil
    }
}

final class NativeModelResident: @unchecked Sendable {
    let context: ModelContext
    let engine: MLXCatEngine
    init(context: ModelContext, engine: MLXCatEngine) { self.context = context; self.engine = engine }
}
