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

    private let context: ModelContext
    private let engine: MLXCatEngine
    private let parameters: GenerateParameters
    private let prefixStore: SessionPrefixKVStore
    private let cacheCapabilities: ModelCacheCapabilities
    private let eosTokenIds: Set<Int>
    private let maxContextTokens: Int?
    private let grammarVocabularyLock = NSLock()
    private var cachedGrammarVocabulary: JSONGrammarVocabulary?
    private var cachedToolGrammarVocabulary: JSONGrammarVocabulary?
    private var cachedToolGrammarDenyIndex: QwenXMLToolBodyDenyIndex?

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
        self.context = context
        self.modelID = modelID
        self.maxContextTokens = maxContextTokens
        self.parameters = GenerateParameters(maxTokens: 16, temperature: 0)
        self.cacheCapabilities = cacheCapabilities
        self.prefixStore = SessionPrefixKVStore(maxBytes: Self.prefixCacheMaxBytes())
        self.engine = MLXCatEngine(
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
        var eosTokenIds = context.configuration.eosTokenIds
        if let tokenizerEosTokenId = context.tokenizer.eosTokenId {
            eosTokenIds.insert(tokenizerEosTokenId)
        }
        self.eosTokenIds = eosTokenIds
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        try await prepareChatCompletion(request).stream
    }

    struct PreparedChat {
        let stream: OpenAIChatStream
        let promptTokens: Int
    }

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
            stream: makeStream(from: mlxRequest, promptTokens: promptTokens),
            promptTokens: promptTokens
        )
    }

    func startCompletion(_ request: OpenAICompletionRequest) async throws -> OpenAIChatStream {
        try await prepareCompletion(request).stream
    }

    func prepareCompletion(_ request: OpenAICompletionRequest) async throws -> PreparedChat {
        guard case .string(let prompt) = request.prompt else {
            throw NativeModelEngineError.invalidPrompt
        }
        let tokenIDs = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
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
            stream: makeStream(from: mlxRequest, promptTokens: tokenIDs.count),
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
        return context.tokenizer.encode(text: prompt, addSpecialTokens: true).count
    }

    static func validateContextWindow(
        promptTokens: Int,
        maxTokens: Int,
        contextWindow: Int?
    ) throws {
        guard let contextWindow, promptTokens + maxTokens > contextWindow else { return }
        throw OpenAIHTTPError(
            status: 400,
            message: "This model's maximum context length is \(contextWindow) tokens, but the request requires \(promptTokens + maxTokens) tokens (\(promptTokens) prompt + \(maxTokens) completion).",
            code: "context_length_exceeded"
        )
    }

    func estimatedKVCacheBytes(promptTokens: Int, maxGeneratedTokens: Int) -> Int64 {
        cacheCapabilities.kvCacheProfile?.estimatedBytes(
            promptTokens: promptTokens,
            maxGeneratedTokens: maxGeneratedTokens
        ) ?? 0
    }

    private func makeStream(from mlxRequest: Request, promptTokens: Int) -> OpenAIChatStream {
        let responseStream = engine.stream(mlxRequest)
        let chunks = AsyncThrowingStream<OpenAIChatChunk, Error> { continuation in
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
                        let text = self.context.tokenizer.decode(
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
            continuation.onTermination = { _ in
                task.cancel()
            }
        }

        return OpenAIChatStream(promptTokens: promptTokens, chunks: chunks)
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
            let explicit = Int64(raw)
        {
            return explicit
        }
        let physical = Int64(ProcessInfo.processInfo.physicalMemory)
        let candidate = physical / 12
        let minimumBytes: Int64 = 512 << 20
        let maximumBytes: Int64 = 4 << 30
        return min(max(candidate, minimumBytes), maximumBytes)
    }

    private func xtcSpecialTokens() -> [Int] {
        let newlineTokens = context.tokenizer.encode(text: "\n", addSpecialTokens: false)
        return Array(Set(newlineTokens).union(eosTokenIds)).sorted()
    }

    private func userInput(
        from request: OpenAIChatRequest,
        reasoningEffort: ReasoningEffortTemplateValue
    ) throws -> UserInput {
        UserInput(
            chat: try injectedMessages(from: request).map(chatMessage),
            tools: toolSpecDictionaries(
                from: selectOpenAITools(tools: request.tools, toolChoice: request.toolChoice)
            ),
            additionalContext: additionalContext(from: request, reasoningEffort: reasoningEffort)
        )
    }

    private func prepareChatInput(_ request: OpenAIChatRequest) async throws -> LMInput {
        let candidates = ReasoningEffortCompatibility.candidates(
            for: request.chatTemplateKwargs?["reasoning_effort"],
            isHarmony: usesHarmonyChannels
        )
        var originalError: Error?

        for candidate in candidates {
            do {
                return try await context.processor.prepare(
                    input: try userInput(from: request, reasoningEffort: candidate)
                )
            } catch {
                if originalError == nil {
                    originalError = error
                }
            }
        }

        throw originalError ?? NativeModelEngineError.invalidPrompt
    }

    private func samplingParameters(from request: OpenAIChatRequest) throws -> SamplingParameters {
        SamplingParameters(
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            minP: request.minP,
            repetitionPenalty: request.repetitionPenalty,
            presencePenalty: request.presencePenalty,
            frequencyPenalty: request.frequencyPenalty,
            minTokens: request.minTokens,
            logitBias: request.logitBias,
            eosTokenIds: eosTokenIds,
            xtcProbability: request.xtcProbability,
            xtcThreshold: request.xtcThreshold,
            xtcSpecialTokens: xtcSpecialTokens(),
            seed: request.seed,
            allowedSequences: allowedSequences(from: request.structuredOutput),
            jsonGrammar: jsonGrammar(from: request.structuredOutput),
            regexGrammar: try regexGrammar(from: request.structuredOutput),
            gbnfGrammar: try gbnfGrammar(from: request.structuredOutput),
            toolGrammar: toolGrammar(from: request),
            thinkingBudget: thinkingBudgetConfiguration(from: request)
        )
    }

    private func samplingParameters(from request: OpenAICompletionRequest) throws -> SamplingParameters {
        SamplingParameters(
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            minP: request.minP,
            repetitionPenalty: request.repetitionPenalty,
            presencePenalty: request.presencePenalty,
            frequencyPenalty: request.frequencyPenalty,
            minTokens: request.minTokens,
            logitBias: request.logitBias,
            eosTokenIds: eosTokenIds,
            xtcProbability: 0,
            xtcThreshold: 0.1,
            xtcSpecialTokens: xtcSpecialTokens(),
            seed: request.seed,
            allowedSequences: allowedSequences(from: request.structuredOutput),
            jsonGrammar: jsonGrammar(from: request.structuredOutput),
            regexGrammar: try regexGrammar(from: request.structuredOutput),
            gbnfGrammar: try gbnfGrammar(from: request.structuredOutput)
        )
    }

    private func jsonGrammar(from structuredOutput: StructuredOutputSpec) -> JSONGrammarConfiguration? {
        switch structuredOutput {
        case .none, .choice, .regex, .grammar:
            return nil
        case .jsonObject:
            return JSONGrammarConfiguration(vocabulary: grammarVocabulary(), schema: .jsonObject)
        case .jsonSchema(_, let schema):
            return JSONGrammarConfiguration(
                vocabulary: grammarVocabulary(),
                schema: JSONSchemaNode(openAISchema: schema)
            )
        }
    }

    private func regexGrammar(from structuredOutput: StructuredOutputSpec) throws -> RegexGrammarConfiguration? {
        guard case .regex(let pattern) = structuredOutput else { return nil }
        do {
            return try RegexGrammarConfiguration(vocabulary: grammarVocabulary(), pattern: pattern)
        } catch let error as RegexGrammarError {
            throw OpenAIServerError.invalidStructuredOutput(error.description)
        }
    }

    private func gbnfGrammar(from structuredOutput: StructuredOutputSpec) throws -> GBNFGrammarConfiguration? {
        guard case .grammar(let grammar) = structuredOutput else { return nil }
        do {
            return try GBNFGrammarConfiguration(vocabulary: grammarVocabulary(), grammar: grammar)
        } catch let error as GBNFGrammarError {
            throw OpenAIServerError.invalidStructuredOutput(error.description)
        }
    }

    private func toolGrammar(from request: OpenAIChatRequest) -> QwenXMLToolGrammarConfiguration? {
        guard QwenXMLToolGrammarPolicy.isEnabled(
            requestOverride: request.toolGrammar,
            environmentValue: ProcessInfo.processInfo.environment["MLXCAT_TOOL_GRAMMAR"]
        ) else {
            return nil
        }
        guard case .none = request.structuredOutput else {
            return nil
        }
        switch request.toolChoice {
        case nil, .some(.auto):
            break
        case .some(.none), .some(.required), .some(.function):
            return nil
        }
        guard toolCallModelFamily(forModel: modelID) == .qwenXML
            || toolCallModelFamily(forModel: request.model) == .qwenXML
        else {
            return nil
        }
        guard let tools = selectOpenAITools(tools: request.tools, toolChoice: request.toolChoice),
            !tools.isEmpty
        else {
            return nil
        }
        let toolNames = tools.compactMap(openAIToolFunctionName)
        guard !toolNames.isEmpty,
            let startTokenID = singleTokenID(for: "<tool_call>"),
            let endTokenID = singleTokenID(for: "</tool_call>")
        else {
            return nil
        }

        let (vocabulary, denyIndex) = toolGrammarVocabulary(closeTokenID: endTokenID)
        let whitespaceTokenIDs = vocabulary.tokens.compactMap { token -> Int? in
            guard !token.isEOS,
                !token.text.isEmpty,
                token.text.allSatisfy(QwenXMLToolGrammarConfiguration.isXMLWhitespace)
            else {
                return nil
            }
            return token.id
        }
        var postToolAllowedTokenIDs = Set(whitespaceTokenIDs)
        postToolAllowedTokenIDs.insert(startTokenID)
        postToolAllowedTokenIDs.formUnion(eosTokenIds)
        if let imEndTokenID = singleTokenID(for: "<|im_end|>") {
            postToolAllowedTokenIDs.insert(imEndTokenID)
        }

        return QwenXMLToolGrammarConfiguration(
            vocabulary: vocabulary,
            toolNames: toolNames,
            toolCallStartTokenID: startTokenID,
            toolCallEndTokenID: endTokenID,
            postToolAllowedTokenIDs: Array(postToolAllowedTokenIDs),
            denyIndex: denyIndex
        )
    }

    private func singleTokenID(for text: String) -> Int? {
        context.tokenizer.convertTokenToId(text)
    }

    /// Both the augmented vocabulary and its deny index are O(vocab) to build and
    /// depend only on the loaded model, but this runs on every tools-carrying
    /// request — so cache them next to the base vocabulary rather than rebuilding
    /// ~150k tokens per completion.
    private func toolGrammarVocabulary(closeTokenID: Int) -> (JSONGrammarVocabulary, QwenXMLToolBodyDenyIndex) {
        grammarVocabularyLock.lock()
        if let cachedToolGrammarVocabulary, let cachedToolGrammarDenyIndex {
            grammarVocabularyLock.unlock()
            return (cachedToolGrammarVocabulary, cachedToolGrammarDenyIndex)
        }
        grammarVocabularyLock.unlock()

        let baseVocabulary = grammarVocabulary()
        let vocabulary: JSONGrammarVocabulary
        if baseVocabulary.tokens.contains(where: { $0.id == closeTokenID }) {
            vocabulary = baseVocabulary
        } else {
            var tokens = baseVocabulary.tokens
            tokens.append(JSONGrammarToken(id: closeTokenID, text: "</tool_call>"))
            vocabulary = JSONGrammarVocabulary(tokens: tokens)
        }
        let denyIndex = QwenXMLToolBodyDenyIndex(vocabulary: vocabulary)

        grammarVocabularyLock.lock()
        cachedToolGrammarVocabulary = vocabulary
        cachedToolGrammarDenyIndex = denyIndex
        grammarVocabularyLock.unlock()
        return (vocabulary, denyIndex)
    }

    private func grammarVocabulary() -> JSONGrammarVocabulary {
        grammarVocabularyLock.lock()
        defer { grammarVocabularyLock.unlock() }
        if let cachedGrammarVocabulary {
            return cachedGrammarVocabulary
        }
        let vocabulary = buildGrammarVocabulary()
        cachedGrammarVocabulary = vocabulary
        return vocabulary
    }

    private func buildGrammarVocabulary() -> JSONGrammarVocabulary {
        // The tokenizer protocol exposes no vocabulary size; ids are dense, so scan until a
        // long run of unmapped ids marks the end. Built once per loaded model, lazily on the
        // first json_object/json_schema request.
        let missRunLimit = 2048
        let hardCap = 1_000_000
        var tokens: [JSONGrammarToken] = []
        var consecutiveMisses = 0
        var id = 0
        while id < hardCap && consecutiveMisses < missRunLimit {
            defer { id += 1 }
            guard context.tokenizer.convertIdToToken(id) != nil else {
                consecutiveMisses += 1
                continue
            }
            consecutiveMisses = 0
            if eosTokenIds.contains(id) {
                tokens.append(JSONGrammarToken(id: id, text: "", isEOS: true))
                continue
            }
            let text = context.tokenizer.decode(tokenIds: [id], skipSpecialTokens: true)
            // Skip specials (decode to empty) and byte-fragment tokens (replacement char):
            // masking them keeps constrained output valid UTF-8 JSON.
            guard !text.isEmpty, !text.contains("\u{FFFD}") else { continue }
            tokens.append(JSONGrammarToken(id: id, text: text))
        }
        return JSONGrammarVocabulary(tokens: tokens)
    }

    private func allowedSequences(from structuredOutput: StructuredOutputSpec) -> [[Int]]? {
        guard case .choice(let choices) = structuredOutput else { return nil }
        let eosTokenId = canonicalEOSTokenId()
        let sequences = choices.compactMap { choice -> [Int]? in
            var sequence = context.tokenizer.encode(text: choice, addSpecialTokens: false)
            guard !sequence.isEmpty else { return nil }
            if let eosTokenId {
                sequence.append(eosTokenId)
            }
            return sequence
        }
        return sequences.isEmpty ? nil : sequences
    }

    private func thinkingBudgetConfiguration(from request: OpenAIChatRequest) -> ServeThinkingBudgetConfiguration? {
        guard let budget = request.thinkingBudget else { return nil }

        let closeTokenIDs: [Int]
        let trailingTokenIDs: [Int]
        if usesHarmonyChannels {
            closeTokenIDs = context.tokenizer.encode(text: "<|end|>", addSpecialTokens: false)
            trailingTokenIDs = context.tokenizer.encode(
                text: "<|start|>assistant<|channel|>final<|message|>",
                addSpecialTokens: false
            )
        } else {
            closeTokenIDs = context.tokenizer.encode(text: "</think>", addSpecialTokens: false)
            trailingTokenIDs = []
        }

        guard !closeTokenIDs.isEmpty else { return nil }
        return ServeThinkingBudgetConfiguration(
            budget: budget,
            closeTokenIDs: closeTokenIDs,
            startTokenIDs: context.tokenizer.encode(text: "<think>", addSpecialTokens: false),
            trailingTokenIDs: trailingTokenIDs,
            startsInThinking: true
        )
    }

    private func canonicalEOSTokenId() -> Int? {
        // Choice constraints are terminal only when the choice trie includes EOS.
        // If a model exposes no EOS id, fall back to bare choices and let maxTokens stop.
        context.tokenizer.eosTokenId ?? eosTokenIds.first
    }

    private func injectedMessages(from request: OpenAIChatRequest) throws -> [OpenAIChatMessage] {
        let directive: String?
        switch request.structuredOutput {
        case .jsonObject:
            directive = "You must respond with only a single valid JSON object and no other text."
        case .jsonSchema(_, let schema):
            let schemaJSON = try serializedJSONSchema(schema)
            directive = "You must respond with only a single valid JSON object conforming to this JSON Schema: \(schemaJSON). Output only the JSON."
        case .regex(let pattern):
            directive = "You must respond with text matching this regular expression and no other text: \(pattern)"
        case .grammar(let grammar):
            directive = "You must respond with text matching this GBNF grammar and no other text: \(grammar)"
        case .none, .choice:
            directive = nil
        }

        guard let directive else { return request.messages }
        return [OpenAIChatMessage(role: "system", content: directive)] + request.messages
    }

    private func serializedJSONSchema(_ schema: [String: OpenAIJSONValue]) throws -> String {
        let object = schema.mapValues { jsonObject(from: $0) }
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    private func jsonObject(from value: OpenAIJSONValue) -> Any {
        switch value {
        case .string(let string):
            return string
        case .number(let number):
            return number
        case .bool(let bool):
            return bool
        case .object(let object):
            return object.mapValues { jsonObject(from: $0) }
        case .array(let array):
            return array.map { jsonObject(from: $0) }
        case .null:
            return NSNull()
        }
    }

    private func chatMessage(from message: OpenAIChatMessage) throws -> Chat.Message {
        let role = Chat.Message.Role(rawValue: message.role) ?? .user
        let images = try message.imageReferences.map(image)
        // TODO(M6b): render prior assistant tool_calls once OpenAIChatMessage carries them.
        return Chat.Message(
            role: role,
            content: message.content,
            images: images,
            videos: try message.videoReferences.map { try media($0, kind: .video) }.map { .url($0) },
            audios: try message.audioReferences.map { try media($0, kind: .audio) }.map { .url($0) }
        )
    }

    private enum MediaKind {
        case audio
        case video

        var filePrefix: String {
            switch self {
            case .audio: return "mlxcat-audio"
            case .video: return "mlxcat-video"
            }
        }

        var fallbackExtension: String {
            switch self {
            case .audio: return "wav"
            case .video: return "mp4"
            }
        }
    }

    /// Resolve an audio/video reference to a file URL the model layer can open.
    ///
    /// `UserInput.Audio` and `UserInput.Video` take a URL (or decoded samples/frames), so
    /// an inline data URI has to become a real file. That mirrors what the legacy in-app
    /// path already does (`TextGenerationRuntime.mediaDataURIToFile`), deliberately: two
    /// engines that disagree about whether inline audio works would make the migration's
    /// own A/B comparison lie.
    ///
    /// A bad reference THROWS rather than being dropped. Silently discarding a modality
    /// is the failure mode this whole change exists to prevent — a model would answer
    /// about audio it never received, and the response would look perfectly healthy.
    private func media(_ reference: OpenAIChatMediaReference, kind: MediaKind) throws -> URL {
        switch reference.source {
        case .url(let value):
            guard let url = URL(string: value), url.scheme != nil else {
                throw NativeModelEngineError.invalidImageReference(value)
            }
            return url
        case .dataURI(let dataURI):
            return try mediaFile(fromDataURI: dataURI, kind: kind)
        }
    }

    private func mediaFile(fromDataURI dataURI: String, kind: MediaKind) throws -> URL {
        guard let commaIndex = dataURI.firstIndex(of: ",") else {
            throw NativeModelEngineError.invalidImageReference("data URI is missing media data")
        }
        let metadata = String(dataURI[..<commaIndex]).lowercased()
        guard metadata.hasPrefix("data:"), metadata.contains(";base64") else {
            throw NativeModelEngineError.invalidImageReference("data URI media must be base64 encoded")
        }
        let encoded = String(dataURI[dataURI.index(after: commaIndex)...])
        guard let data = Data(base64Encoded: encoded, options: [.ignoreUnknownCharacters]) else {
            throw NativeModelEngineError.invalidImageReference("data URI media could not be decoded")
        }
        // "data:audio/wav;base64" -> "wav". The subtype drives the extension because the
        // decoders below dispatch on it.
        var ext = kind.fallbackExtension
        if let slash = metadata.firstIndex(of: "/") {
            let afterSlash = metadata[metadata.index(after: slash)...]
            let subtype = afterSlash.prefix { $0 != ";" }
            if !subtype.isEmpty {
                ext = subtype == "mpeg" ? "mp3" : String(subtype)
            }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(kind.filePrefix)-\(UUID().uuidString).\(ext)")
        try data.write(to: url)
        return url
    }

    private func image(from reference: OpenAIChatImageReference) throws -> UserInput.Image {
        switch reference.source {
        case .dataURI(let dataURI):
            let image = try ciImage(fromDataURI: dataURI)
            return .ciImage(image)
        case .url(let value):
            guard let url = imageURL(from: value) else {
                throw NativeModelEngineError.invalidImageReference(value)
            }
            return .url(url)
        }
    }

    private func ciImage(fromDataURI dataURI: String) throws -> CIImage {
        guard let commaIndex = dataURI.firstIndex(of: ",") else {
            throw NativeModelEngineError.invalidImageReference("data URI is missing image data")
        }

        let metadata = dataURI[..<commaIndex].lowercased()
        guard metadata.hasPrefix("data:"), metadata.contains(";base64") else {
            throw NativeModelEngineError.invalidImageReference("data URI image data must be base64 encoded")
        }

        let encodedData = String(dataURI[dataURI.index(after: commaIndex)...])
        guard let data = Data(base64Encoded: encodedData, options: [.ignoreUnknownCharacters]),
            let image = CIImage(data: data)
        else {
            throw NativeModelEngineError.invalidImageReference("data URI image data could not be decoded")
        }
        return image
    }

    private func imageURL(from value: String) -> URL? {
        if let url = URL(string: value),
            let scheme = url.scheme?.lowercased(),
            ["http", "https", "file"].contains(scheme)
        {
            return url
        }
        if value.hasPrefix("/") {
            return URL(fileURLWithPath: value)
        }
        return nil
    }

    private func additionalContext(
        from request: OpenAIChatRequest,
        reasoningEffort: ReasoningEffortTemplateValue = .omitted
    ) -> [String: any Sendable]? {
        var context: [String: any Sendable] = [:]
        if let chatTemplateKwargs = request.chatTemplateKwargs {
            for (key, value) in chatTemplateKwargs where key != "reasoning_effort" {
                context[key] = sendableValue(from: value)
            }
        }
        if case .value(let value) = reasoningEffort {
            context["reasoning_effort"] = sendableValue(from: value)
        }
        if let enableThinking = request.enableThinking {
            context["enable_thinking"] = enableThinking
        } else if request.thinkingBudget != nil {
            context["enable_thinking"] = true
        }
        return context.isEmpty ? nil : context
    }

    private var usesHarmonyChannels: Bool {
        let normalized = modelID.lowercased()
        return normalized.contains("gpt-oss") || normalized.contains("harmony")
    }

    private func toolSpecDictionaries(from tools: [OpenAIJSONValue]?) -> [[String: any Sendable]]? {
        guard let tools else { return nil }

        var toolSpecs: [[String: any Sendable]] = []
        for tool in tools {
            guard case .object(let object) = tool else { continue }
            var converted: [String: any Sendable] = [:]
            for (key, value) in object {
                converted[key] = sendableValue(from: value)
            }
            toolSpecs.append(converted)
        }
        return toolSpecs.isEmpty ? nil : toolSpecs
    }

    private func sendableValue(from value: OpenAIJSONValue) -> any Sendable {
        switch value {
        case .string(let string):
            return string
        case .number(let number):
            return number
        case .bool(let bool):
            return bool
        case .object(let object):
            return object.mapValues { sendableValue(from: $0) }
        case .array(let array):
            return array.map { sendableValue(from: $0) }
        case .null:
            return Optional<String>.none as String?
        }
    }
}

enum ReasoningEffortTemplateValue: Equatable {
    case value(OpenAIJSONValue)
    case omitted
}

enum ReasoningEffortCompatibility {
    private static let aliases: [String: String] = [
        "off": "low",
        "none": "low",
        "minimal": "low",
        "moderate": "medium",
        "medium": "high",
        "high": "xhigh",
        "xhigh": "max",
        "max": "xhigh",
        "maximum": "max",
        "ultra": "max",
    ]

    private static let harmonyAliases: [String: String] = [
        "off": "low",
        "none": "low",
        "minimal": "low",
        "low": "low",
        "moderate": "medium",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "high",
        "maximum": "high",
        "ultra": "high",
    ]

    private static let namedLevels = Set(aliases.keys)

    static func candidates(
        for value: OpenAIJSONValue?,
        isHarmony: Bool
    ) -> [ReasoningEffortTemplateValue] {
        guard let value else { return [.omitted] }

        if isHarmony {
            guard case .string(let rawValue) = value,
                let mapped = harmonyAliases[rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()]
            else {
                return [.omitted]
            }
            return [.value(.string(mapped))]
        }

        let normalized = normalized(value)
        var candidates: [ReasoningEffortTemplateValue] = [.value(normalized)]
        if let fallback = fallback(for: normalized), fallback != normalized {
            candidates.append(.value(fallback))
        }
        candidates.append(.omitted)
        return candidates
    }

    private static func normalized(_ value: OpenAIJSONValue) -> OpenAIJSONValue {
        guard case .string(let rawValue) = value else { return value }
        let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if namedLevels.contains(normalized) || finiteNumber(normalized) != nil {
            return .string(normalized)
        }
        return value
    }

    private static func fallback(for value: OpenAIJSONValue) -> OpenAIJSONValue? {
        guard case .string(let string) = value else { return nil }
        if let numeric = finiteNumber(string) {
            return .number(numeric)
        }
        return aliases[string].map(OpenAIJSONValue.string)
    }

    private static func finiteNumber(_ value: String) -> Double? {
        guard let number = Double(value), number.isFinite else { return nil }
        return number
    }
}

// NOTE: Cross-model concurrent MLX eval is not process-serialized here. Each
// model has its own scheduler; a process-wide MLX eval gate is a known follow-up
// if native serving needs to mirror omlx's single executor more tightly.
public struct NativeModelLoader: EnginePoolModelLoader {
    let maxConcurrentRequests: Int

    public init(maxConcurrentRequests: Int) {
        self.maxConcurrentRequests = maxConcurrentRequests
    }

    /// Brackets the load with an MLX active-memory snapshot so the pool can
    /// account for the model by its real resident footprint rather than the
    /// discovery-time estimate. `Memory.activeMemory` counts live `MLXArray`
    /// allocations (the model weights), distinct from the reclaimable cache pool
    /// (`Memory.cacheMemory`). The pool serializes loads behind its load gate, so
    /// this process-wide delta is attributable to this one load. A non-positive
    /// delta (weights still lazy / mmapped, or a concurrent free) reports `nil`,
    /// falling the pool back to the estimate.
    public func loadModelMeasuringActualSize(
        id: String,
        modelURL: URL
    ) async throws -> EnginePoolMeasuredLoad<NativeModelEngine> {
        let activeBefore = Memory.activeMemory
        let engine = try await loadModel(id: id, modelURL: modelURL)
        let delta = Memory.activeMemory - activeBefore
        let measured: Int64? = delta > 0 ? Int64(delta) : nil
        return EnginePoolMeasuredLoad(engine: engine, actualSizeBytes: measured)
    }

    public func loadModel(id: String, modelURL: URL) async throws -> NativeModelEngine {
        let modelConfiguration = try modelConfiguration(in: modelURL)
        let modelType = modelConfiguration.modelType.lowercased()
        let isVLM = try isVLMModelDirectory(modelURL, modelType: modelType)
        let container =
            if isVLM {
                try await VLMModelFactory.shared.loadContainer(
                    from: modelURL,
                    using: #huggingFaceTokenizerLoader()
                )
            } else {
                try await LLMModelFactory.shared.loadContainer(
                    from: modelURL,
                    using: #huggingFaceTokenizerLoader()
                )
            }
        return await container.perform { context in
            NativeModelEngine(
                context: context,
                modelID: id,
                maxContextTokens: try? contextWindow(in: modelURL),
                maxConcurrentRequests: maxConcurrentRequests,
                cacheCapabilities: Self.cacheCapabilities(for: modelConfiguration),
                serializationPolicy: Self.serializationPolicy(modelType: modelType, isVLM: isVLM),
                schedulerManagedTextPrefill: !isVLM,
                chunkIdlePrefill: Self.chunksIdlePrefill(modelType: modelType),
                // Resolved per model, not per process: `gpt_oss` is excluded
                // because its quantized attention route drops attention sinks.
                kvQuantization: KVQuantizationPolicy.resolved(
                    .fromEnvironment(), modelType: modelType)
            )
        }
    }

    public func unloadModel(_ engine: NativeModelEngine, id: String) async {
        Memory.clearCache()
    }

    private func modelConfiguration(in modelURL: URL) throws -> ModelKindConfiguration {
        let configURL = modelURL.appending(component: "config.json")
        let configData = try Data(contentsOf: configURL)
        return try JSONDecoder.json5().decode(ModelKindConfiguration.self, from: configData)
    }

    func contextWindow(in modelURL: URL) throws -> Int? {
        let configURL = modelURL.appending(component: "config.json")
        let data = try Data(contentsOf: configURL)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let textConfig = object["text_config"] as? [String: Any]
        let candidates = [
            object["max_position_embeddings"],
            object["max_sequence_length"],
            object["model_max_length"],
            textConfig?["max_position_embeddings"],
            textConfig?["max_sequence_length"],
            textConfig?["model_max_length"],
        ]
        return candidates.compactMap { ($0 as? NSNumber)?.intValue }.first { $0 > 0 }
    }

    func cacheCapabilities(in modelURL: URL) throws -> ModelCacheCapabilities {
        try Self.cacheCapabilities(for: modelConfiguration(in: modelURL))
    }

    private func isVLMModelDirectory(_ modelURL: URL, modelType: String) throws -> Bool {
        if let processorClass = processorClass(in: modelURL),
            Self.vlmProcessorClasses.contains(processorClass)
        {
            return true
        }

        guard Self.vlmModelTypes.contains(modelType) else {
            return false
        }
        return hasProcessorConfiguration(in: modelURL)
    }

    /// Families whose serialization exclusion has been lifted for a measurement,
    /// comma-separated model types, or `all`.
    ///
    /// The two exclusion lists below switch batched decode off for five of the
    /// six models on the leaderboard, and that costs mlxcat its concurrency:
    /// TTFT scales 15-86x from c1 to c8 where every competitor scales 4-9x
    /// (`docs/COMPETITIVE.md`). Neither list has been re-earned. The logit-level
    /// gate that would justify them, `BatchInvarianceTests`, was pinned to
    /// Qwen3-0.6B-4bit — a model on neither list — so it had never been run
    /// against a single excluded family. Run across families on 2026-08-22,
    /// gemma-4-E2B (0.69/0.69/0.95 at batch 2/4/8) and Qwen3.5-4B
    /// (0.0/0.38/0.66) both come in UNDER the 1.25 tolerance and under
    /// Qwen3-0.6B's own 1.19, with zero mismatched wide-margin tokens.
    ///
    /// That is evidence, not a decision: this override exists so the benchmark
    /// can A/B the lists on real workloads instead of the question being settled
    /// by argument. Nothing changes by default.
    public static func serializationOverrides(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Set<String> {
        let raw = environment["MLXCAT_UNSERIALIZE_MODEL_TYPES"] ?? ""
        return Set(
            raw.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                .filter { !$0.isEmpty }
        )
    }

    public static func usesSerializedDecode(
        modelType: String,
        isVLM: Bool,
        overrides: Set<String> = serializationOverrides()
    ) -> Bool {
        serializationPolicy(modelType: modelType, isVLM: isVLM, overrides: overrides) != .never
    }

    /// How much batching a family may do.
    ///
    /// The scalar-offset families were treated as "never batch", which protects
    /// the image path by disabling batching for text chat as well. That is most
    /// of the cost: on gemma-4-E2B, serialising everything is 50x TTFT and 2.17x
    /// aggregate throughput at c4 (`docs/COMPETITIVE.md`), and its text rows are
    /// exact against serial and pass logit invariance more cleanly than the model
    /// the gate was pinned to. Its ragged IMAGE rows genuinely do break — all
    /// three rows stop at the same early token — so those keep the batch to
    /// themselves and nothing else does.
    ///
    /// `batchDecodeRegressedModelTypes` stays `.always`: what is left on that
    /// list (`qwen2`, `qwen3`) is a throughput claim, not a correctness one, and
    /// it has not been re-measured. `qwen3_moe` moved off it to
    /// `batchDecodeWidthCeilings` — its re-measurement produced a width, not a
    /// verdict. `qwen3_5` moved off too: "batched decode over its hybrid cache
    /// returns no tokens at all" was a symptom, not a cause. The cause was
    /// Qwen3.5's RoPE anchor being dropped when a second row joined, which
    /// trapped the process inside the model; with the anchor carried per row
    /// (``BatchPositionalState``) it is token-exact against serial at width 8.
    public static func serializationPolicy(
        modelType: String,
        isVLM: Bool,
        overrides: Set<String> = serializationOverrides()
    ) -> SerializationPolicy {
        let normalizedModelType = modelType.lowercased()
        if overrides.contains("all") || overrides.contains(normalizedModelType) {
            return .never
        }
        let ceiling = batchDecodeWidthCeilings[normalizedModelType]
        if batchDecodeRegressedModelTypes.contains(normalizedModelType) {
            return .always
        }
        if isVLM, scalarOffsetVLMModelTypes.contains(normalizedModelType) {
            guard multimodalOnlyModelTypes.contains(normalizedModelType) else { return .always }
            return .batched(maxWidth: ceiling, multimodalSolitude: true)
        }
        if let ceiling {
            return .maxWidth(ceiling)
        }
        return .never
    }

    /// Scalar-offset families whose TEXT rows are proven safe to batch, so only
    /// their multimodal rows are serialized. Entry here requires both gates:
    /// `BatchInvarianceTests.testBatchInvarianceAcrossModelFamilies` (logits) and
    /// `SchedulerEngineTests.testVLMBatchEqualityAcrossModels` (images). With
    /// `multimodalSolitude` the image gate is about a path that no longer
    /// batches, so what actually earns entry is the numeric one — stated here
    /// because the sentence above predates solitude and reads stricter than the
    /// policy it describes.
    ///
    /// `qwen3_5` (Qwen3.5-4B and Qwen3.8-27B) joined on 2026-08-23 with evidence
    /// stronger than the tolerance: `PositionalStateBatchIntegrationTests` holds
    /// it token-EXACT against serial at 4 ragged rows and again at width 8, on
    /// the `VLMModelFactory` path production actually loads. Its scalar offset is
    /// corrected per row by ``BatchPositionalState`` rather than tolerated.
    ///
    /// **`gemma4` and `gemma4_unified` were REMOVED on 2026-08-24.** They were
    /// granted entry on the morning of 2026-08-23 and it produced the biggest
    /// result on the board — gemma-4-E2B longgen c8 TTFT 39,346 -> 1,793 ms.
    /// The grant was wrong, and not by a tolerance: batched gemma decode
    /// produces **unrotated** queries and keys for every row but the first.
    ///
    /// MLX's RoPE has a fast path taken when the input is row-contiguous, the
    /// sequence length is 1, and the offset is a SCALAR
    /// (`mlx/backend/metal/rope.cpp:96`). That branch dispatches
    /// `MTL::Size(dims/2, N, 1)`, where `N` covers only the head axes — the
    /// batch size is absent (`rope.cpp:134-139`), while the general branch has
    /// it at `:149`. The kernel therefore rotates batch row 0 and, because the
    /// input is donated, rows 1..B-1 pass through untouched.
    ///
    /// Gemma's VLM attention is the only code in our stack that reaches that
    /// primitive with a scalar offset on a batched decode tensor
    /// (`Gemma4.swift:872,880,903`); every other family rotates through
    /// `applyRotaryPosition(..., offset: cache?.ropeOffset)`, and a per-row
    /// array offset forces the correct branch. `RoPEBatchGridProbeTests`
    /// reproduces it in under a second with no model:
    /// `rows-unrotated=[1, 2, 3]`, output bit-identical to input.
    ///
    /// So this was not a latency-versus-exactness trade — width >= 2 gemma was
    /// wrong. `.always` was a correctness floor, not conservatism, and it held
    /// until the RoPE call was fixed upstream or in a fork.
    ///
    /// **RE-GRANTED later on 2026-08-24: both defects are fixed, in forks.**
    /// The kernel: mlx-swift is pinned to atlas-open-sources/mlx-swift, whose
    /// vendored mlx carries the three-line backport of ml-explore/mlx
    /// `76a977ca` (#3498) — the fast-path grid gets its batch axis back
    /// (`RoPEBatchGridProbeTests`: `rows-unrotated=[]`). The anchor:
    /// mlx-swift-lm is pinned to atlas-open-sources/mlx-swift-lm `7b93094e`,
    /// where `Gemma4TextAttention` rotates with the per-row `cache?.ropeOffset`
    /// instead of the padded batch length. Evidence on the production factory
    /// (`GemmaRoPEOffsetProbeTests`, real chat-templated prompts, greedy):
    /// serial-exact at ragged width 2 and ragged width 4 — the arms the revert
    /// was made on. The residue is one late-step row flip on equal-length rows
    /// (unrotated gemma diverged at step 0 on 3 of 4 rows; this is the near-tie
    /// batch-variance every serving engine on non-invariant kernels shows, and
    /// the same category the logit sweep measured at `mismatched = 0`).
    private static let multimodalOnlyModelTypes: Set<String> = [
        "qwen3_5",
        "gemma4",
        "gemma4_unified",
    ]

    private static func cacheCapabilities(for configuration: ModelKindConfiguration) -> ModelCacheCapabilities {
        ModelCacheCapabilities(
            usesWindowedKVCache: usesWindowedKVCache(configuration: configuration),
            kvCacheProfile: configuration.kvCacheProfile
        )
    }

    private static func usesWindowedKVCache(configuration: ModelKindConfiguration) -> Bool {
        let modelType = configuration.modelType.lowercased()
        if windowedKVModelTypes.contains(modelType) {
            return true
        }
        if configuration.useSlidingWindow == false {
            return false
        }
        if let slidingWindow = configuration.slidingWindow, slidingWindow > 0 {
            return true
        }
        if let attentionChunkSize = configuration.attentionChunkSize, attentionChunkSize > 0 {
            return true
        }
        if configuration.layerTypes.contains(where: { $0.localizedCaseInsensitiveContains("sliding") }) {
            return true
        }
        return false
    }

    private func processorClass(in modelURL: URL) -> String? {
        for url in processorConfigurationURLs(in: modelURL) {
            guard FileManager.default.fileExists(atPath: url.path) else {
                continue
            }
            guard let data = try? Data(contentsOf: url),
                let config = try? JSONDecoder.json5().decode(ProcessorKindConfiguration.self, from: data)
            else {
                continue
            }
            return config.processorClass
        }
        return nil
    }

    private func hasProcessorConfiguration(in modelURL: URL) -> Bool {
        processorConfigurationURLs(in: modelURL).contains { url in
            FileManager.default.fileExists(atPath: url.path)
        }
    }

    private func processorConfigurationURLs(in modelURL: URL) -> [URL] {
        [
            modelURL.appending(component: "preprocessor_config.json"),
            modelURL.appending(component: "processor_config.json"),
        ]
    }

    private struct ModelKindConfiguration: Decodable {
        let modelType: String
        let slidingWindow: Int?
        let useSlidingWindow: Bool?
        let attentionChunkSize: Int?
        let layerTypes: [String]
        let kvCacheProfile: ModelKVCacheProfile?

        enum CodingKeys: String, Swift.CodingKey {
            case modelType = "model_type"
            case slidingWindow = "sliding_window"
            case useSlidingWindow = "use_sliding_window"
            case attentionChunkSize = "attention_chunk_size"
            case layerTypes = "layer_types"
            case hiddenLayerCount = "num_hidden_layers"
            case attentionHeadCount = "num_attention_heads"
            case keyValueHeadCount = "num_key_value_heads"
            case headDimension = "head_dim"
            case hiddenSize = "hidden_size"
            case torchDType = "torch_dtype"
            case dtype
            case textConfig = "text_config"
        }

        init(from decoder: Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try container.decode(String.self, forKey: .modelType)
            self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
            self.useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow)
            self.attentionChunkSize = try container.decodeIfPresent(Int.self, forKey: .attentionChunkSize)
            self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
            let textConfig = try container.decodeIfPresent(TextConfiguration.self, forKey: .textConfig)
            let hiddenLayerCount = try container.decodeIfPresent(Int.self, forKey: .hiddenLayerCount)
                ?? textConfig?.hiddenLayerCount
            let attentionHeadCount = try container.decodeIfPresent(Int.self, forKey: .attentionHeadCount)
                ?? textConfig?.attentionHeadCount
            let keyValueHeadCount = try container.decodeIfPresent(Int.self, forKey: .keyValueHeadCount)
                ?? textConfig?.keyValueHeadCount
            let headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
                ?? textConfig?.headDimension
            let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? textConfig?.hiddenSize
            let dtype = try container.decodeIfPresent(String.self, forKey: .torchDType)
                ?? container.decodeIfPresent(String.self, forKey: .dtype)
                ?? textConfig?.dtype
            self.kvCacheProfile = ModelKVCacheProfile(
                hiddenLayerCount: hiddenLayerCount,
                attentionHeadCount: attentionHeadCount,
                keyValueHeadCount: keyValueHeadCount,
                headDimension: headDimension,
                hiddenSize: hiddenSize,
                scalarByteCount: ModelKVCacheProfile.scalarByteCount(forDType: dtype)
            )
        }
    }

    private struct TextConfiguration: Decodable {
        let hiddenLayerCount: Int?
        let attentionHeadCount: Int?
        let keyValueHeadCount: Int?
        let headDimension: Int?
        let hiddenSize: Int?
        let dtype: String?

        enum CodingKeys: String, Swift.CodingKey {
            case hiddenLayerCount = "num_hidden_layers"
            case attentionHeadCount = "num_attention_heads"
            case keyValueHeadCount = "num_key_value_heads"
            case headDimension = "head_dim"
            case hiddenSize = "hidden_size"
            case torchDType = "torch_dtype"
            case dtype
        }

        init(from decoder: Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.hiddenLayerCount = try container.decodeIfPresent(Int.self, forKey: .hiddenLayerCount)
            self.attentionHeadCount = try container.decodeIfPresent(Int.self, forKey: .attentionHeadCount)
            self.keyValueHeadCount = try container.decodeIfPresent(Int.self, forKey: .keyValueHeadCount)
            self.headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            self.dtype = try container.decodeIfPresent(String.self, forKey: .torchDType)
                ?? container.decodeIfPresent(String.self, forKey: .dtype)
        }
    }

    private struct ProcessorKindConfiguration: Decodable {
        let processorClass: String

        enum CodingKeys: String, CodingKey {
            case processorClass = "processor_class"
        }
    }

    private static let vlmModelTypes: Set<String> = [
        "paligemma",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "idefics3",
        "gemma3",
        "gemma4",
        // Unified multimodal Gemma 4 (e.g. gemma-4-12B). Routes to the VLM
        // Gemma4Unified, which has the language_model.* weight sanitizer; the
        // LLM Gemma4Model expects flat model.layers.* and 500s with keyNotFound.
        "gemma4_unified",
        "smolvlm",
        "fastvlm",
        "llava_qwen2",
        "pixtral",
        "mistral3",
        "lfm2_vl",
        "lfm2-vl",
        "glm_ocr",
    ]

    private static let scalarOffsetVLMModelTypes: Set<String> = [
        // These VLMs still derive MRoPE position ids, mask lengths, attention
        // scaling, or shared-KV RoPE offsets from scalar cache.offset values.
        // Membership means "needs a policy", not "must serialize": `qwen3_5` is
        // still listed and still scalar-offset, but the batch generator now
        // corrects that scalar per row, so its policy is multimodal solitude
        // rather than `.always`.
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "gemma4",
        "gemma4_unified",
        "mistral3",
        // GlmOcr builds scalar MRoPE position ids from cache.offset in
        // GlmOcr.swift:308, :356, and :383.
        "glm_ocr",
    ]

    // The stated reason for this list is WRONG, and the list is still right.
    //
    // It said: "explicit-demand two-request benchmarks showed lower aggregate
    // throughput for these model families under batched decode". Re-measured on
    // 2026-08-23 at the longgen tier, Qwen3-Coder-30B (`qwen3_moe`) c4:
    //
    //     serialized  TTFT 44,760 ms · prefill 24 tok/s  · aggregate 30.0 tok/s
    //     batched     TTFT  4,089 ms · prefill 254 tok/s · aggregate 54.5 tok/s
    //
    // Aggregate throughput is 82% HIGHER batched, not lower. The original claim
    // came from two requests at a short tier, where a ~1 s serial prefill
    // dominates the wall of a 128-token generation — so it measured admission
    // latency and called it throughput. bench/README.md documents that trap now.
    //
    // Keep the list anyway, for a reason nobody had checked: batched decode for
    // `qwen3_moe` breaks the LOGIT tolerance at width >= 4.
    //
    //     batch 2  maxLogitError 0.94   ok
    //     batch 4  maxLogitError 2.69   over the 1.25 tolerance
    //     batch 8  maxLogitError 2.69   over the 1.25 tolerance
    //
    // For scale, gpt-oss-20b — batching enabled, not on this list — measures
    // 1.7e-05 at the same widths. Correct batched decode is essentially exact;
    // this is five orders of magnitude away from it. No wide-margin token
    // actually flipped, so the output may well be fine in practice, but that is
    // not the standard the other families are held to.
    //
    // So the throughput justification is retired and a correctness one replaces
    // it. `qwen2` and `qwen3` have NOT been re-measured on either axis — they
    // inherit the old, now-discredited reason and are due the same treatment.
    private static let batchDecodeRegressedModelTypes: Set<String> = [
        "qwen2",
        "qwen3",
    ]

    /// Families that batch correctly only up to a width, with the width they
    /// have actually been measured at.
    ///
    /// `qwen3_moe` is the whole list, and it is here rather than in
    /// `batchDecodeRegressedModelTypes` because the measurement above does not
    /// say "this family cannot batch" — it says width 2 is inside the 1.25
    /// tolerance (0.94) and widths 4 and 8 are not (2.69). Refusing width 2 as
    /// well threw away the half of the win that is earned: serialized decode
    /// measures 10.9x TTFT and 0.55x aggregate throughput against batched on
    /// this family. A ceiling takes what the numbers support and nothing more.
    ///
    /// Raising an entry requires re-running
    /// `BatchInvarianceTests.testBatchInvarianceAcrossModelFamilies` (logits,
    /// via `MLXCAT_BATCH_INVARIANCE_MODELS`) and `MoEBatchIntegrationTests`
    /// (emitted tokens through the serving engine, via
    /// `MLXSERVE_MOE_TEST_MODEL`) at the new width. Removing the entry entirely
    /// means the width->=4 numerics were fixed; the win waiting there is the
    /// rest of that 10.9x.
    /// Whether an IDLE admission chunks its prefill forward pass.
    ///
    /// Chunking removes a whole-prompt `[1, L, vocab]` logits tensor and its
    /// full-length attention scratch. On its own that was a 26-40% cut on three
    /// families and a 63% REGRESSION on a fourth, which looked like a trade
    /// needing a per-family exemption list — and was not. The missing piece was
    /// read out of the reference rather than benchmarked into existence:
    /// `mlx-lm` calls `mx.clear_cache()` at the end of every prefill chunk
    /// (`guest/mlx-lm/mlx_lm/generate.py:451`, `:586`). Each chunk attends over
    /// a longer key range than the last, so no two allocate the same
    /// attention-scratch shape and MLX's buffer cache can never reuse one; it
    /// simply accumulates every shape for the whole prefill. Freeing between
    /// chunks is what makes chunked prefill cheap for them, and it was the
    /// entire difference. Measured at 16k on an M5 Max, after load -> peak:
    ///
    ///     model                  single-pass   chunked   chunked+clear
    ///     Qwen3.8-27B              53.02        39.02      18.90   -64%
    ///     gemma-4-12B              34.51        20.74      13.42   -61%
    ///     gpt-oss-20b              35.17        25.37      14.69   -58%
    ///     Qwen3-Coder-30B-A3B      24.75        40.34      19.71   -20%
    ///
    /// Every family wins, so `singlePassPrefillModelTypes` is EMPTY. It is kept
    /// rather than deleted because the mechanism that would refill it is
    /// understood: a family whose chunked scratch cannot be freed between chunks
    /// would belong here. Peak is now 1.22-1.33x loaded weights across all four,
    /// against 1.5-3.7x before — the consistency is the tell that this addressed
    /// the mechanism and not a symptom.
    ///
    /// `MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES` overrides the list (`all` for
    /// every family) so the benchmark can A/B it without a rebuild.
    static func chunksIdlePrefill(
        modelType: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let normalized = modelType.lowercased()
        let raw = environment["MLXCAT_SINGLE_PASS_PREFILL_MODEL_TYPES"] ?? ""
        let overrides = Set(
            raw.split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                .filter { !$0.isEmpty }
        )
        if !overrides.isEmpty {
            return !(overrides.contains("all") || overrides.contains(normalized))
        }
        return !singlePassPrefillModelTypes.contains(normalized)
    }

    /// Families measured to peak HIGHER with a chunked idle prefill than with a
    /// single whole-prompt pass. Empty since the per-chunk `Memory.clearCache()`
    /// landed — see `chunksIdlePrefill` for the numbers that emptied it.
    private static let singlePassPrefillModelTypes: Set<String> = []

    private static let batchDecodeWidthCeilings: [String: Int] = [
        // RAISED 2 -> 8 on 2026-08-24, on a complete evidence pack re-measured
        // that day (M4 Pro, Qwen3-Coder-30B-A3B-Instruct-4bit):
        //
        //   * `MoEBatchIntegrationTests` — emitted tokens through the engine —
        //     PASSES with strict serial-greedy equality at widths 2, 4 AND 8.
        //   * `MoEWidthNumericsProbeTests` (the discriminator that settled
        //     gemma, KNOWN-FAILURES 1f): IDENTICAL rows decoded together are
        //     token-identical to serial at widths 2/4/8 — vsSerial=0 and
        //     vsEachOther=0 everywhere, so rows are NOT crossing. Real ragged
        //     prompts: 1 late flip (step 7 of 12) in 8 rows — the near-tie
        //     batch-variance category, the same residue gemma's grant carries.
        //   * `BatchInvarianceTests` still reads maxLogitError 2.6875 at
        //     widths 4/8 (0.5 at width 2, 0.0 at width 1) against the 1.25
        //     tolerance, with zero wide-margin flips and margin 6.5 at the
        //     max-error step. With contamination excluded by the probe, that
        //     number is width-level accumulation order in the expert mix — an
        //     MoE reduction reordering with width, not a defect a per-row fix
        //     could touch.
        //
        // The old width-2 entry guarded against exactly the contamination the
        // probe now rules out; what remains over the tolerance is explained
        // and token-invisible on every end-to-end gate. 8 is the widest width
        // MEASURED, so 8 is the ceiling — "what the numbers support and
        // nothing more". This cap was ~2/3 of the whole c8 board gap (the
        // `mlxcat-moe-uncapped` arm priced it at +46-48% aggregate on the
        // coder cells); deleting the entry outright (unbounded) waits on a
        // width-16+ measurement, not on more adjudication.
        "qwen3_moe": 8,
        // gemma4_unified is here for a different reason than qwen3_moe, and the
        // difference is the whole argument.
        //
        // `multimodalOnlyModelTypes` documents entry as requiring both the logit
        // gate and the image gate. The logit evidence on record is gemma-4-E2B —
        // a `gemma4`. gemma-4-12B, the `gemma4_unified` that shares the grant,
        // had never been run through it. Run 2026-08-23 on an M5 Max, with a
        // width-1 CONTROL arm added so a path difference could be told from a
        // batching defect:
        //
        //     batch 1  maxLogitError 0.0       margin  0.0    (exact; path cleared)
        //     batch 2  maxLogitError 1.65625   margin 11.5    over 1.25
        //     batch 4  maxLogitError 1.65625   margin 11.5    over 1.25
        //     batch 8  maxLogitError 1.7402    margin  1.125  over 1.25, and OVER MARGIN
        //
        // Width 1 being bit-exact means this is rows contaminating each other,
        // not the batched path's own numerics — the defect is real. But the
        // widths are not equally dangerous. A token can only flip when the error
        // exceeds the top-1/top-2 margin; at widths 2 and 4 the margin is 7x the
        // error, and at width 8 the error EXCEEDS the margin. `mismatched` is 0
        // everywhere, so nothing has flipped yet — at width 8 that is luck, and
        // at widths 2 and 4 it is headroom.
        //
        // A ceiling of 4 removes the width where a flip is possible and keeps the
        // one the concurrency win was measured at (c4 TTFT 11,999 -> 870 ms,
        // `d630cee`). It is strictly safer than today, which is unbounded.
        //
        // It does NOT clear the 1.25 tolerance at any width >= 2, and that is
        // still an open defect: gemma-4 is the iOS flagship and `gemma4_unified`
        // is a distinct loader path from `gemma4`, which passes at 0.69. Whether
        // the answer is to fix that path or to drop to 1 (i.e. `.always`) is
        // Phil's call — see docs/KNOWN-FAILURES.md 1e.
        "gemma4_unified": 4,
    ]

    private static let windowedKVModelTypes: Set<String> = [
        "gemma3",
        "gemma4",
        "gemma4_unified",
        "gpt_oss",
    ]

    private static let vlmProcessorClasses: Set<String> = [
        "PaliGemmaProcessor",
        "Qwen2VLProcessor",
        "Qwen2_5_VLProcessor",
        "Qwen3VLProcessor",
        "Idefics3Processor",
        "Gemma3Processor",
        "Gemma4Processor",
        "Gemma4UnifiedProcessor",
        "SmolVLMProcessor",
        "FastVLMProcessor",
        "PixtralProcessor",
        "Mistral3Processor",
        "Lfm2VlProcessor",
        "Glm46VProcessor",
    ]
}

private enum NativeModelEngineError: Error, CustomStringConvertible {
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
