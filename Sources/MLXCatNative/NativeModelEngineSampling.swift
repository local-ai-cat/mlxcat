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

extension NativeModelEngine {
    func samplingParameters(from request: OpenAIChatRequest) throws -> SamplingParameters {
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

    func samplingParameters(from request: OpenAICompletionRequest) throws -> SamplingParameters {
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
        let postToolAllowedTokenIDs = postToolTokens(vocabulary: vocabulary, startTokenID: startTokenID)

        return QwenXMLToolGrammarConfiguration(
            vocabulary: vocabulary,
            toolNames: toolNames,
            toolCallStartTokenID: startTokenID,
            toolCallEndTokenID: endTokenID,
            postToolAllowedTokenIDs: Array(postToolAllowedTokenIDs),
            denyIndex: denyIndex
        )
    }

    private func postToolTokens(vocabulary: JSONGrammarVocabulary, startTokenID: Int) -> Set<Int> {
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

        return postToolAllowedTokenIDs
    }

    private func singleTokenID(for text: String) -> Int? {
        tokenizer.convertTokenToId(text)
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
            guard tokenizer.convertIdToToken(id) != nil else {
                consecutiveMisses += 1
                continue
            }
            consecutiveMisses = 0
            if eosTokenIds.contains(id) {
                tokens.append(JSONGrammarToken(id: id, text: "", isEOS: true))
                continue
            }
            let text = tokenizer.decode(tokenIds: [id], skipSpecialTokens: true)
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
            var sequence = tokenizer.encode(text: choice, addSpecialTokens: false)
            guard !sequence.isEmpty else { return nil }
            if let eosTokenId {
                sequence.append(eosTokenId)
            }
            return sequence
        }
        return sequences.isEmpty ? nil : sequences
    }

    func thinkingBudgetConfiguration(from request: OpenAIChatRequest) -> ServeThinkingBudgetConfiguration? {
        guard let budget = request.thinkingBudget else { return nil }

        let closeTokenIDs: [Int]
        let trailingTokenIDs: [Int]
        if usesHarmonyChannels {
            closeTokenIDs = tokenizer.encode(text: "<|end|>", addSpecialTokens: false)
            trailingTokenIDs = tokenizer.encode(
                text: "<|start|>assistant<|channel|>final<|message|>",
                addSpecialTokens: false
            )
        } else {
            closeTokenIDs = tokenizer.encode(text: "</think>", addSpecialTokens: false)
            trailingTokenIDs = []
        }

        guard !closeTokenIDs.isEmpty else { return nil }
        return ServeThinkingBudgetConfiguration(
            budget: budget,
            closeTokenIDs: closeTokenIDs,
            startTokenIDs: tokenizer.encode(text: "<think>", addSpecialTokens: false),
            trailingTokenIDs: trailingTokenIDs,
            startsInThinking: true
        )
    }

    func canonicalEOSTokenId() -> Int? {
        // Choice constraints are terminal only when the choice trie includes EOS.
        // If a model exposes no EOS id, fall back to bare choices and let maxTokens stop.
        tokenizer.eosTokenId ?? eosTokenIds.first
    }

}
