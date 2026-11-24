import Foundation
import Hub
import Jinja
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Loads a local tokenizer using the pack’s vocabulary and template.
package struct NativeTokenizerLoader: TokenizerLoader {
    package init() {}

    package func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelFolder: directory)
        guard var tokenizerConfig = try await config.tokenizerConfig else {
            throw Tokenizers.TokenizerError.missingConfig
        }
        if tokenizerConfig.tokenizerClass.string() == "Qwen3_5Tokenizer" {
            var fields = tokenizerConfig.dictionary(or: [:])
            fields["tokenizer_class"] = Config("Qwen2Tokenizer")
            tokenizerConfig = Config(fields)
        }
        let tokenizer = try await PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig, tokenizerData: config.tokenizerData
        )
        let bridge = #adaptHuggingFaceTokenizer(tokenizer)
        switch try await config.modelConfig?.modelType.string() {
        case "qwen3_5", "qwen3_5_moe":
            return QwenToolTokenizer(upstream: bridge)
        default:
            return bridge
        }
    }
}

private struct QwenToolTokenizer: MLXLMCommon.Tokenizer {
    let upstream: any MLXLMCommon.Tokenizer

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var context = additionalContext
        if let tools, context?["tools"] == nil {
            context = context ?? [:]
            context?["tools"] = try Jinja.Value.array(tools.map(NativeQwenToolSchema.value))
        }
        return try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: context)
    }
}

package enum NativeQwenToolSchema {
    // ToolSpec is a Swift dictionary. Restore the OpenAI declaration order
    // used by the reference for known fields, with stable ordering for extras.
    package static func value(_ tool: [String: any Sendable]) throws -> Jinja.Value {
        try orderedObject(tool, first: ["type", "function"]) { key, value in
            guard key == "function", let function = value as? [String: any Sendable] else {
                return try Jinja.Value(any: value)
            }
            return try orderedObject(function, first: ["name", "description", "parameters", "strict"]) {
                key, value in
                try key == "parameters" ? schema(value) : Jinja.Value(any: value)
            }
        }
    }

    private static func schema(_ value: any Sendable) throws -> Jinja.Value {
        guard let object = value as? [String: any Sendable] else {
            if let array = value as? [any Sendable] {
                return try .array(array.map(schema))
            }
            return try Jinja.Value(any: value)
        }
        return try orderedObject(object, first: ["type", "description", "properties", "required", "additionalProperties"]) {
            key, value in
            if key == "properties", let properties = value as? [String: any Sendable] {
                return try orderedObject(properties, first: []) { _, value in try schema(value) }
            }
            return try schema(value)
        }
    }

    private static func orderedObject(
        _ object: [String: any Sendable],
        first: [String],
        transform: (String, any Sendable) throws -> Jinja.Value
    ) throws -> Jinja.Value {
        var fields = OrderedDictionary<String, Jinja.Value>()
        for key in first + object.keys.filter({ !first.contains($0) }).sorted() {
            if let value = object[key] {
                fields[key] = try transform(key, value)
            }
        }
        return .object(fields)
    }
}
