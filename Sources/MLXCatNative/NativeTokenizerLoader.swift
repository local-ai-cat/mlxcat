import Foundation
import Hub
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
        return #adaptHuggingFaceTokenizer(tokenizer)
    }
}
