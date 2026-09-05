import Foundation
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Tokenizers
import MLXCatHTTP

/// The container owns the real tokenizer, embedding model, and pooling layer.
/// Pool leases keep it resident until perform has returned evaluated CPU vectors.
actor NativeEmbeddingModelCache: NativeEmbeddingModelServing {
    private var container: EmbedderModelContainer?

    func load(modelURL: URL) async throws {
        container = try await EmbedderModelFactory.shared.loadContainer(
            from: modelURL, using: #huggingFaceTokenizerLoader()
        )
    }

    func unload() { container = nil }

    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> OpenAIEmbeddingsResult {
        guard let container else {
            throw OpenAIHTTPError(status: 500, message: "Embedding model is not loaded.")
        }
        let inputs = request.input.values
        return await container.perform { context in
            var embeddings: [[Float]] = []
            var promptTokens = 0
            embeddings.reserveCapacity(inputs.count)

            for input in inputs {
                let tokens = context.tokenizer.encode(text: input, addSpecialTokens: true)
                promptTokens += tokens.count
                let modelInput = MLXArray(tokens)[.newAxis, 0...]
                let output = context.model(
                    modelInput,
                    positionIds: nil,
                    tokenTypeIds: nil,
                    attentionMask: nil
                )
                let pooled = context.pooling(output, normalize: true)
                pooled.eval()
                embeddings.append(pooled[0].asArray(Float.self))
            }

            return OpenAIEmbeddingsResult(embeddings: embeddings, promptTokens: promptTokens)
        }
    }
}
