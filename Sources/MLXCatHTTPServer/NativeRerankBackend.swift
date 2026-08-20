import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import Tokenizers

final class NativeRerankBackend: OpenAIRerankBackend, @unchecked Sendable {
    let rerankModels: [OpenAIModelInfo]

    private let modelURLsByID: [String: URL]
    private let cache: NativeRerankModelCache

    init(models: [String: DiscoveredModel], memoryCeilingBytes: Int64 = 0) {
        self.modelURLsByID = models.mapValues(\.modelURL)
        self.rerankModels = models.keys.sorted().map { OpenAIModelInfo(id: $0, maxModelLength: nil) }
        self.cache = NativeRerankModelCache(
            estimatedSizesByID: models.mapValues(\.estimatedSize),
            memoryCeilingBytes: memoryCeilingBytes
        )
    }

    func rerank(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult {
        guard let modelURL = modelURLsByID[request.model] else {
            throw OpenAIHTTPError(status: 404, message: "Rerank model '\(request.model)' not found.")
        }
        let container = try await cache.container(for: request.model, modelURL: modelURL)
        return try await NativeRerankScorer.score(request, container: container)
    }

    static func isRerankModelDirectory(_ url: URL) -> Bool {
        let name = url.lastPathComponent.lowercased()
        let hasRerankNameHint = name.contains("rerank") || name.contains("reranker")
        guard
            let data = try? Data(contentsOf: url.appendingPathComponent("config.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return hasRerankNameHint
        }

        let modelType = (object["model_type"] as? String)?.lowercased() ?? ""
        let architectures = (object["architectures"] as? [String] ?? []).map { $0.lowercased() }
        if architectures.contains(where: { architecture in
            architecture.contains("sequenceclassification")
                || architecture == "jinaforranking"
                || architecture == "bertforsequenceclassification"
                || architecture == "qwen3forsequenceclassification"
        }) {
            return true
        }
        return hasRerankNameHint
            && (modelType.contains("qwen")
                || architectures.contains(where: {
                    $0.contains("causallm") || $0 == "qwen3vlforconditionalgeneration"
                }))
    }
}

private actor NativeRerankModelCache {
    private let estimatedSizesByID: [String: Int64]
    private let memoryCeilingBytes: Int64
    private var state = NativeRerankCacheState()
    private var cached: (id: String, container: ModelContainer)?

    init(estimatedSizesByID: [String: Int64], memoryCeilingBytes: Int64) {
        self.estimatedSizesByID = estimatedSizesByID
        self.memoryCeilingBytes = memoryCeilingBytes
    }

    func container(for id: String, modelURL: URL) async throws -> ModelContainer {
        if let cached, cached.id == id {
            return cached.container
        }
        // Auxiliary model classes do not yet have full audio_stt-style pool citizenship.
        // Until they do, rerank is capped to one resident model and preflights the same
        // memory ceiling used by the main model pool.
        _ = try state.prepareLoad(
            id: id,
            estimatedSize: estimatedSizesByID[id] ?? 0,
            memoryCeilingBytes: memoryCeilingBytes
        )
        cached = nil
        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: modelURL,
                using: #huggingFaceTokenizerLoader()
            )
            cached = (id, container)
            state.finishLoad(id: id)
            return container
        } catch {
            state.clear()
            throw error
        }
    }
}

struct NativeRerankCacheState: Equatable {
    private(set) var loadedModelID: String?

    mutating func prepareLoad(
        id: String,
        estimatedSize: Int64,
        memoryCeilingBytes: Int64
    ) throws -> String? {
        if loadedModelID == id {
            return nil
        }
        if memoryCeilingBytes > 0 && estimatedSize > memoryCeilingBytes {
            throw OpenAIHTTPError(
                status: 507,
                message: EnginePoolError.modelTooLarge(
                    id: id,
                    size: estimatedSize,
                    ceiling: memoryCeilingBytes
                ).description
            )
        }
        let evicted = loadedModelID
        loadedModelID = nil
        return evicted
    }

    mutating func finishLoad(id: String) {
        loadedModelID = id
    }

    mutating func clear() {
        loadedModelID = nil
    }
}

private enum NativeRerankScorer {
    private static let maxLength = 8_192
    private static let systemPrompt =
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    private static let instruction =
        "Given a web search query, retrieve relevant passages that answer the query"
    private static let suffix = "<think>\n\n</think>\n\n"
    private static let sentinel = "<<__MLXSERVE_RERANK_CONTENT__>>"

    static func score(_ request: OpenAIRerankRequest, container: ModelContainer) async throws -> OpenAIRerankResult {
        try await container.perform { context in
            guard
                let yesTokenID = context.tokenizer.convertTokenToId("yes"),
                let noTokenID = context.tokenizer.convertTokenToId("no")
            else {
                throw OpenAIHTTPError(status: 400, message: "rerank tokenizer is missing yes/no token IDs")
            }

            let template = try promptTemplate(tokenizer: context.tokenizer)
            var scores: [Float] = []
            var totalTokens = 0
            scores.reserveCapacity(request.documents.count)

            for document in request.documents {
                let content = "<Instruct>: \(instruction)\n<Query>: \(request.query.text)\n<Document>: \(document.text)"
                let contentTokens = context.tokenizer.encode(text: content, addSpecialTokens: false)
                let budget = max(0, maxLength - template.prefix.count - template.suffix.count)
                let promptTokens = template.prefix + contentTokens.prefix(budget) + template.suffix
                totalTokens += promptTokens.count

                let input = MLXArray(Array(promptTokens))[.newAxis, 0...]
                let output = context.model(input, cache: nil)
                let logits = output[0, -1, 0...]
                logits.eval()
                let noLogit = Double(logits[noTokenID].item(Float.self))
                let yesLogit = Double(logits[yesTokenID].item(Float.self))
                scores.append(Float(softmaxYes(noLogit: noLogit, yesLogit: yesLogit)))
            }

            let sortedIndices = scores.indices.sorted {
                if scores[$0] == scores[$1] {
                    return $0 < $1
                }
                return scores[$0] > scores[$1]
            }
            let topIndices = applyRerankTopN(sortedIndices, topN: request.topN)
            return OpenAIRerankResult(scores: scores, indices: topIndices, totalTokens: totalTokens)
        }
    }

    private static func promptTemplate(tokenizer: any MLXLMCommon.Tokenizer) throws -> (prefix: [Int], suffix: [Int]) {
        let templateTokens = try tokenizer.applyChatTemplate(
            messages: [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": sentinel],
            ],
            tools: nil,
            additionalContext: ["add_generation_prompt": true]
        )
        let sentinelTokens = tokenizer.encode(text: sentinel, addSpecialTokens: false)
        guard
            !sentinelTokens.isEmpty,
            let range = templateTokens.firstRange(of: sentinelTokens)
        else {
            throw OpenAIHTTPError(status: 400, message: "rerank tokenizer chat template could not be prepared")
        }

        let suffixTokens = tokenizer.encode(text: suffix, addSpecialTokens: false)
        return (
            prefix: Array(templateTokens[..<range.lowerBound]),
            suffix: Array(templateTokens[range.upperBound...]) + suffixTokens
        )
    }

    private static func softmaxYes(noLogit: Double, yesLogit: Double) -> Double {
        let maximum = max(noLogit, yesLogit)
        let noExp = exp(noLogit - maximum)
        let yesExp = exp(yesLogit - maximum)
        return yesExp / (noExp + yesExp)
    }
}

private extension Array where Element: Equatable {
    func firstRange(of needle: [Element]) -> Range<Int>? {
        guard !needle.isEmpty, needle.count <= count else {
            return nil
        }
        for start in 0...(count - needle.count) {
            if Array(self[start..<(start + needle.count)]) == needle {
                return start..<(start + needle.count)
            }
        }
        return nil
    }
}
