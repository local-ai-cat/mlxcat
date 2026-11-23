import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One cache object per execution visit couples the KV timeline with the four
/// convolution histories that must advance at the same rate.
final class InklingVisitCache: KVCache {
    var visit: InklingTextModel.VisitState
    let taps: Int
    let maxSize: Int?

    init(taps: Int, window: Int?) {
        self.taps = taps
        self.maxSize = window
        visit = .init(taps: taps)
    }

    var offset: Int { visit.attention.offset }
    var isTrimmable: Bool { false }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        visit.attention.append(keys: keys, values: values, window: maxSize)
    }

    var state: [MLXArray] {
        get {
            guard let keys = visit.attention.keys, let values = visit.attention.values,
                let keyHistory = visit.attention.key.history,
                let valueHistory = visit.attention.value.history,
                let attentionHistory = visit.attentionOutput.history,
                let mlpHistory = visit.mlpOutput.history
            else { return [] }
            return [keys, values, keyHistory, valueHistory, attentionHistory, mlpHistory]
        }
        set {
            if newValue.isEmpty {
                visit = .init(taps: taps)
                return
            }
            precondition(newValue.count == 6)
            visit.attention.keys = newValue[0]
            visit.attention.values = newValue[1]
            visit.attention.key.history = newValue[2]
            visit.attention.value.history = newValue[3]
            visit.attentionOutput.history = newValue[4]
            visit.mlpOutput.history = newValue[5]
        }
    }

    var metaState: [String] {
        get { [String(offset)] }
        set { visit.attention.offset = newValue.first.flatMap(Int.init) ?? 0 }
    }

    func innerState() -> [MLXArray] { state }

    func trim(_ n: Int) -> Int { 0 }

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .array(createCausalMask(n: n, offset: offset, windowSize: windowSize ?? maxSize))
    }

    func copy() -> any KVCache {
        let copy = InklingVisitCache(taps: taps, window: maxSize)
        // Every cache update replaces its array instead of mutating storage.
        copy.visit = visit
        return copy
    }
}

final class InklingLanguageModel: Module, LanguageModel {
    let textModel: InklingTextModel
    let visionModel: InklingVision?
    let imageTokenID: Int

    init(textModel: InklingTextModel, visionModel: InklingVision?, imageTokenID: Int) {
        self.textModel = textModel
        self.visionModel = visionModel
        self.imageTokenID = imageTokenID
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.text.tokens.size > 0 else { throw InklingInputProcessor.InputError.emptyPrompt }
        let tokens = input.text.tokens.reshaped(1, -1)
        let embeddings = textModel.embed(tokens)
        if let image = input.image {
            try insertImageFeatures(image, tokens: tokens, into: embeddings)
        }

        // Match mlx-vlm's chunk boundary: reserve the last prompt position
        // for the forward that produces the first sampled token.
        let chunkSize = prefill.resolvedStepSize(defaultStepSize: 1024)
        let last = tokens.dim(1) - 1
        for start in stride(from: 0, to: last, by: chunkSize) {
            let end = min(last, start + chunkSize)
            _ = forward(embeddings[0..., start..<end, 0...], cache: cache, projectLogits: false)
            eval(cache)
            Memory.clearCache()
        }
        return .logits(.init(logits: forward(embeddings[0..., last..., 0...], cache: cache)))
    }

    private func insertImageFeatures(
        _ image: LMInput.ProcessedImage, tokens: MLXArray, into embeddings: MLXArray
    ) throws {
        guard let visionModel, let frames = image.frames, !frames.isEmpty else {
            throw InklingInputProcessor.InputError.unsupportedMedia
        }
        let slots = tokens.asArray(Int.self).enumerated().compactMap { offset, token in
            token == imageTokenID ? offset : nil
        }
        guard slots.count == frames.reduce(0, { $0 + $1.h * $1.w }) else {
            throw InklingImageProcessor.InputError.imageCountMismatch
        }
        var offset = 0
        let arrays = frames.map { frame in
            let end = offset + frame.h * frame.w
            let pixels = image.pixels[offset..<end, .ellipsis]
            offset = end
            return visionModel(pixels, rows: frame.h, columns: frame.w)
        }
        let features = concatenated(arrays, axis: 0)
        embeddings[0, MLXArray(slots), 0...] = features.asType(embeddings.dtype)
    }

    func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        textModel.loop.visits.map { visit in
            InklingVisitCache(
                taps: textModel.config.convolutionSize,
                window: textModel.layers[visit.layer].attention.window)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let tokens = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        return forward(textModel.embed(tokens), cache: cache)
    }

    private func forward(
        _ embeddings: MLXArray, cache: [any KVCache]?, projectLogits: Bool = true
    ) -> MLXArray {
        let caches = cache?.map { $0 as! InklingVisitCache }
        var visits = caches?.map(\.visit) ?? textModel.makeCache()
        let result = textModel.forward(embeddings: embeddings, cache: &visits, projectLogits: projectLogits)
        if let caches {
            for (cache, visit) in zip(caches, visits) { cache.visit = visit }
        }
        return result
    }
}

extension InklingCheckpoint {
    func makeModel(
        config: InklingConfiguration, tokenizer: any MLXLMCommon.Tokenizer
    ) throws -> (model: InklingLanguageModel, processor: InklingInputProcessor) {
        let imageID = config.imageTokenID ?? 200054
        let model = try InklingLanguageModel(
            textModel: textModel(config: config.text),
            visionModel: config.vision.map { try visionModel(config: $0) },
            imageTokenID: imageID)
        let processor = InklingInputProcessor(
            tokenizer: tokenizer, imageTokenID: imageID,
            imagePatchSize: config.vision.map { $0.patchSize * $0.merge })
        return (model, processor)
    }
}

struct InklingInputProcessor: UserInputProcessor {
    enum InputError: Error {
        case unsupportedMedia
        case emptyPrompt
    }

    let tokenizer: any MLXLMCommon.Tokenizer
    let imageTokenID: Int
    let imagePatchSize: Int?

    func prepare(input: UserInput) throws -> LMInput {
        guard input.videos.isEmpty, input.audios.isEmpty else { throw InputError.unsupportedMedia }
        let generator = DefaultMessageGenerator()
        var messages = generator.generate(from: input)
        if case .chat(let chat) = input.prompt {
            for (index, message) in chat.enumerated() where !message.images.isEmpty {
                let imageParts = message.images.map { _ in ["type": "image"] }
                messages[index]["content"] = imageParts + [["type": "text", "text": message.content]]
            }
        }
        var additional = input.additionalContext
        if let history = additional?.removeValue(forKey: "inkling_reasoning_history") as? [String],
            history.count <= messages.count
        {
            let first = messages.count - history.count
            for (offset, reasoning) in history.enumerated() where !reasoning.isEmpty {
                messages[first + offset]["reasoning_content"] = reasoning
            }
        }
        let prompt = try InklingChatTemplate.render(
            messages: messages, tools: input.tools, reasoningEffort: additional?["reasoning_effort"])
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
        guard !input.images.isEmpty else { return LMInput(tokens: MLXArray(tokens)) }
        guard let imagePatchSize else { throw InputError.unsupportedMedia }
        let prepared = try input.images.map { try InklingImageProcessor.prepare($0, patch: imagePatchSize) }
        var expanded: [Int] = []
        var imageIndex = 0
        for token in tokens {
            if token == imageTokenID {
                guard imageIndex < prepared.count else {
                    throw InklingImageProcessor.InputError.imageCountMismatch
                }
                let grid = prepared[imageIndex].1
                expanded.append(contentsOf: repeatElement(token, count: grid.h * grid.w))
                imageIndex += 1
            } else {
                expanded.append(token)
            }
        }
        guard imageIndex == prepared.count else {
            throw InklingImageProcessor.InputError.imageCountMismatch
        }
        let pixels = concatenated(prepared.map { $0.0 }, axis: 0)
        return LMInput(
            text: .init(tokens: MLXArray(expanded)),
            image: .init(pixels: pixels, frames: prepared.map { $0.1 }))
    }
}
