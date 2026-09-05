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
    func userInput(
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

    func prepareChatInput(_ request: OpenAIChatRequest) async throws -> LMInput {
        let candidates = ReasoningEffortCompatibility.candidates(
            for: request.chatTemplateKwargs?["reasoning_effort"],
            isHarmony: usesHarmonyChannels
        )
        var originalError: Error?

        for candidate in candidates {
            do {
                return try await resident.withValue { state in
                    let input = try self.userInput(from: request, reasoningEffort: candidate)
                    return try await state.context.processor.prepare(input: input)
                }
            } catch {
                if originalError == nil {
                    originalError = error
                }
            }
        }

        throw originalError ?? NativeModelEngineError.invalidPrompt
    }

    private func injectedMessages(from request: OpenAIChatRequest) throws -> [OpenAIChatMessage] {
        let directive: String?
        switch request.structuredOutput {
        case .jsonObject:
            directive = "You must respond with only a single valid JSON object and no other text."
        case .jsonSchema(_, let schema):
            let schemaJSON = try serializedJSONSchema(schema)
            directive = "You must respond with only a single valid JSON object conforming "
                + "to this JSON Schema: \(schemaJSON). Output only the JSON."
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
        guard let json = String(data: data, encoding: .utf8) else { throw OpenAIServerError.invalidJSON }
        return json
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
        // OpenAIChatMessage does not yet carry prior assistant tool calls; see the accounting document
        // for the unchanged native prompt-history limitation.
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
           ["http", "https", "file"].contains(scheme) {
            return url
        }
        if value.hasPrefix("/") {
            return URL(fileURLWithPath: value)
        }
        return nil
    }

    func additionalContext(
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

    var usesHarmonyChannels: Bool {
        let normalized = modelID.lowercased()
        return normalized.contains("gpt-oss") || normalized.contains("harmony")
    }

    func toolSpecDictionaries(from tools: [OpenAIJSONValue]?) -> [[String: any Sendable]]? {
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
