import Foundation
import Network

public extension OpenAIChatRequest {
    static func parse(_ body: Data) throws -> OpenAIChatRequest {
        let rawObject: Any
        do {
            rawObject = try JSONSerialization.jsonObject(with: body)
        } catch {
            throw OpenAIServerError.invalidJSON
        }

        guard let object = rawObject as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        guard let model = object["model"] as? String, !model.isEmpty else {
            throw OpenAIServerError.missingField("model")
        }
        guard let rawMessages = object["messages"] as? [[String: Any]] else {
            throw OpenAIServerError.missingField("messages")
        }

        let messages = rawMessages.compactMap(Self.message)
        guard !messages.isEmpty else { throw OpenAIServerError.missingField("messages") }
        let streamOptions = object["stream_options"] as? [String: Any]
        let chatTemplateKwargs = try parseChatTemplateKwargs(object["chat_template_kwargs"])
        let structuredOutput = try StructuredOutputParser.parse(from: object)
        let tools = try parseTools(object["tools"])
        let toolChoice = try OpenAIToolChoice.parse(object["tool_choice"])

        return OpenAIChatRequest(
            model: model,
            messages: messages,
            maxTokens: intValue(object["max_tokens"] ?? object["max_completion_tokens"]) ?? 16,
            temperature: floatValue(object["temperature"]) ?? 0,
            topP: floatValue(object["top_p"]) ?? 0,
            topK: intValue(object["top_k"]) ?? 0,
            repetitionPenalty: floatValue(object["repetition_penalty"]) ?? 1,
            minP: floatValue(object["min_p"]) ?? 0,
            xtcProbability: floatValue(object["xtc_probability"]) ?? 0,
            xtcThreshold: floatValue(object["xtc_threshold"]) ?? 0.1,
            presencePenalty: floatValue(object["presence_penalty"]) ?? 0,
            frequencyPenalty: floatValue(object["frequency_penalty"]) ?? 0,
            minTokens: intValue(object["min_tokens"]) ?? 0,
            logitBias: try openAILogitBias(object["logit_bias"]),
            stop: try stopValues(object["stop"]),
            seed: intValue(object["seed"]),
            stream: object["stream"] as? Bool ?? false,
            includeUsage: streamOptions?["include_usage"] as? Bool ?? false,
            enableThinking: object["enable_thinking"] as? Bool,
            thinkingBudget: intValue(object["thinking_budget"]),
            chatTemplateKwargs: chatTemplateKwargs,
            structuredOutput: structuredOutput,
            tools: tools,
            toolChoice: toolChoice,
            toolGrammar: object["tool_grammar"] as? Bool,
            cacheSession: stringValue(object["cache_session"])
        )
    }

    private static func message(_ raw: [String: Any]) -> OpenAIChatMessage? {
        guard let role = raw["role"] as? String else { return nil }
        let reasoningContent = raw["reasoning_content"] as? String
        if let content = raw["content"] as? String {
            return OpenAIChatMessage(role: role, content: content, reasoningContent: reasoningContent)
        }
        if let parts = raw["content"] as? [[String: Any]] {
            let text = parts.compactMap { part in
                part["type"] as? String == "text" ? part["text"] as? String : nil
            }.joined()
            let imageReferences = parts.compactMap(Self.imageReference)
            return OpenAIChatMessage(
                role: role,
                content: text,
                reasoningContent: reasoningContent,
                imageReferences: imageReferences,
                audioReferences: parts.compactMap { Self.mediaReference(from: $0, kind: .audio) },
                videoReferences: parts.compactMap { Self.mediaReference(from: $0, kind: .video) }
            )
        }
        return nil
    }

    private static func imageReference(from part: [String: Any]) -> OpenAIChatImageReference? {
        guard part["type"] as? String == "image_url",
              let imageURL = part["image_url"] as? [String: Any],
              let url = imageURL["url"] as? String,
              !url.isEmpty
        else {
            return nil
        }

        if url.lowercased().hasPrefix("data:") {
            return OpenAIChatImageReference(source: .dataURI(url))
        }
        return OpenAIChatImageReference(source: .url(url))
    }

    fileprivate enum MediaKind {
        case audio
        case video

        /// The content-part spellings we accept. `audio_url`/`video_url` mirror
        /// `image_url` and are what Local AI Cat sends; `input_audio` is OpenAI's own
        /// spelling for inline audio and is accepted so ordinary OpenAI clients work.
        var partTypes: [String] {
            switch self {
            case .audio: return ["audio_url", "input_audio"]
            case .video: return ["video_url"]
            }
        }

        var urlKey: String {
            switch self {
            case .audio: return "audio_url"
            case .video: return "video_url"
            }
        }
    }

    private static func mediaReference(
        from part: [String: Any],
        kind: MediaKind
    ) -> OpenAIChatMediaReference? {
        guard let type = part["type"] as? String, kind.partTypes.contains(type) else {
            return nil
        }

        // OpenAI's `input_audio` carries base64 + a format rather than a URL, so it is
        // reassembled into the data URI the rest of this path already understands.
        if type == "input_audio" {
            guard let audio = part["input_audio"] as? [String: Any],
                  let data = audio["data"] as? String, !data.isEmpty
            else {
                return nil
            }
            let format = (audio["format"] as? String) ?? "wav"
            return OpenAIChatMediaReference(source: .dataURI("data:audio/\(format);base64,\(data)"))
        }

        guard let container = part[kind.urlKey] as? [String: Any],
              let url = container["url"] as? String,
              !url.isEmpty
        else {
            return nil
        }

        if url.lowercased().hasPrefix("data:") {
            return OpenAIChatMediaReference(source: .dataURI(url))
        }
        return OpenAIChatMediaReference(source: .url(url))
    }

    private static func floatValue(_ value: Any?) -> Float? {
        switch value {
        case let value as Double:
            return Float(value)
        case let value as Float:
            return value
        case let value as Int:
            return Float(value)
        case let value as NSNumber:
            return value.floatValue
        default:
            return nil
        }
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as NSNumber:
            return value.intValue
        default:
            return nil
        }
    }

    private static func stringValue(_ value: Any?) -> String? {
        guard let value = value as? String, !value.isEmpty else { return nil }
        return value
    }

    private static func stopValues(_ value: Any?) throws -> [String] {
        guard let value else { return [] }
        if let string = value as? String {
            return [string]
        }
        if let strings = value as? [String] {
            return strings
        }
        throw OpenAIServerError.invalidJSON
    }

    private static func parseChatTemplateKwargs(_ value: Any?) throws -> [String: OpenAIJSONValue]? {
        guard let value else { return nil }
        guard let object = value as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        var converted: [String: OpenAIJSONValue] = [:]
        for (key, value) in object {
            guard let jsonValue = OpenAIJSONValue(value) else {
                throw OpenAIServerError.invalidJSON
            }
            converted[key] = jsonValue
        }
        return converted
    }

    private static func parseTools(_ value: Any?) throws -> [OpenAIJSONValue]? {
        guard let value else { return nil }
        guard let array = value as? [Any] else {
            throw OpenAIServerError.invalidJSON
        }
        var converted: [OpenAIJSONValue] = []
        for item in array {
            guard let jsonValue = OpenAIJSONValue(item) else {
                throw OpenAIServerError.invalidJSON
            }
            converted.append(jsonValue)
        }
        return converted
    }
}
