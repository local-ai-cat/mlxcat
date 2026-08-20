import Foundation

public struct AudioTranscriptionRequest: Sendable {
    public let model: String
    public let fileName: String
    public let fileData: Data
    public let language: String?
    public let responseFormat: String
    public let temperature: Float

    public init(
        model: String,
        fileName: String,
        fileData: Data,
        language: String? = nil,
        responseFormat: String = "json",
        temperature: Float = 0
    ) {
        self.model = model
        self.fileName = fileName
        self.fileData = fileData
        self.language = language
        self.responseFormat = responseFormat
        self.temperature = temperature
    }

    public static func parse(contentType: String, body: Data) throws -> AudioTranscriptionRequest {
        let form = try MultipartFormData(contentType: contentType, body: body)

        guard let file = form.filePart(named: "file") else {
            throw OpenAIServerError.missingField("file")
        }
        guard let model = form.stringValue(forField: "model"), !model.isEmpty else {
            throw OpenAIServerError.missingField("model")
        }

        let language = form.stringValue(forField: "language").nilIfEmpty
        let responseFormat = form.stringValue(forField: "response_format").nilIfEmpty ?? "json"
        let temperature = form.stringValue(forField: "temperature").flatMap(Float.init) ?? 0

        return AudioTranscriptionRequest(
            model: model,
            fileName: file.filename,
            fileData: file.data,
            language: language,
            responseFormat: responseFormat,
            temperature: temperature
        )
    }
}

public struct AudioTranscriptionSegment: Sendable, Equatable {
    public let id: Int?
    public let start: Double?
    public let end: Double?
    public let text: String

    public init(id: Int? = nil, start: Double? = nil, end: Double? = nil, text: String) {
        self.id = id
        self.start = start
        self.end = end
        self.text = text
    }
}

public struct AudioTranscriptionResult: Sendable, Equatable {
    public let text: String
    public let language: String?
    public let duration: Double?
    public let segments: [AudioTranscriptionSegment]?

    public init(
        text: String,
        language: String? = nil,
        duration: Double? = nil,
        segments: [AudioTranscriptionSegment]? = nil
    ) {
        self.text = text
        self.language = language
        self.duration = duration
        self.segments = segments
    }
}

public protocol AudioTranscriptionBackend: Sendable {
    var transcriptionModels: [OpenAIModelInfo] { get }
    func transcribe(_ request: AudioTranscriptionRequest) async throws -> AudioTranscriptionResult
}

public func buildAudioTranscriptionResponse(_ result: AudioTranscriptionResult) -> [String: Any] {
    var response: [String: Any] = [
        "text": result.text
    ]
    if let language = result.language {
        response["language"] = language
    }
    if let duration = result.duration {
        response["duration"] = duration
    }
    if let segments = result.segments {
        response["segments"] = segments.map(buildAudioTranscriptionSegment)
    }
    return response
}

private func buildAudioTranscriptionSegment(_ segment: AudioTranscriptionSegment) -> [String: Any] {
    var response: [String: Any] = [
        "text": segment.text
    ]
    if let id = segment.id {
        response["id"] = id
    }
    if let start = segment.start {
        response["start"] = start
    }
    if let end = segment.end {
        response["end"] = end
    }
    return response
}

private extension Optional where Wrapped == String {
    var nilIfEmpty: String? {
        guard let value = self, !value.isEmpty else {
            return nil
        }
        return value
    }
}
