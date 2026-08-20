import Foundation
@testable import MLXCatHTTP
import XCTest

final class AudioTranscriptionTests: XCTestCase {
    func testMultipartFormDataParsesTextFieldsAndBinaryFileExactly() throws {
        let boundary = "audio-boundary-123"
        let fileBytes = Data([0x52, 0x49, 0x46, 0x46, 0x00, 0x0d, 0x0a, 0xff])
        let body = multipartBody(
            boundary: boundary,
            parts: [
                MultipartTestPart(name: "model", value: Data("whisper-large-v3".utf8)),
                MultipartTestPart(name: "language", value: Data("en".utf8)),
                MultipartTestPart(
                    name: "file",
                    filename: "sample.wav",
                    contentType: "audio/wav",
                    value: fileBytes
                ),
            ]
        )

        let form = try MultipartFormData(
            contentType: "multipart/form-data; boundary=\"\(boundary)\"",
            body: body
        )

        XCTAssertEqual(form.stringValue(forField: "model"), "whisper-large-v3")
        XCTAssertEqual(form.stringValue(forField: "language"), "en")

        let file = try XCTUnwrap(form.filePart(named: "file"))
        XCTAssertEqual(file.filename, "sample.wav")
        XCTAssertEqual(file.contentType, "audio/wav")
        XCTAssertEqual(file.data, fileBytes)
    }

    func testAudioTranscriptionRequestParsesHappyPathWithDefaults() throws {
        let boundary = "transcription-boundary"
        let fileBytes = Data([0x00, 0x01, 0x0d, 0x0a, 0x02])
        let body = multipartBody(
            boundary: boundary,
            parts: [
                MultipartTestPart(name: "model", value: Data("audio-model".utf8)),
                MultipartTestPart(name: "language", value: Data("fr".utf8)),
                MultipartTestPart(
                    name: "file",
                    filename: "clip.m4a",
                    contentType: "audio/mp4",
                    value: fileBytes
                ),
            ]
        )

        let request = try AudioTranscriptionRequest.parse(
            contentType: "multipart/form-data; boundary=\(boundary)",
            body: body
        )

        XCTAssertEqual(request.model, "audio-model")
        XCTAssertEqual(request.fileName, "clip.m4a")
        XCTAssertEqual(request.fileData, fileBytes)
        XCTAssertEqual(request.language, "fr")
        XCTAssertEqual(request.responseFormat, "json")
        XCTAssertEqual(request.temperature, 0)
    }

    func testAudioTranscriptionRequestParsesResponseFormatAndTemperature() throws {
        let boundary = "transcription-options-boundary"
        let body = multipartBody(
            boundary: boundary,
            parts: [
                MultipartTestPart(name: "model", value: Data("audio-model".utf8)),
                MultipartTestPart(name: "response_format", value: Data("verbose_json".utf8)),
                MultipartTestPart(name: "temperature", value: Data("0.25".utf8)),
                MultipartTestPart(name: "file", filename: "clip.wav", value: Data([0x01])),
            ]
        )

        let request = try AudioTranscriptionRequest.parse(
            contentType: "multipart/form-data; boundary=\(boundary)",
            body: body
        )

        XCTAssertEqual(request.responseFormat, "verbose_json")
        XCTAssertEqual(request.temperature, 0.25, accuracy: 0.0001)
    }

    func testAudioTranscriptionRequestMissingFileThrowsMissingField() {
        let boundary = "missing-file-boundary"
        let body = multipartBody(
            boundary: boundary,
            parts: [
                MultipartTestPart(name: "model", value: Data("audio-model".utf8))
            ]
        )

        XCTAssertThrowsError(
            try AudioTranscriptionRequest.parse(
                contentType: "multipart/form-data; boundary=\(boundary)",
                body: body
            )
        ) { error in
            XCTAssertEqual(error as? OpenAIServerError, .missingField("file"))
        }
    }

    func testAudioTranscriptionRequestMissingModelThrowsMissingField() {
        let boundary = "missing-model-boundary"
        let body = multipartBody(
            boundary: boundary,
            parts: [
                MultipartTestPart(name: "file", filename: "clip.wav", value: Data([0x01]))
            ]
        )

        XCTAssertThrowsError(
            try AudioTranscriptionRequest.parse(
                contentType: "multipart/form-data; boundary=\(boundary)",
                body: body
            )
        ) { error in
            XCTAssertEqual(error as? OpenAIServerError, .missingField("model"))
        }
    }

    func testAudioTranscriptionResponseBuilderIncludesOptionalsAndSegmentsWhenPresent() throws {
        let response = buildAudioTranscriptionResponse(
            AudioTranscriptionResult(
                text: "hello world",
                language: "en",
                duration: 1.5,
                segments: [
                    AudioTranscriptionSegment(id: 7, start: 0.25, end: 1.25, text: "hello")
                ]
            )
        )

        XCTAssertEqual(response["text"] as? String, "hello world")
        XCTAssertEqual(response["language"] as? String, "en")
        XCTAssertEqual(response["duration"] as? Double, 1.5)
        let segments = try XCTUnwrap(response["segments"] as? [[String: Any]])
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0]["id"] as? Int, 7)
        XCTAssertEqual(segments[0]["start"] as? Double, 0.25)
        XCTAssertEqual(segments[0]["end"] as? Double, 1.25)
        XCTAssertEqual(segments[0]["text"] as? String, "hello")
    }

    func testAudioTranscriptionResponseBuilderOmitsAbsentOptionals() {
        let response = buildAudioTranscriptionResponse(AudioTranscriptionResult(text: "plain text"))

        XCTAssertEqual(Set(response.keys), ["text"])
        XCTAssertEqual(response["text"] as? String, "plain text")
        XCTAssertNil(response["language"])
        XCTAssertNil(response["duration"])
        XCTAssertNil(response["segments"])
    }
}

private struct MultipartTestPart {
    let name: String
    let filename: String?
    let contentType: String?
    let value: Data

    init(name: String, filename: String? = nil, contentType: String? = nil, value: Data) {
        self.name = name
        self.filename = filename
        self.contentType = contentType
        self.value = value
    }
}

private func multipartBody(boundary: String, parts: [MultipartTestPart]) -> Data {
    var body = Data()
    for part in parts {
        body.append(Data("--\(boundary)\r\n".utf8))
        var disposition = "Content-Disposition: form-data; name=\"\(part.name)\""
        if let filename = part.filename {
            disposition += "; filename=\"\(filename)\""
        }
        body.append(Data("\(disposition)\r\n".utf8))
        if let contentType = part.contentType {
            body.append(Data("Content-Type: \(contentType)\r\n".utf8))
        }
        body.append(Data("\r\n".utf8))
        body.append(part.value)
        body.append(Data("\r\n".utf8))
    }
    body.append(Data("--\(boundary)--\r\n".utf8))
    return body
}
