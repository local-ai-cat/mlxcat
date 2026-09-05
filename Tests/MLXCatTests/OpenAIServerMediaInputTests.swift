import Foundation
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

final class OpenAIServerMediaInputTests: XCTestCase {
    private func request(parts: String) throws -> OpenAIChatRequest {
        try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": [\(parts)]}]
                }
                """.utf8
            )
        )
    }

    func testAudioURLPartIsCarried() throws {
        let parsed = try request(parts: #"{"type": "audio_url", "audio_url": {"url": "file:///tmp/a.wav"}}"#)
        XCTAssertEqual(parsed.messages[0].audioReferences,
                       [OpenAIChatMediaReference(source: .url("file:///tmp/a.wav"))])
        XCTAssertTrue(parsed.messages[0].videoReferences.isEmpty)
    }

    func testVideoURLPartIsCarried() throws {
        let parsed = try request(parts: #"{"type": "video_url", "video_url": {"url": "file:///tmp/v.mp4"}}"#)
        XCTAssertEqual(parsed.messages[0].videoReferences,
                       [OpenAIChatMediaReference(source: .url("file:///tmp/v.mp4"))])
        XCTAssertTrue(parsed.messages[0].audioReferences.isEmpty)
    }

    func testInlineDataURIAudioIsCarried() throws {
        let parsed = try request(parts: #"{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}}"#)
        XCTAssertEqual(parsed.messages[0].audioReferences,
                       [OpenAIChatMediaReference(source: .dataURI("data:audio/wav;base64,AAAA"))])
    }

    /// OpenAI's own spelling carries base64 + a format instead of a URL; it is reassembled
    /// into the data URI the rest of the path understands, so ordinary OpenAI clients work.
    func testOpenAIInputAudioPartIsAccepted() throws {
        let parsed = try request(parts: #"{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}}"#)
        XCTAssertEqual(parsed.messages[0].audioReferences,
                       [OpenAIChatMediaReference(source: .dataURI("data:audio/mp3;base64,AAAA"))])
    }

    /// Text, image, audio and video in one turn must not cannibalise each other — each
    /// extractor filters by part type independently.
    func testMixedModalitiesInOneMessageDoNotCollide() throws {
        let parsed = try request(parts: """
            {"type": "text", "text": "what is in these?"},
            {"type": "image_url", "image_url": {"url": "file:///tmp/i.png"}},
            {"type": "audio_url", "audio_url": {"url": "file:///tmp/a.wav"}},
            {"type": "video_url", "video_url": {"url": "file:///tmp/v.mp4"}}
            """)
        let message = parsed.messages[0]
        XCTAssertEqual(message.content, "what is in these?")
        XCTAssertEqual(message.imageReferences.count, 1)
        XCTAssertEqual(message.audioReferences.count, 1)
        XCTAssertEqual(message.videoReferences.count, 1)
    }

    /// A text-only turn must stay byte-identical to before this feature existed.
    func testTextOnlyMessageGainsNoMediaReferences() throws {
        let parsed = try request(parts: #"{"type": "text", "text": "hello"}"#)
        XCTAssertTrue(parsed.messages[0].audioReferences.isEmpty)
        XCTAssertTrue(parsed.messages[0].videoReferences.isEmpty)
        XCTAssertTrue(parsed.messages[0].imageReferences.isEmpty)
    }

    /// A malformed part is skipped, not turned into an empty reference that would look
    /// like media the model should have received.
    func testMalformedMediaPartIsSkipped() throws {
        let parsed = try request(parts: #"{"type": "audio_url", "audio_url": {"url": ""}}"#)
        XCTAssertTrue(parsed.messages[0].audioReferences.isEmpty)
    }
}
