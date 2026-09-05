import Foundation
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

extension OpenAIServerTests {
    func testChatRequestParseCapturesDataURIImageURLPart() throws {
        let dataURI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "image_url", "image_url": {"url": "\(dataURI)"}}
                      ]
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.messages[0].content, "")
        XCTAssertEqual(request.messages[0].imageReferences, [
            OpenAIChatImageReference(source: .dataURI(dataURI))
        ])
    }

    func testChatRequestParseCapturesHTTPImageURLPart() throws {
        let imageURL = "https://example.com/image.png"
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "image_url", "image_url": {"url": "\(imageURL)"}}
                      ]
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.messages[0].content, "")
        XCTAssertEqual(request.messages[0].imageReferences, [
            OpenAIChatImageReference(source: .url(imageURL))
        ])
    }

    func testChatRequestParseMixedTextAndImagePreservesTextConcatenation() throws {
        let imageURL = "https://example.com/photo.jpg"
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "text", "text": "Describe "},
                        {"type": "image_url", "image_url": {"url": "\(imageURL)"}},
                        {"type": "text", "text": "please."}
                      ]
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.messages[0].content, "Describe please.")
        XCTAssertEqual(request.messages[0].imageReferences, [
            OpenAIChatImageReference(source: .url(imageURL))
        ])
    }

    func testChatRequestParseTextOnlyLeavesImageReferencesEmpty() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": " world"}
                      ]
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.messages[0].content, "hello world")
        XCTAssertEqual(request.messages[0].imageReferences, [])
    }

    func testChatRequestParseCapturesMultipleImagesInOrder() throws {
        let first = "data:image/jpeg;base64,abc123"
        let second = "https://example.com/second.png"
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "image_url", "image_url": {"url": "\(first)"}},
                        {"type": "text", "text": "compare"},
                        {"type": "image_url", "image_url": {"url": "\(second)"}}
                      ]
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.messages[0].content, "compare")
        XCTAssertEqual(request.messages[0].imageReferences, [
            OpenAIChatImageReference(source: .dataURI(first)),
            OpenAIChatImageReference(source: .url(second))
        ])
    }

    func testChatRequestParseCoercesSingleStopString() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stop": "END"
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.stop, ["END"])
    }

    func testChatRequestParseMissingModelIsValidationError() {
        XCTAssertThrowsError(
            try OpenAIChatRequest.parse(
                Data(
                    """
                    {
                      "messages": [{"role": "user", "content": "hello"}]
                    }
                    """.utf8
                )
            )
        ) { error in
            XCTAssertEqual(error as? OpenAIServerError, .missingField("model"))
            XCTAssertEqual((error as? OpenAIServerError)?.httpStatus, 422)
        }
    }

    func testChatRequestParseMissingMessagesIsValidationError() {
        XCTAssertThrowsError(
            try OpenAIChatRequest.parse(
                Data(
                    """
                    {
                      "model": "test-model"
                    }
                    """.utf8
                )
            )
        ) { error in
            XCTAssertEqual(error as? OpenAIServerError, .missingField("messages"))
            XCTAssertEqual((error as? OpenAIServerError)?.httpStatus, 422)
        }
    }

    func testChatRequestParseUsesNeutralDefaults() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.maxTokens, 16)
        XCTAssertEqual(request.temperature, 0)
        XCTAssertEqual(request.topP, 0)
        XCTAssertEqual(request.topK, 0)
        XCTAssertEqual(request.repetitionPenalty, 1)
        XCTAssertEqual(request.minP, 0)
        XCTAssertEqual(request.xtcProbability, 0)
        XCTAssertEqual(request.xtcThreshold, 0.1, accuracy: 0.0001)
        XCTAssertEqual(request.presencePenalty, 0)
        XCTAssertEqual(request.frequencyPenalty, 0)
        XCTAssertEqual(request.stop, [])
        XCTAssertNil(request.seed)
        XCTAssertFalse(request.stream)
        XCTAssertFalse(request.includeUsage)
        XCTAssertNil(request.enableThinking)
        XCTAssertFalse(request.startInThinking)
        XCTAssertNil(request.chatTemplateKwargs)
        XCTAssertNil(request.tools)
        XCTAssertNil(request.toolChoice)
    }

    func testChatRequestParseIncludeUsageDefaultsFalse() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "stream": true
                }
                """.utf8
            )
        )

        XCTAssertFalse(request.includeUsage)
    }

}
