import Foundation
@testable import MLXServeHTTP
@testable import MLXServeNative
import XCTest

final class ContextWindowParityTests: XCTestCase {
    func testRequestBeyondModelContextReturnsStableOpenAIError() {
        XCTAssertThrowsError(
            try NativeModelEngine.validateContextWindow(
                promptTokens: 4_090,
                maxTokens: 16,
                contextWindow: 4_096
            )
        ) { error in
            let httpError = error as? OpenAIHTTPError
            XCTAssertEqual(httpError?.status, 400)
            XCTAssertEqual(httpError?.code, "context_length_exceeded")
        }
    }

    func testRequestAtModelContextBoundaryPasses() throws {
        XCTAssertNoThrow(
            try NativeModelEngine.validateContextWindow(
                promptTokens: 4_080,
                maxTokens: 16,
                contextWindow: 4_096
            )
        )
    }

    func testLoaderReadsNestedVisionTextContextWindow() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"text_config":{"max_position_embeddings":32768}}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))

        let loader = NativeModelLoader(maxConcurrentRequests: 1)

        XCTAssertEqual(try loader.contextWindow(in: directory), 32_768)
    }

    func testOpenAIErrorResponseCarriesStableCode() {
        let response = openAIErrorResponse(
            message: "context too long",
            status: 400,
            code: "context_length_exceeded"
        )

        XCTAssertEqual(response.body["code"] as? String, "context_length_exceeded")
    }
}
