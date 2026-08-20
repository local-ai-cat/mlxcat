import Foundation
@testable import MLXCatHTTP
import XCTest

final class EmbeddingsTests: XCTestCase {
    func testEmbeddingsRequestParsesStringInputWithDefaults() throws {
        let request = try OpenAIEmbeddingsRequest.parse(
            Data(
                """
                {
                  "model": "embed-model",
                  "input": "hello"
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "embed-model")
        XCTAssertEqual(request.input, .string("hello"))
        XCTAssertEqual(request.encodingFormat, .float)
        XCTAssertNil(request.dimensions)
    }

    func testEmbeddingsRequestParsesArrayInputEncodingFormatAndDimensions() throws {
        let request = try OpenAIEmbeddingsRequest.parse(
            Data(
                """
                {
                  "model": "embed-model",
                  "input": ["one", "two"],
                  "encoding_format": "base64",
                  "dimensions": 2
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "embed-model")
        XCTAssertEqual(request.input, .strings(["one", "two"]))
        XCTAssertEqual(request.encodingFormat, .base64)
        XCTAssertEqual(request.dimensions, 2)
    }

    func testEmbeddingsResponseShapeUsesFloatVectorsAndRenormalizesTruncation() async throws {
        let backend = FakeEmbeddingsBackend(
            result: OpenAIEmbeddingsResult(
                embeddings: [[3, 4, 12]],
                promptTokens: 7
            )
        )
        let request = OpenAIEmbeddingsRequest(
            input: .string("hello"),
            model: "embed-model",
            encodingFormat: .float,
            dimensions: 2
        )
        let response = buildEmbeddingsResponse(request: request, result: try await backend.embed(request))

        XCTAssertEqual(response["object"] as? String, "list")
        XCTAssertEqual(response["model"] as? String, "embed-model")
        let data = try XCTUnwrap(response["data"] as? [[String: Any]])
        XCTAssertEqual(data[0]["object"] as? String, "embedding")
        XCTAssertEqual(data[0]["index"] as? Int, 0)
        let embedding = try XCTUnwrap(data[0]["embedding"] as? [Float])
        XCTAssertEqual(embedding, [0.6, 0.8])
        let usage = try XCTUnwrap(response["usage"] as? [String: Int])
        XCTAssertEqual(usage["prompt_tokens"], 7)
        XCTAssertEqual(usage["total_tokens"], 7)
    }

    func testEmbeddingsResponseShapeUsesBase64Vectors() async throws {
        let backend = FakeEmbeddingsBackend(
            result: OpenAIEmbeddingsResult(
                embeddings: [[1, 2]],
                promptTokens: 3
            )
        )
        let request = OpenAIEmbeddingsRequest(
            input: .string("hello"),
            model: "embed-model",
            encodingFormat: .base64
        )
        let response = buildEmbeddingsResponse(request: request, result: try await backend.embed(request))

        let data = try XCTUnwrap(response["data"] as? [[String: Any]])
        XCTAssertEqual(data[0]["embedding"] as? String, "AACAPwAAAEA=")
    }

    func testEncodeEmbeddingBase64UsesLittleEndianFloat32Bytes() {
        let encoded = encodeEmbeddingBase64([1, 2])
        let decoded = Data(base64Encoded: encoded)

        XCTAssertEqual(encoded, "AACAPwAAAEA=")
        XCTAssertEqual(decoded, Data([0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40]))
    }

    func testEmbeddingModelAvailabilityReturnsNotFoundWhenBackendUnavailable() throws {
        let response = try XCTUnwrap(
            embeddingModelAvailabilityResponse(requestedModel: "text-model", embeddingModels: [])
        )

        XCTAssertEqual(response.status, 404)
        let error = try XCTUnwrap(response.body["error"] as? [String: Any])
        XCTAssertEqual(error["message"] as? String, "embeddings backend unavailable")
        XCTAssertEqual(error["type"] as? String, "not_found_error")
    }

    func testEmbeddingModelAvailabilityReturnsBadRequestForNonEmbeddingModel() throws {
        let response = try XCTUnwrap(
            embeddingModelAvailabilityResponse(
                requestedModel: "text-model",
                embeddingModels: [OpenAIModelInfo(id: "embed-model")]
            )
        )

        XCTAssertEqual(response.status, 400)
        let error = try XCTUnwrap(response.body["error"] as? [String: Any])
        XCTAssertEqual(
            error["message"] as? String,
            "Model 'text-model' is not an embedding model. Use /v1/chat/completions for LLM models."
        )
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
    }

    func testEmbeddingModelAvailabilityAcceptsConfiguredEmbeddingModel() {
        XCTAssertNil(
            embeddingModelAvailabilityResponse(
                requestedModel: "embed-model",
                embeddingModels: [OpenAIModelInfo(id: "embed-model")]
            )
        )
    }
}

private struct FakeEmbeddingsBackend: OpenAIEmbeddingsBackend {
    let embeddingModels = [OpenAIModelInfo(id: "embed-model")]
    let result: OpenAIEmbeddingsResult

    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> OpenAIEmbeddingsResult {
        result
    }
}
