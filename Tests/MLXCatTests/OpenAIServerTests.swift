import Foundation
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

final class OpenAIServerTests: XCTestCase {
    func testNegativeContentLengthIsRejected() throws {
        let rawRequest = Data(
            """
            POST /v1/chat/completions HTTP/1.1\r
            Host: localhost\r
            Content-Length: -1\r
            \r
            {}
            """.utf8
        )

        XCTAssertThrowsError(try HTTPRequest.parseComplete(rawRequest)) { error in
            XCTAssertEqual(error as? OpenAIServerError, .invalidContentLength)
        }
    }

    func testOpenAIErrorBodyShapeIncludesTypeParamAndCode() throws {
        let body = openAIErrorBody(message: "not found", status: 404)
        let error = try XCTUnwrap(body["error"] as? [String: Any])

        XCTAssertEqual(Set(error.keys), ["message", "type", "param", "code"])
        XCTAssertEqual(error["message"] as? String, "not found")
        XCTAssertEqual(error["type"] as? String, "not_found_error")
        XCTAssertTrue(error["param"] is NSNull)
        XCTAssertTrue(error["code"] is NSNull)
    }

    func testHTTPRequestParsesCacheSessionHeaderCaseInsensitively() throws {
        let rawRequest = Data(
            """
            POST /v1/chat/completions HTTP/1.1\r
            Host: localhost\r
            X-Cache-Session: header-session\r
            Content-Length: 2\r
            \r
            {}
            """.utf8
        )

        let request = try HTTPRequest.parseComplete(rawRequest)

        XCTAssertEqual(request.headers["x-cache-session"], "header-session")
    }

    func testOpenAIErrorTypeMapping() {
        XCTAssertEqual(openAIErrorType(status: 401), "authentication_error")
        XCTAssertEqual(openAIErrorType(status: 400), "invalid_request_error")
        XCTAssertEqual(openAIErrorType(status: 413), "invalid_request_error")
        XCTAssertEqual(openAIErrorType(status: 422), "invalid_request_error")
        XCTAssertEqual(openAIErrorType(status: 404), "not_found_error")
        XCTAssertEqual(openAIErrorType(status: 429), "rate_limit_error")
        XCTAssertEqual(openAIErrorType(status: 500), "server_error")
        XCTAssertEqual(openAIErrorType(status: 503), "server_error")
    }

    func testOpenAIErrorResponseCarriesRetryAfter() {
        let response = openAIErrorResponse(
            message: "Scheduler waiting queue full (1/1). Try again shortly.",
            status: 503,
            retryAfterSeconds: 1
        )

        XCTAssertEqual(response.status, 503)
        XCTAssertEqual(response.retryAfterSeconds, 1)
        XCTAssertEqual(response.body["type"] as? String, "server_error")
    }

    func testModelsStatusLifecycleRouteShape() async throws {
        let backend = FakePoolLifecycleBackend()
        let server = try OpenAIServer(port: 0, backend: backend)
        let response = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "GET", path: "/v1/models/status", headers: [:], body: Data())
        )

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(response.body["final_ceiling"] as? Int64, 1_000)
        XCTAssertEqual(response.body["current_model_memory"] as? Int64, 0)
        XCTAssertEqual(response.body["model_count"] as? Int, 2)
        XCTAssertEqual(response.body["loaded_count"] as? Int, 0)
        let models = try XCTUnwrap(response.body["models"] as? [[String: Any]])
        XCTAssertEqual(models.count, 2)
        let first = try XCTUnwrap(models.first)
        XCTAssertEqual(
            Set(first.keys),
            [
                "id",
                "model_type",
                "model_path",
                "loaded",
                "is_loading",
                "estimated_size",
                "actual_size",
                "pinned",
                "last_access",
                "in_use",
                "footprint_source",
                "accounted_bytes"
            ]
        )
    }

    func testModelLoadAndUnloadLifecycleRoutes() async throws {
        let backend = FakePoolLifecycleBackend()
        let server = try OpenAIServer(port: 0, backend: backend)

        let load = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/alpha/load", headers: [:], body: Data())
        )
        XCTAssertEqual(load.status, 200)
        XCTAssertEqual(load.body["status"] as? String, "ok")
        XCTAssertEqual(load.body["model_id"] as? String, "alpha")
        XCTAssertEqual(load.body["message"] as? String, "Loaded: alpha")

        let unload = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/alpha/unload", headers: [:], body: Data())
        )
        XCTAssertEqual(unload.status, 200)
        XCTAssertEqual(unload.body["status"] as? String, "ok")
        XCTAssertEqual(unload.body["model_id"] as? String, "alpha")
    }

    func testModelLifecycleRouteErrorPathsUseOpenAIEnvelope() async throws {
        let backend = FakePoolLifecycleBackend()
        let server = try OpenAIServer(port: 0, backend: backend)

        let unknown = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/missing/load", headers: [:], body: Data())
        )
        XCTAssertEqual(unknown.status, 404)
        var error = try XCTUnwrap(unknown.body["error"] as? [String: Any])
        XCTAssertEqual(error["type"] as? String, "not_found_error")
        XCTAssertEqual(error["message"] as? String, "Model 'missing' not found. Available models: alpha, beta")

        let notLoaded = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/beta/unload", headers: [:], body: Data())
        )
        XCTAssertEqual(notLoaded.status, 400)
        error = try XCTUnwrap(notLoaded.body["error"] as? [String: Any])
        XCTAssertEqual(error["message"] as? String, "Model not loaded: beta")

        _ = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/alpha/load", headers: [:], body: Data())
        )
        let lease = try await backend.acquireForTest("alpha")
        let busy = await server.lifecycleRouteResponseForTesting(
            HTTPRequest(method: "POST", path: "/v1/models/alpha/unload", headers: [:], body: Data())
        )
        XCTAssertEqual(busy.status, 409)
        error = try XCTUnwrap(busy.body["error"] as? [String: Any])
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        await backend.releaseForTest(lease)
    }

    func testChatStreamOpeningChunksStartWithKeepaliveThenRoleAndDoneFrame() throws {
        let chunks = chatStreamOpeningChunks(id: "chatcmpl-test", created: 123, model: "test-model")

        XCTAssertEqual(chunks.count, 2)

        let keepalive = chunks[0]
        XCTAssertEqual(keepalive["id"] as? String, "chatcmpl-test")
        XCTAssertEqual(keepalive["object"] as? String, "chat.completion.chunk")
        XCTAssertEqual(keepalive["created"] as? Int, 0)
        XCTAssertEqual(keepalive["model"] as? String, "keepalive")
        let keepaliveChoices = try XCTUnwrap(keepalive["choices"] as? [[String: Any]])
        XCTAssertEqual(keepaliveChoices.count, 1)
        XCTAssertEqual(keepaliveChoices[0]["index"] as? Int, 0)
        let keepaliveDelta = try XCTUnwrap(keepaliveChoices[0]["delta"] as? [String: String])
        XCTAssertEqual(keepaliveDelta["content"], "")
        XCTAssertTrue(keepaliveChoices[0]["finish_reason"] is NSNull)

        let role = chunks[1]
        XCTAssertEqual(role["id"] as? String, "chatcmpl-test")
        XCTAssertEqual(role["object"] as? String, "chat.completion.chunk")
        XCTAssertEqual(role["created"] as? Int, 123)
        XCTAssertEqual(role["model"] as? String, "test-model")
        let roleChoices = try XCTUnwrap(role["choices"] as? [[String: Any]])
        XCTAssertEqual(roleChoices.count, 1)
        XCTAssertEqual(roleChoices[0]["index"] as? Int, 0)
        let roleDelta = try XCTUnwrap(roleChoices[0]["delta"] as? [String: String])
        XCTAssertEqual(roleDelta["role"], "assistant")
        XCTAssertTrue(roleChoices[0]["finish_reason"] is NSNull)

        XCTAssertEqual(chatStreamingDoneFrame, "data: [DONE]\n\n")
    }

    func testTruncateAtStopRemovesStopSequence() {
        let result = truncateAtStop("hello STOP hidden", stopSequences: ["STOP"])

        XCTAssertTrue(result.stopped)
        XCTAssertEqual(result.text, "hello ")
    }

    func testStreamingStopMatcherBuffersStopSplitAcrossChunks() {
        var matcher = StreamingStopSequenceMatcher(stopSequences: ["STOP"])

        let first = matcher.feed("hello ST")
        let second = matcher.feed("OP hidden")

        XCTAssertFalse(first.stopped)
        XCTAssertEqual(first.text, "hello")
        XCTAssertTrue(second.stopped)
        XCTAssertEqual(second.text, " ")
    }

    func testStreamingStopMatcherFlushesTailWhenNoStopMatches() {
        var matcher = StreamingStopSequenceMatcher(stopSequences: ["STOP"])

        let first = matcher.feed("hello ST")
        let final = matcher.finish()

        XCTAssertFalse(first.stopped)
        XCTAssertEqual(first.text, "hello")
        XCTAssertFalse(final.stopped)
        XCTAssertEqual(final.text, " ST")
    }

}
