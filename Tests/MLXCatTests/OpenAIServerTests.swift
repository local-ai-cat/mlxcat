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

    func testChatRequestParsePopulatesSamplingAndOpenAIFields() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [
                    {
                      "role": "assistant",
                      "content": "hello",
                      "reasoning_content": "prior reasoning"
                    }
                  ],
                  "max_completion_tokens": 32,
                  "temperature": 0.7,
                  "top_p": 0.8,
                  "top_k": 42,
                  "repetition_penalty": 1.2,
                  "min_p": 0.05,
                  "xtc_probability": 0.3,
                  "xtc_threshold": 0.4,
                  "presence_penalty": 0.5,
                  "frequency_penalty": 0.6,
                  "min_tokens": 7,
                  "logit_bias": {"42": 3.5, "99": -2},
                  "stop": ["END", "STOP"],
                  "seed": 1234,
                  "stream": true,
                  "stream_options": { "include_usage": true },
                  "enable_thinking": false,
                  "thinking_budget": 24,
                  "cache_session": "body-session",
                  "chat_template_kwargs": {
                    "reasoning_effort": "low",
                    "budget": 16,
                    "nested": { "enabled": true }
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages[0].reasoningContent, "prior reasoning")
        XCTAssertEqual(request.maxTokens, 32)
        XCTAssertEqual(request.temperature, 0.7, accuracy: 0.0001)
        XCTAssertEqual(request.topP, 0.8, accuracy: 0.0001)
        XCTAssertEqual(request.topK, 42)
        XCTAssertEqual(request.repetitionPenalty, 1.2, accuracy: 0.0001)
        XCTAssertEqual(request.minP, 0.05, accuracy: 0.0001)
        XCTAssertEqual(request.xtcProbability, 0.3, accuracy: 0.0001)
        XCTAssertEqual(request.xtcThreshold, 0.4, accuracy: 0.0001)
        XCTAssertEqual(request.presencePenalty, 0.5, accuracy: 0.0001)
        XCTAssertEqual(request.frequencyPenalty, 0.6, accuracy: 0.0001)
        XCTAssertEqual(request.minTokens, 7)
        XCTAssertEqual(try XCTUnwrap(request.logitBias[42]), 3.5, accuracy: 0.0001)
        XCTAssertEqual(try XCTUnwrap(request.logitBias[99]), -2, accuracy: 0.0001)
        XCTAssertEqual(request.stop, ["END", "STOP"])
        XCTAssertEqual(request.seed, 1234)
        XCTAssertTrue(request.stream)
        XCTAssertTrue(request.includeUsage)
        XCTAssertEqual(request.enableThinking, false)
        XCTAssertFalse(request.startInThinking)
        XCTAssertEqual(request.thinkingBudget, 24)
        XCTAssertEqual(request.cacheSession, "body-session")
        XCTAssertEqual(request.chatTemplateKwargs?["reasoning_effort"], .string("low"))
        XCTAssertEqual(request.chatTemplateKwargs?["budget"], .number(16))
        XCTAssertEqual(request.chatTemplateKwargs?["nested"], .object(["enabled": .bool(true)]))
    }

    func testChatRequestParseCapturesToolsArray() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "weather?"}],
                  "tools": [
                    {
                      "type": "function",
                      "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                          "type": "object",
                          "properties": {
                            "city": {"type": "string"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                          },
                          "required": ["city"]
                        }
                      }
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(
            request.tools,
            [
                .object([
                    "type": .string("function"),
                    "function": .object([
                        "name": .string("get_weather"),
                        "description": .string("Get weather for a city"),
                        "parameters": .object([
                            "type": .string("object"),
                            "properties": .object([
                                "city": .object(["type": .string("string")]),
                                "units": .object([
                                    "type": .string("string"),
                                    "enum": .array([.string("celsius"), .string("fahrenheit")]),
                                ]),
                            ]),
                            "required": .array([.string("city")]),
                        ]),
                    ]),
                ])
            ]
        )
    }

    func testChatRequestParseToolChoiceStringForms() throws {
        for (rawChoice, expectedChoice) in [
            ("none", OpenAIToolChoice.none),
            ("auto", OpenAIToolChoice.auto),
            ("required", OpenAIToolChoice.required),
        ] {
            let request = try OpenAIChatRequest.parse(
                Data(
                    """
                    {
                      "model": "test-model",
                      "messages": [{"role": "user", "content": "hello"}],
                      "tool_choice": "\(rawChoice)"
                    }
                    """.utf8
                )
            )

            XCTAssertEqual(request.toolChoice, expectedChoice)
        }
    }

    func testChatRequestParseToolChoiceFunctionForm() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"}
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.toolChoice, .function("get_weather"))
    }

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
            OpenAIChatImageReference(source: .url(second)),
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

    func testStartServesHealthOverLoopbackThenStopReleasesPort() async throws {
        let backend = FakePoolLifecycleBackend()
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        let port = try XCTUnwrap(server.boundPort)
        XCTAssertNotEqual(port, 0)

        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/health"))
        let (_, response) = try await URLSession.shared.data(from: url)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)

        server.stop()
        server.stop()  // idempotent

        // The port must actually be released: rebinding it succeeds (with a
        // brief retry window for kernel teardown).
        var rebound: OpenAIServer?
        for _ in 0..<50 {
            let candidate = try OpenAIServer(port: port, backend: backend)
            do {
                try await candidate.start()
                rebound = candidate
                break
            } catch {
                try await Task.sleep(nanoseconds: 100_000_000)
            }
        }
        XCTAssertNotNil(rebound, "port \(port) was not released after stop()")
        rebound?.stop()
    }

    func testBufferedChatUsesEnableThinkingWhenOpeningTagWasConsumed() async throws {
        let backend = ThinkingResponseBackend(chunks: [
            OpenAIChatChunk(text: "unfinished private reasoning", tokenID: 1, finishReason: "length"),
        ])
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let response = try await postChat(
            to: try XCTUnwrap(server.boundPort),
            stream: false,
            enableThinking: true
        )
        let body = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: response) as? [String: Any]
        )
        let choices = try XCTUnwrap(body["choices"] as? [[String: Any]])
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])

        XCTAssertEqual(message["reasoning_content"] as? String, "unfinished private reasoning")
        XCTAssertEqual(message["content"] as? String, "")
    }

    func testStreamingChatUsesEnableThinkingWhenOpeningTagWasConsumed() async throws {
        let backend = ThinkingResponseBackend(chunks: [
            OpenAIChatChunk(text: "private reasoning", tokenID: 1),
            OpenAIChatChunk(text: "</think>answer", tokenID: 2, finishReason: "stop"),
        ])
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let response = try await postChat(
            to: try XCTUnwrap(server.boundPort),
            stream: true,
            enableThinking: true
        )
        let events = String(decoding: response, as: UTF8.self)
            .split(separator: "\n")
            .compactMap { line -> [String: Any]? in
                guard line.hasPrefix("data: "), line != "data: [DONE]" else { return nil }
                let payload = line.dropFirst("data: ".count)
                guard let data = String(payload).data(using: .utf8) else { return nil }
                return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            }
        let deltas = events.compactMap { event -> [String: Any]? in
            guard let choices = event["choices"] as? [[String: Any]] else { return nil }
            return choices.first?["delta"] as? [String: Any]
        }

        XCTAssertEqual(
            deltas.compactMap { $0["reasoning_content"] as? String }.joined(),
            "private reasoning"
        )
        XCTAssertEqual(deltas.compactMap { $0["content"] as? String }.joined(), "answer")
    }

    func testBufferedChatKeepsPlainOutputAsContentWhenThinkingIsDisabled() async throws {
        let backend = ThinkingResponseBackend(chunks: [
            OpenAIChatChunk(text: "plain answer", tokenID: 1, finishReason: "stop"),
        ])
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let response = try await postChat(
            to: try XCTUnwrap(server.boundPort),
            stream: false,
            enableThinking: false
        )
        let body = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: response) as? [String: Any]
        )
        let choices = try XCTUnwrap(body["choices"] as? [[String: Any]])
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])

        XCTAssertNil(message["reasoning_content"])
        XCTAssertEqual(message["content"] as? String, "plain answer")
    }

    func testBufferedChatReportsCachedPromptTokens() async throws {
        let backend = ThinkingResponseBackend(chunks: [
            OpenAIChatChunk(
                text: "cached answer",
                tokenID: 1,
                finishReason: "stop",
                cachedPromptTokens: 37
            ),
        ])
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let response = try await postChat(
            to: try XCTUnwrap(server.boundPort),
            stream: false,
            enableThinking: false
        )
        let body = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: response) as? [String: Any]
        )
        let usage = try XCTUnwrap(body["usage"] as? [String: Any])
        let promptDetails = try XCTUnwrap(usage["prompt_tokens_details"] as? [String: Any])

        XCTAssertEqual(promptDetails["cached_tokens"] as? Int, 37)
    }

    private func postChat(
        to port: UInt16,
        stream: Bool,
        enableThinking: Bool
    ) async throws -> Data {
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions"))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "test-model",
            "messages": [["role": "user", "content": "test"]],
            "max_tokens": 16,
            "stream": stream,
            "enable_thinking": enableThinking,
        ])
        let (data, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        return data
    }
}

private struct ThinkingResponseBackend: OpenAIChatBackend {
    let chunks: [OpenAIChatChunk]
    let models = [OpenAIModelInfo(id: "test-model")]

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        OpenAIChatStream(
            promptTokens: 1,
            chunks: AsyncThrowingStream { continuation in
                for chunk in chunks {
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
        )
    }
}

private struct RouteFakeEngine: Sendable {
    let id: String
}

private struct RouteFakeLoader: EnginePoolModelLoader {
    func loadModel(id: String, modelURL: URL) async throws -> RouteFakeEngine {
        RouteFakeEngine(id: id)
    }

    func unloadModel(_ engine: RouteFakeEngine, id: String) async {}
}

private final class FakePoolLifecycleBackend: OpenAIModelLifecycleBackend, @unchecked Sendable {
    let models: [OpenAIModelInfo] = [
        OpenAIModelInfo(id: "alpha"),
        OpenAIModelInfo(id: "beta"),
    ]

    private let pool: EnginePool<RouteFakeLoader>

    init() {
        self.pool = EnginePool(
            models: [
                "alpha": DiscoveredModel(
                    id: "alpha",
                    modelURL: URL(fileURLWithPath: "/models/alpha", isDirectory: true),
                    estimatedSize: 100
                ),
                "beta": DiscoveredModel(
                    id: "beta",
                    modelURL: URL(fileURLWithPath: "/models/beta", isDirectory: true),
                    estimatedSize: 200
                ),
            ],
            loader: RouteFakeLoader(),
            finalCeiling: 1_000
        )
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        OpenAIChatStream(
            promptTokens: 0,
            chunks: AsyncThrowingStream { continuation in
                continuation.finish()
            }
        )
    }

    func modelPoolStatus() async throws -> OpenAIModelPoolStatus {
        OpenAIModelPoolStatus(await pool.status())
    }

    func loadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        do {
            let result = try await pool.load(id)
            return OpenAIModelLifecycleResult(
                modelID: id,
                message: result.alreadyLoaded ? "Already loaded: \(id)" : "Loaded: \(id)"
            )
        } catch {
            throw openAIError(from: error)
        }
    }

    func unloadModel(_ id: String) async throws -> OpenAIModelLifecycleResult {
        do {
            try await pool.unload(id)
            return OpenAIModelLifecycleResult(modelID: id)
        } catch {
            throw openAIError(from: error)
        }
    }

    func acquireForTest(_ id: String) async throws -> EnginePoolLease<RouteFakeEngine> {
        try await pool.acquire(id)
    }

    func releaseForTest(_ lease: EnginePoolLease<RouteFakeEngine>) async {
        await pool.release(lease)
    }

    private func openAIError(from error: Error) -> OpenAIHTTPError {
        guard let poolError = error as? EnginePoolError else {
            return OpenAIHTTPError(status: 500, message: String(describing: error))
        }
        return OpenAIHTTPError(
            status: poolError.httpStatus,
            message: poolError.message,
            retryAfterSeconds: poolError.retryAfterSeconds
        )
    }
}

private extension OpenAIModelPoolStatus {
    init(_ status: EnginePoolStatus) {
        self.init(
            finalCeiling: status.finalCeiling,
            currentModelMemory: status.currentModelMemory,
            modelCount: status.modelCount,
            loadedCount: status.loadedCount,
            models: status.models.map(OpenAIModelRuntimeStatus.init)
        )
    }
}

private extension OpenAIModelRuntimeStatus {
    init(_ status: EnginePoolModelStatus) {
        self.init(
            id: status.id,
            modelPath: status.modelPath,
            loaded: status.loaded,
            isLoading: status.isLoading,
            estimatedSize: status.estimatedSize,
            actualSize: status.actualSize,
            pinned: status.pinned,
            lastAccess: status.lastAccess,
            inUse: status.inUse
        )
    }
}

extension OpenAIServerTests {
    /// Sustained-serving smoke: 300 fresh-connection requests must all succeed.
    /// Context: untrack() must cancel() NWConnections — the daemon leaked one fd
    /// per request without it and wedged at the 256-fd rlimit (~245 requests;
    /// 5h soak 2026-07-08). The leak needs whatever reference the daemon holds
    /// and does NOT reproduce in-process (ARC closes the fd on dealloc here), so
    /// the true regression check is daemon-level: 300x /health then
    /// `lsof -p <pid> | grep -c TCP` → must stay ~1 (was 301 pre-fix).
    func testSustainedFreshConnectionsDoNotExhaustFileDescriptors() async throws {
        let backend = FakePoolLifecycleBackend()
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let port = try XCTUnwrap(server.boundPort)
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/health"))

        for attempt in 1...300 {
            // Ephemeral session per request forces a fresh TCP connection, the
            // pattern that leaked. Reused keep-alive sessions would mask it.
            let session = URLSession(configuration: .ephemeral)
            defer { session.finishTasksAndInvalidate() }
            let (_, response) = try await session.data(from: url)
            let status = (response as? HTTPURLResponse)?.statusCode
            XCTAssertEqual(
                status, 200,
                "request \(attempt) failed — fd-leak regression (pre-fix wedge was ~245)"
            )
            if status != 200 { return }
        }
    }
}

// MARK: - Audio and video as model input

/// These cover the modality gap closed on 2026-08-28: `OpenAIChatMessage` carried images
/// only, so audio and video sent to this server were parsed away to nothing. A model would
/// then answer about media it never received and the response looked perfectly healthy,
/// which is why every assertion below checks the reference actually ARRIVED rather than
/// checking the request merely parsed.
///
/// Not to be confused with `/v1/audio/transcriptions` — that is speech-to-text and has
/// nothing to do with audio as model input.
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
