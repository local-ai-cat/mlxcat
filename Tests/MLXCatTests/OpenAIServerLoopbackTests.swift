import Foundation
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

extension OpenAIServerTests {
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
            OpenAIChatChunk(text: "unfinished private reasoning", tokenID: 1, finishReason: "length")
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
            OpenAIChatChunk(text: "</think>answer", tokenID: 2, finishReason: "stop")
        ])
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let response = try await postChat(
            to: try XCTUnwrap(server.boundPort),
            stream: true,
            enableThinking: true
        )
        let events = try XCTUnwrap(String(data: response, encoding: .utf8))
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
            OpenAIChatChunk(text: "plain answer", tokenID: 1, finishReason: "stop")
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
            )
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
            "enable_thinking": enableThinking
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
