import Foundation
import XCTest

@testable import MLXCatHTTP

/// Concurrent SSE streaming through the real `OpenAIServer` socket.
///
/// `OpenAIServerTests` starts real servers, but every streaming case there
/// drives exactly ONE request at a time. The failure this repo is currently
/// hunting is the opposite shape: with batched `qwen3_5` the benchmark sees
/// width 4 return a completion carrying no `usage` frame and width 8 kill the
/// server process silently (`docs/KNOWN-FAILURES.md` §1d). `HybridBatchScaleTests`
/// cleared the engine of it — 8 rows x 512 tokens, prefix store, chunked prefill,
/// all clean — which leaves the layer this file covers and nothing else:
/// SSE framing, per-connection state and concurrent admission in `OpenAIServer`.
///
/// These run without weights or Metal, so unlike the model-gated suites they
/// execute on every `swift test` and in hosted CI. A fake backend cannot
/// reproduce a defect that needs real decode, but it CAN prove the socket layer
/// keeps N streams separate, complete and correctly framed — which is exactly
/// the half of §1d that was never under test.
final class OpenAIServerConcurrentStreamingTests: XCTestCase {
    func testEightConcurrentStreamsEachCompleteWithTheirOwnTokens() async throws {
        let backend = ConcurrentStreamingBackend(tokensPerStream: SSEProbe.tokensPerStream)
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let port = try XCTUnwrap(server.boundPort)

        let width = 8
        let bodies = try await withThrowingTaskGroup(of: (Int, String).self) { group in
            for index in 0 ..< width {
                group.addTask {
                    (index, try await SSEProbe.postStream(to: port, marker: "row\(index)"))
                }
            }
            var collected: [Int: String] = [:]
            for try await (index, body) in group {
                collected[index] = body
            }
            return collected
        }

        XCTAssertEqual(bodies.count, width, "not every concurrent stream produced a response")

        // The test is only meaningful if the streams actually overlapped. A
        // server that quietly serialized them would satisfy every other
        // assertion here and prove nothing.
        XCTAssertGreaterThan(
            backend.maxConcurrent, 1,
            "no two requests were ever in flight at once — this ran serially and tested nothing"
        )

        for index in 0 ..< width {
            let body = try XCTUnwrap(bodies[index])
            let marker = "row\(index)"

            XCTAssertTrue(
                body.hasSuffix("data: [DONE]\n\n"),
                "stream \(index) did not terminate with the DONE frame"
            )

            let events = SSEProbe.dataEvents(in: body)
            XCTAssertFalse(events.isEmpty, "stream \(index) carried no data frames")

            // Every delta in this connection must belong to THIS request. A
            // shared per-connection buffer or a mis-keyed continuation shows up
            // here as another row's marker arriving on this socket.
            let text = events.compactMap { event -> String? in
                guard let choices = event["choices"] as? [[String: Any]],
                    let delta = choices.first?["delta"] as? [String: Any]
                else { return nil }
                return delta["content"] as? String
            }.joined()
            XCTAssertEqual(
                text, String(repeating: marker, count: SSEProbe.tokensPerStream),
                "stream \(index) did not receive exactly its own tokens"
            )

            // The §1d symptom, asserted directly: a stream that asked for usage
            // and completed must carry a usage frame.
            let usageFrames = events.filter { $0["usage"] is [String: Any] }
            XCTAssertEqual(
                usageFrames.count, 1,
                "stream \(index) carried \(usageFrames.count) usage frames, expected exactly 1"
            )

            let finishes = events.compactMap { event -> String? in
                guard let choices = event["choices"] as? [[String: Any]] else { return nil }
                return choices.first?["finish_reason"] as? String
            }
            XCTAssertEqual(finishes, ["stop"], "stream \(index) reported \(finishes) as finish reasons")
        }
    }

    /// A stream the client abandons mid-flight must not take its neighbours with
    /// it. The width-8 symptom in §1d is the whole process dying, so the
    /// neighbour surviving is the property worth pinning.
    func testAbandoningOneStreamDoesNotDisturbTheOthers() async throws {
        let backend = ConcurrentStreamingBackend(tokensPerStream: SSEProbe.tokensPerStream)
        let server = try OpenAIServer(port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let port = try XCTUnwrap(server.boundPort)

        let survivor = Task { try await SSEProbe.postStream(to: port, marker: "survivor") }

        // Cancelled after it has certainly started but before it can finish.
        let abandoned = Task { try await SSEProbe.postStream(to: port, marker: "abandoned") }
        try await Task.sleep(nanoseconds: 20_000_000)
        abandoned.cancel()

        let body = try await survivor.value
        XCTAssertTrue(body.hasSuffix("data: [DONE]\n\n"), "the surviving stream was truncated")
        let text = SSEProbe.dataEvents(in: body).compactMap { event -> String? in
            guard let choices = event["choices"] as? [[String: Any]],
                let delta = choices.first?["delta"] as? [String: Any]
            else { return nil }
            return delta["content"] as? String
        }.joined()
        XCTAssertEqual(text, String(repeating: "survivor", count: SSEProbe.tokensPerStream))
    }
}

/// The client half of these tests, kept off `XCTestCase`.
///
/// It lives here rather than as static members of the test class because a
/// `Task` closure that reaches `Self.` on a non-Sendable `XCTestCase` subclass
/// makes Swift 6's region-based isolation checker give up outright ("pattern
/// that the region-based isolation checker does not understand how to check").
/// A caseless enum has no instance to capture, so the closures capture only the
/// port and the marker.
private enum SSEProbe {
    static let tokensPerStream = 12

    static func dataEvents(in body: String) -> [[String: Any]] {
        body.split(separator: "\n").compactMap { line -> [String: Any]? in
            guard line.hasPrefix("data: "), line != "data: [DONE]" else { return nil }
            guard let data = String(line.dropFirst("data: ".count)).data(using: .utf8) else {
                return nil
            }
            return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
    }

    static func postStream(to port: UInt16, marker: String) async throws -> String {
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions"))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "test-model",
            "messages": [["role": "user", "content": marker]],
            "max_tokens": tokensPerStream,
            "stream": true,
            "stream_options": ["include_usage": true],
        ])
        let session = URLSession(configuration: .ephemeral)
        let (data, response) = try await session.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        return String(decoding: data, as: UTF8.self)
    }
}

/// Emits the request's own user message once per token, with a gap between
/// tokens so concurrent requests genuinely overlap inside the server, and
/// records the widest overlap it ever saw.
private final class ConcurrentStreamingBackend: OpenAIChatBackend, @unchecked Sendable {
    let models = [OpenAIModelInfo(id: "test-model")]
    private let tokensPerStream: Int
    private let lock = NSLock()
    private var inFlight = 0
    private var widest = 0

    init(tokensPerStream: Int) {
        self.tokensPerStream = tokensPerStream
    }

    var maxConcurrent: Int {
        lock.lock()
        defer { lock.unlock() }
        return widest
    }

    private func enter() {
        lock.lock()
        inFlight += 1
        widest = max(widest, inFlight)
        lock.unlock()
    }

    private func leave() {
        lock.lock()
        inFlight -= 1
        lock.unlock()
    }

    func startChatCompletion(_ request: OpenAIChatRequest) async throws -> OpenAIChatStream {
        let marker = request.messages.last?.content ?? "?"
        let count = tokensPerStream
        enter()
        return OpenAIChatStream(
            promptTokens: 1,
            chunks: AsyncThrowingStream { continuation in
                Task { [weak self] in
                    defer { self?.leave() }
                    for index in 0 ..< count {
                        // Long enough that eight requests are reliably in flight
                        // together, short enough to keep the suite quick.
                        try? await Task.sleep(nanoseconds: 3_000_000)
                        continuation.yield(
                            OpenAIChatChunk(
                                text: marker,
                                tokenID: index,
                                finishReason: index == count - 1 ? "stop" : nil
                            )
                        )
                    }
                    continuation.finish()
                }
            }
        )
    }
}
