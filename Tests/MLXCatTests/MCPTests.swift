import Foundation
import Network
@testable import MLXCatHTTP
import XCTest

final class MCPTests: XCTestCase {
    func testMCPConfigParsesClaudeDesktopShape() throws {
        let config = try MCPConfig.parse(
            [
                "mcpServers": [
                    "fake": [
                        "command": "python3",
                        "args": ["server.py"],
                        "env": ["TOKEN": "abc"],
                    ],
                    "disabled": [
                        "command": "ignored",
                        "enabled": false,
                    ],
                    "sse": [
                        "transport": "sse",
                        "url": "http://127.0.0.1:1",
                        "headers": ["Authorization": "Bearer test"],
                    ],
                    "stream": [
                        "transport": "streamable-http",
                        "url": "http://127.0.0.1:2/mcp",
                        "timeout": 3,
                    ],
                ]
            ]
        )

        XCTAssertEqual(
            config.servers,
            [
                MCPServerConfig(name: "fake", command: "python3", args: ["server.py"], env: ["TOKEN": "abc"]),
                MCPServerConfig(
                    name: "sse",
                    transport: "sse-endpoint",
                    url: "http://127.0.0.1:1",
                    headers: ["Authorization": "Bearer test"]
                ),
                MCPServerConfig(name: "stream", transport: "streamable-http", url: "http://127.0.0.1:2/mcp", timeoutMs: 3_000),
            ]
        )
    }

    func testMCPConfigParsesPerServerTimeout() throws {
        let config = try MCPConfig.parse(
            [
                "mcpServers": [
                    "fake": [
                        "command": "python3",
                        "args": ["server.py"],
                        "timeoutMs": 250,
                    ]
                ]
            ]
        )

        XCTAssertEqual(config.servers.first?.timeoutMs, 250)
    }

    func testMCPHandlerUnconfiguredEndpointShapes() async {
        let handler = MCPHandler(manager: nil)

        let tools = await handler.toolsResponse()
        XCTAssertEqual(tools.status, 200)
        XCTAssertEqual(tools.body["count"] as? Int, 0)
        XCTAssertEqual((tools.body["tools"] as? [Any])?.count, 0)

        let servers = await handler.serversResponse()
        XCTAssertEqual(servers.status, 200)
        XCTAssertEqual((servers.body["servers"] as? [Any])?.count, 0)

        let execute = await handler.executeResponse(body: Data(#"{"tool":"fake__echo","arguments":{}}"#.utf8))
        XCTAssertEqual(execute.status, 503)
        XCTAssertNotNil(execute.body["error"])
    }

    func testMCPStdioClientDiscoversAndExecutesFakeServer() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()

        let statuses = await manager.serverStatuses()
        XCTAssertEqual(statuses.first?.state, "connected", statuses.first?.error ?? "")
        XCTAssertEqual(statuses.first?.toolsCount, 1)

        let tools = await manager.allTools()
        XCTAssertEqual(tools.count, 1)
        XCTAssertEqual(tools[0].fullName, "fake__echo")
        XCTAssertEqual(tools[0].description, "Echo input")

        let result = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("hello")])
        )
        XCTAssertFalse(result.isError)
        guard case .array(let content) = result.content,
            case .object(let first)? = content.first,
            case .string(let text)? = first["text"]
        else {
            XCTFail("expected MCP text content")
            return
        }
        XCTAssertEqual(text, "hello")
        await manager.shutdown()
    }

    func testMCPStreamableHTTPDiscoversAndExecutesWithHeaders() async throws {
        let server = try MockMCPHTTPServer(requiredAuthorization: "Bearer stream-token")
        defer { server.stop() }
        let manager = MCPManager(
            config: MCPConfig(
                servers: [
                    MCPServerConfig(
                        name: "remote",
                        transport: "streamable-http",
                        url: "\(server.baseURL)/mcp",
                        headers: ["Authorization": "Bearer stream-token"],
                        timeoutMs: 1_000
                    )
                ]
            )
        )
        await manager.connectAll()

        let statuses = await manager.serverStatuses()
        XCTAssertEqual(statuses.first?.state, "connected", statuses.first?.error ?? "")
        XCTAssertEqual(statuses.first?.transport, "streamable-http")
        XCTAssertEqual(statuses.first?.toolsCount, 1)

        let result = await manager.execute(
            toolName: "remote__echo",
            arguments: .object(["message": .string("over-http")])
        )
        XCTAssertFalse(result.isError)
        XCTAssertEqual(Self.textContent(result), "over-http")
        XCTAssertTrue(server.sawAuthorizedRequest)
        await manager.shutdown()
    }

    func testMCPSSEEndpointHandshakeDiscoversEndpointAndExecutesWithHeaders() async throws {
        let server = try MockMCPHTTPServer(requiredAuthorization: "Bearer sse-token")
        defer { server.stop() }
        let manager = MCPManager(
            config: MCPConfig(
                servers: [
                    MCPServerConfig(
                        name: "remote",
                        transport: "sse",
                        url: "\(server.baseURL)/sse",
                        headers: ["Authorization": "Bearer sse-token"],
                        timeoutMs: 1_000
                    )
                ]
            )
        )
        await manager.connectAll()

        let statuses = await manager.serverStatuses()
        XCTAssertEqual(statuses.first?.state, "connected")
        XCTAssertEqual(statuses.first?.transport, "sse-endpoint")
        XCTAssertEqual(statuses.first?.toolsCount, 1)

        let result = await manager.execute(
            toolName: "remote__echo",
            arguments: .object(["message": .string("over-sse")])
        )
        XCTAssertFalse(result.isError)
        XCTAssertEqual(Self.textContent(result), "over-sse")
        XCTAssertEqual(server.sseConnectCount, 1)
        XCTAssertTrue(server.sawAuthorizedRequest)
        await manager.shutdown()
    }

    func testMCPHandlerExecuteShapeWithFakeServer() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()
        let handler = MCPHandler(manager: manager)

        let response = await handler.executeResponse(
            body: Data(#"{"tool":"fake__echo","arguments":{"message":"route"}}"#.utf8)
        )

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(response.body["tool_name"] as? String, "fake__echo")
        XCTAssertEqual(response.body["is_error"] as? Bool, false)
        let content = try XCTUnwrap(response.body["content"] as? [[String: Any]])
        XCTAssertEqual(content.first?["text"] as? String, "route")
        await manager.shutdown()
    }

    func testMergedOpenAIChatRequestAddsDiscoveredTools() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()
        let userTool = OpenAIJSONValue(
            [
                "type": "function",
                "function": [
                    "name": "user_tool",
                    "parameters": ["type": "object", "properties": [:]],
                ],
            ]
        )

        let request = OpenAIChatRequest(
            model: "test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            maxTokens: 1,
            tools: userTool.map { [$0] }
        )
        let tools = await manager.mergedOpenAITools(
            userTools: request.tools,
            toolChoice: request.toolChoice
        )
        let mergedRequest = openAIChatRequestByReplacingTools(request, tools: tools)

        let names = (mergedRequest.tools ?? []).compactMap(openAIToolFunctionName).sorted()
        XCTAssertEqual(names, ["fake__echo", "user_tool"])
        await manager.shutdown()
    }

    func testMergedAnthropicMessagesRequestAddsDiscoveredTools() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()
        let request = try AnthropicMessagesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "max_tokens": 16,
                  "messages": [{"role": "user", "content": "hello"}],
                  "tools": [
                    {
                      "name": "user_tool",
                      "description": "User tool",
                      "input_schema": {"type": "object", "properties": {}}
                    }
                  ]
                }
                """.utf8
            )
        )

        let mergedRequest = await anthropicMessagesRequestByMergingMCPTools(request, manager: manager)
        let names = (mergedRequest.openAIRequest().tools ?? []).compactMap(openAIToolFunctionName).sorted()

        XCTAssertEqual(names, ["fake__echo", "user_tool"])
        await manager.shutdown()
    }

    func testMergedResponsesRequestAddsDiscoveredTools() async throws {
        let manager = try makeFakeMCPManager()
        await manager.connectAll()
        let request = try ResponsesRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "input": "hello",
                  "tools": [
                    {
                      "type": "function",
                      "name": "user_tool",
                      "description": "User tool",
                      "parameters": {"type": "object", "properties": {}}
                    }
                  ]
                }
                """.utf8
            )
        )

        let mergedRequest = await responsesRequestByMergingMCPTools(request, manager: manager)
        let names = (mergedRequest.openAIRequest().tools ?? []).compactMap(openAIToolFunctionName).sorted()

        XCTAssertEqual(names, ["fake__echo", "user_tool"])
        await manager.shutdown()
    }

    func testMCPStdioTimeoutReturnsToolErrorWithinDeadline() async throws {
        let manager = try makeFakeMCPManager(mode: "hang-call", timeoutMs: 150)
        await manager.connectAll()

        let started = Date()
        let result = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("never")])
        )
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertTrue(result.isError)
        XCTAssertEqual(result.errorMessage, "tool failed: timeout after 150ms")
        XCTAssertLessThan(elapsed, 1.5)
        let statuses = await manager.serverStatuses()
        XCTAssertEqual(statuses.first?.state, "error")
        XCTAssertEqual(statuses.first?.error, "tool failed: timeout after 150ms")
        await manager.shutdown()
    }

    func testMCPStdioSlowReplyWithinTimeoutSucceeds() async throws {
        let manager = try makeFakeMCPManager(mode: "slow-call", timeoutMs: 800)
        await manager.connectAll()

        let result = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("slow")])
        )

        XCTAssertFalse(result.isError)
        guard case .array(let content) = result.content,
            case .object(let first)? = content.first,
            case .string(let text)? = first["text"]
        else {
            XCTFail("expected MCP text content")
            return
        }
        XCTAssertEqual(text, "slow")
        await manager.shutdown()
    }

    func testMCPStdioTimeoutRespawnsServerOnNextUse() async throws {
        let markerURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxcat-mcp-respawn-\(UUID().uuidString)")
        let manager = try makeFakeMCPManager(mode: "hang-once", timeoutMs: 150, markerURL: markerURL)
        await manager.connectAll()

        let timedOut = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("first")])
        )
        XCTAssertTrue(timedOut.isError)
        XCTAssertEqual(timedOut.errorMessage, "tool failed: timeout after 150ms")

        let respawned = await manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("second")])
        )
        XCTAssertFalse(respawned.isError)
        guard case .array(let content) = respawned.content,
            case .object(let first)? = content.first,
            case .string(let text)? = first["text"]
        else {
            XCTFail("expected MCP text content")
            return
        }
        XCTAssertEqual(text, "second")
        let statuses = await manager.serverStatuses()
        XCTAssertEqual(statuses.first?.state, "connected")
        await manager.shutdown()
    }

    func testMCPStdioConcurrentCallsRouteOutOfOrderReplies() async throws {
        let manager = try makeFakeMCPManager(mode: "out-of-order", timeoutMs: 800)
        await manager.connectAll()

        async let slow = manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("slow")])
        )
        async let fast = manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("fast")])
        )

        let results = await (slow, fast)

        XCTAssertEqual(Self.textContent(results.0), "slow")
        XCTAssertEqual(Self.textContent(results.1), "fast")
        XCTAssertFalse(results.0.isError)
        XCTAssertFalse(results.1.isError)
        await manager.shutdown()
    }

    func testMCPStdioConcurrentFastCallCompletesWhileOtherCallTimesOut() async throws {
        let manager = try makeFakeMCPManager(mode: "hang-one-call", timeoutMs: 150)
        await manager.connectAll()

        async let hung = manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("hang")])
        )
        async let fast = manager.execute(
            toolName: "fake__echo",
            arguments: .object(["message": .string("fast")])
        )

        let results = await (hung, fast)

        XCTAssertTrue(results.0.isError)
        XCTAssertEqual(results.0.errorMessage, "tool failed: timeout after 150ms")
        XCTAssertFalse(results.1.isError)
        XCTAssertEqual(Self.textContent(results.1), "fast")
        await manager.shutdown()
    }

    private static func textContent(_ result: MCPToolExecutionResult) -> String? {
        guard case .array(let content) = result.content,
            case .object(let first)? = content.first,
            case .string(let text)? = first["text"]
        else {
            return nil
        }
        return text
    }

    private func makeFakeMCPManager(mode: String = "echo", timeoutMs: Int = 30_000, markerURL: URL? = nil) throws -> MCPManager {
        let scriptURL = try fakeMCPServerScriptURL()
        var args = [scriptURL.path, mode]
        if let markerURL {
            args.append(markerURL.path)
        }
        return MCPManager(
            config: MCPConfig(
                servers: [
                    MCPServerConfig(name: "fake", command: "python3", args: args, timeoutMs: timeoutMs)
                ]
            )
        )
    }

    private func fakeMCPServerScriptURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxcat-mcp-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let scriptURL = directory.appendingPathComponent("fake_mcp_server.py")
        try fakeMCPServerScript.write(to: scriptURL, atomically: true, encoding: .utf8)
        return scriptURL
    }

    private final class MockMCPHTTPServer: @unchecked Sendable {
        private let listener: NWListener
        private let requiredAuthorization: String
        private let queue = DispatchQueue(label: "MockMCPHTTPServer.state")
        private var _sseConnectCount = 0
        private var _sawAuthorizedRequest = false

        private(set) var baseURL: String = ""

        var sseConnectCount: Int {
            queue.sync { _sseConnectCount }
        }

        var sawAuthorizedRequest: Bool {
            queue.sync { _sawAuthorizedRequest }
        }

        init(requiredAuthorization: String) throws {
            self.requiredAuthorization = requiredAuthorization
            listener = try NWListener(using: .tcp, on: .any)
            listener.newConnectionHandler = { [weak self] connection in
                connection.start(queue: .global())
                Task {
                    await self?.handle(connection)
                }
            }
            let ready = DispatchSemaphore(value: 0)
            listener.stateUpdateHandler = { state in
                if case .ready = state {
                    ready.signal()
                }
            }
            listener.start(queue: .global())
            guard ready.wait(timeout: .now() + 2) == .success else {
                throw NSError(domain: "MockMCPHTTPServer", code: 2)
            }
            guard let port = listener.port else {
                throw NSError(domain: "MockMCPHTTPServer", code: 1)
            }
            baseURL = "http://127.0.0.1:\(port.rawValue)"
        }

        func stop() {
            listener.cancel()
        }

        private func handle(_ connection: NWConnection) async {
            do {
                let request = try await HTTPRequest.read(from: connection)
                let authorized = request.headers["authorization"] == requiredAuthorization
                    || request.headers["Authorization"] == requiredAuthorization
                if authorized {
                    queue.sync {
                        _sawAuthorizedRequest = true
                    }
                } else {
                    try await send(
                        status: 401,
                        contentType: "application/json",
                        body: #"{"error":"unauthorized"}"#,
                        connection: connection
                    )
                    return
                }

                if request.method == "GET", request.path == "/sse" {
                    queue.sync {
                        _sseConnectCount += 1
                    }
                    try await send(
                        status: 200,
                        contentType: "text/event-stream",
                        body: "event: endpoint\ndata: /message\n\n",
                        connection: connection
                    )
                    return
                }

                if request.method == "POST", request.path == "/mcp" || request.path == "/message" {
                    let body = try responseBody(for: request.body)
                    try await send(
                        status: 200,
                        contentType: "application/json",
                        body: body,
                        connection: connection
                    )
                    return
                }

                try await send(
                    status: 404,
                    contentType: "application/json",
                    body: #"{"error":"not found"}"#,
                    connection: connection
                )
            } catch {
                connection.cancel()
            }
        }

        private func responseBody(for body: Data) throws -> String {
            guard let request = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                return #"{"jsonrpc":"2.0","error":{"code":-32700,"message":"invalid json"}}"#
            }
            guard let id = request["id"] else {
                return "{}"
            }
            let method = request["method"] as? String
            let result: [String: Any]
            switch method {
            case "initialize":
                result = [
                    "protocolVersion": "2024-11-05",
                    "capabilities": [:],
                    "serverInfo": ["name": "mock", "version": "1"],
                ]
            case "tools/list":
                result = [
                    "tools": [
                        [
                            "name": "echo",
                            "description": "Echo input",
                            "inputSchema": [
                                "type": "object",
                                "properties": ["message": ["type": "string"]],
                            ],
                        ]
                    ]
                ]
            case "tools/call":
                let params = request["params"] as? [String: Any] ?? [:]
                let arguments = params["arguments"] as? [String: Any] ?? [:]
                result = [
                    "content": [
                        [
                            "type": "text",
                            "text": arguments["message"] as? String ?? "",
                        ]
                    ],
                    "isError": false,
                ]
            default:
                let response: [String: Any] = [
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": ["code": -32601, "message": "method not found"],
                ]
                return String(data: try JSONSerialization.data(withJSONObject: response), encoding: .utf8) ?? "{}"
            }
            let response: [String: Any] = [
                "jsonrpc": "2.0",
                "id": id,
                "result": result,
            ]
            return String(data: try JSONSerialization.data(withJSONObject: response), encoding: .utf8) ?? "{}"
        }

        private func send(
            status: Int,
            contentType: String,
            body: String,
            connection: NWConnection
        ) async throws {
            let reason = status == 200 ? "OK" : status == 401 ? "Unauthorized" : "Not Found"
            let data = Data(body.utf8)
            let header = "HTTP/1.1 \(status) \(reason)\r\n"
                + "Content-Type: \(contentType)\r\n"
                + "Content-Length: \(data.count)\r\n"
                + "Connection: close\r\n"
                + "\r\n"
            try await connection.sendFinal(data: Data(header.utf8) + data)
            connection.cancel()
        }
    }

    private var fakeMCPServerScript: String {
        #"""
        import json
        import os
        import sys
        import threading
        import time

        mode = sys.argv[1] if len(sys.argv) > 1 else "echo"
        marker_path = sys.argv[2] if len(sys.argv) > 2 else None
        write_lock = threading.Lock()

        def read_message():
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            return json.loads(line.decode("utf-8"))

        def write_message(message):
            body = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
            with write_lock:
                sys.stdout.buffer.write(body)
                sys.stdout.buffer.flush()

        def write_tool_response(request_id, arguments):
            if mode == "out-of-order":
                time.sleep(0.2 if arguments.get("message") == "slow" else 0.02)
            if mode == "hang-one-call" and arguments.get("message") == "hang":
                while True:
                    time.sleep(1)
            write_message({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": arguments.get("message", "")}],
                    "isError": False
                }
            })

        while True:
            message = read_message()
            if message is None:
                break
            method = message.get("method")
            request_id = message.get("id")
            if request_id is None:
                continue
            if method == "initialize":
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "fake", "version": "1"}
                    }
                })
            elif method == "tools/list":
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [{
                            "name": "echo",
                            "description": "Echo input",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}}
                            }
                        }]
                    }
                })
            elif method == "tools/call":
                arguments = message.get("params", {}).get("arguments", {})
                if mode == "hang-call":
                    while True:
                        time.sleep(1)
                if mode == "slow-call":
                    time.sleep(0.2)
                if mode == "hang-once" and marker_path and not os.path.exists(marker_path):
                    with open(marker_path, "w", encoding="utf-8") as marker:
                        marker.write("hung")
                    while True:
                        time.sleep(1)
                if mode in ("out-of-order", "hang-one-call"):
                    thread = threading.Thread(target=write_tool_response, args=(request_id, arguments), daemon=True)
                    thread.start()
                    continue
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": arguments.get("message", "")}],
                        "isError": False
                    }
                })
            else:
                write_message({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "method not found"}
                })
        """#
    }
}
