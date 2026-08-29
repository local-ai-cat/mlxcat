import Foundation

public struct MCPServerConfig: Sendable, Equatable {
    public let name: String
    public let transport: String
    public let command: String
    public let args: [String]
    public let env: [String: String]
    public let url: String?
    public let headers: [String: String]
    public let timeoutMs: Int

    public init(
        name: String,
        transport: String = "stdio",
        command: String = "",
        args: [String] = [],
        env: [String: String] = [:],
        url: String? = nil,
        headers: [String: String] = [:],
        timeoutMs: Int = 30_000
    ) {
        self.name = name
        self.transport = Self.normalizeTransport(transport)
        self.command = command
        self.args = args
        self.env = env
        self.url = url
        self.headers = headers
        self.timeoutMs = timeoutMs
    }

    public static func normalizeTransport(_ transport: String) -> String {
        switch transport {
        case "streamable_http":
            return "streamable-http"
        case "sse":
            return "sse-endpoint"
        default:
            return transport
        }
    }
}

public struct MCPConfig: Sendable, Equatable {
    public let servers: [MCPServerConfig]

    public init(servers: [MCPServerConfig]) {
        self.servers = servers
    }

    public static func load(from url: URL) throws -> MCPConfig {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }
        return try parse(object)
    }

    public static func parse(_ object: [String: Any]) throws -> MCPConfig {
        let rawServers = object["mcpServers"] ?? object["servers"]
        guard let serversObject = rawServers as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }

        var servers: [MCPServerConfig] = []
        for name in serversObject.keys.sorted() {
            guard let serverObject = serversObject[name] as? [String: Any] else {
                throw OpenAIServerError.invalidJSON
            }
            let enabled = serverObject["enabled"] as? Bool ?? true
            guard enabled else { continue }
            let transport = serverObject["transport"] as? String ?? "stdio"
            let normalizedTransport = MCPServerConfig.normalizeTransport(transport)
            let args = serverObject["args"] as? [String] ?? []
            let env = serverObject["env"] as? [String: String] ?? [:]
            let url = serverObject["url"] as? String
            let headers = serverObject["headers"] as? [String: String] ?? [:]
            let timeoutSeconds = (serverObject["timeout"] as? Double)
                ?? (serverObject["timeout"] as? Int).map(Double.init)
            let timeoutMs = serverObject["timeoutMs"] as? Int
                ?? timeoutSeconds.map { Int($0 * 1000) }
                ?? 30_000
            guard timeoutMs > 0 else {
                throw OpenAIServerError.invalidJSON
            }
            switch normalizedTransport {
            case "stdio":
                guard let command = serverObject["command"] as? String, !command.isEmpty else {
                    throw OpenAIServerError.invalidJSON
                }
                servers.append(
                    MCPServerConfig(
                        name: name,
                        transport: normalizedTransport,
                        command: command,
                        args: args,
                        env: env,
                        headers: headers,
                        timeoutMs: timeoutMs
                    )
                )
            case "sse-endpoint", "streamable-http":
                guard let url, !url.isEmpty else {
                    throw OpenAIServerError.invalidJSON
                }
                servers.append(
                    MCPServerConfig(
                        name: name,
                        transport: normalizedTransport,
                        env: env,
                        url: url,
                        headers: headers,
                        timeoutMs: timeoutMs
                    )
                )
            default:
                throw OpenAIServerError.invalidJSON
            }
        }
        return MCPConfig(servers: servers)
    }
}

public struct MCPTool: Sendable, Equatable {
    public let serverName: String
    public let name: String
    public let description: String
    public let inputSchema: OpenAIJSONValue

    public var fullName: String {
        "\(serverName)__\(name)"
    }

    public init(serverName: String, name: String, description: String, inputSchema: OpenAIJSONValue) {
        self.serverName = serverName
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
    }
}

public struct MCPServerStatus: Sendable, Equatable {
    public let name: String
    public let state: String
    public let transport: String
    public let toolsCount: Int
    public let error: String?

    public init(name: String, state: String, transport: String, toolsCount: Int, error: String?) {
        self.name = name
        self.state = state
        self.transport = transport
        self.toolsCount = toolsCount
        self.error = error
    }
}

public struct MCPToolExecutionResult: Sendable, Equatable {
    public let toolName: String
    public let content: OpenAIJSONValue
    public let isError: Bool
    public let errorMessage: String?

    public init(toolName: String, content: OpenAIJSONValue, isError: Bool = false, errorMessage: String? = nil) {
        self.toolName = toolName
        self.content = content
        self.isError = isError
        self.errorMessage = errorMessage
    }
}

public actor MCPManager {
    private var clients: [String: MCPStdioClient]

    public init(config: MCPConfig) {
        var clients: [String: MCPStdioClient] = [:]
        for server in config.servers {
            clients[server.name] = MCPStdioClient(config: server)
        }
        self.clients = clients
    }

    public static func load(from url: URL) throws -> MCPManager {
        MCPManager(config: try MCPConfig.load(from: url))
    }

    public func connectAll() async {
        for name in clients.keys.sorted() {
            guard let client = clients[name] else { continue }
            do {
                try await client.connect()
            } catch {
                await client.markError(String(describing: error))
            }
        }
    }

    public func shutdown() async {
        for name in clients.keys.sorted() {
            guard let client = clients[name] else { continue }
            await client.shutdown()
        }
    }

    public func allTools() async -> [MCPTool] {
        var tools: [MCPTool] = []
        for name in clients.keys.sorted() {
            guard let client = clients[name] else { continue }
            tools.append(contentsOf: await client.tools)
        }
        return tools
    }

    public func serverStatuses() async -> [MCPServerStatus] {
        var statuses: [MCPServerStatus] = []
        for name in clients.keys.sorted() {
            guard let client = clients[name] else { continue }
            statuses.append(await client.status)
        }
        return statuses
    }

    public func execute(toolName: String, arguments: OpenAIJSONValue) async -> MCPToolExecutionResult {
        let (serverName, localName) = splitToolName(toolName)
        if !serverName.isEmpty {
            guard let client = clients[serverName] else {
                return MCPToolExecutionResult(
                    toolName: toolName,
                    content: .null,
                    isError: true,
                    errorMessage: "MCP server not found: \(serverName)"
                )
            }
            return await client.callTool(fullName: toolName, localName: localName, arguments: arguments)
        }

        for name in clients.keys.sorted() {
            guard let client = clients[name], await client.hasTool(named: localName) else { continue }
            return await client.callTool(fullName: toolName, localName: localName, arguments: arguments)
        }

        return MCPToolExecutionResult(
            toolName: toolName,
            content: .null,
            isError: true,
            errorMessage: "MCP tool not found: \(toolName)"
        )
    }

    public func mergedOpenAITools(
        userTools: [OpenAIJSONValue]?,
        toolChoice: OpenAIToolChoice?
    ) async -> [OpenAIJSONValue]? {
        if case .some(OpenAIToolChoice.none) = toolChoice {
            return nil
        }

        var mergedByName: [String: OpenAIJSONValue] = [:]
        for tool in await allTools() {
            mergedByName[tool.fullName] = openAITool(from: tool)
        }
        for tool in userTools ?? [] {
            guard let name = openAIToolFunctionName(tool) else { continue }
            mergedByName[name] = tool
        }
        let names = mergedByName.keys.sorted()
        let merged = names.compactMap { mergedByName[$0] }
        return merged.isEmpty ? nil : merged
    }

    private func splitToolName(_ toolName: String) -> (server: String, local: String) {
        guard let range = toolName.range(of: "__") else {
            return ("", toolName)
        }
        return (String(toolName[..<range.lowerBound]), String(toolName[range.upperBound...]))
    }

    private func openAITool(from tool: MCPTool) -> OpenAIJSONValue {
        .object(
            [
                "type": .string("function"),
                "function": .object(
                    [
                        "name": .string(tool.fullName),
                        "description": .string(tool.description),
                        "parameters": tool.inputSchema,
                    ]
                ),
            ]
        )
    }
}

public func openAIChatRequestByReplacingTools(
    _ request: OpenAIChatRequest,
    tools: [OpenAIJSONValue]?
) -> OpenAIChatRequest {
    OpenAIChatRequest(
        model: request.model,
        messages: request.messages,
        maxTokens: request.maxTokens,
        temperature: request.temperature,
        topP: request.topP,
        topK: request.topK,
        repetitionPenalty: request.repetitionPenalty,
        minP: request.minP,
        xtcProbability: request.xtcProbability,
        xtcThreshold: request.xtcThreshold,
        presencePenalty: request.presencePenalty,
        frequencyPenalty: request.frequencyPenalty,
        minTokens: request.minTokens,
        logitBias: request.logitBias,
        stop: request.stop,
        seed: request.seed,
        stream: request.stream,
        includeUsage: request.includeUsage,
        enableThinking: request.enableThinking,
        thinkingBudget: request.thinkingBudget,
        chatTemplateKwargs: request.chatTemplateKwargs,
        structuredOutput: request.structuredOutput,
        tools: tools,
        toolChoice: request.toolChoice,
        cacheSession: request.cacheSession
    )
}

public func openAIChatRequestByMergingMCPTools(
    _ request: OpenAIChatRequest,
    manager: MCPManager?
) async -> OpenAIChatRequest {
    guard let manager else {
        return request
    }
    let tools = await manager.mergedOpenAITools(
        userTools: request.tools,
        toolChoice: request.toolChoice
    )
    return openAIChatRequestByReplacingTools(request, tools: tools)
}

public func anthropicMessagesRequestByMergingMCPTools(
    _ request: AnthropicMessagesRequest,
    manager: MCPManager?
) async -> AnthropicMessagesRequest {
    guard let manager else {
        return request
    }
    let tools = await manager.mergedOpenAITools(
        userTools: request.tools,
        toolChoice: request.toolChoice
    )
    return anthropicMessagesRequestByReplacingTools(request, tools: tools)
}

public func anthropicMessagesRequestByReplacingTools(
    _ request: AnthropicMessagesRequest,
    tools: [OpenAIJSONValue]?
) -> AnthropicMessagesRequest {
    AnthropicMessagesRequest(
        model: request.model,
        maxTokens: request.maxTokens,
        messages: request.messages,
        stopSequences: request.stopSequences,
        stream: request.stream,
        temperature: request.temperature,
        topP: request.topP,
        topK: request.topK,
        enableThinking: request.enableThinking,
        thinkingBudget: request.thinkingBudget,
        chatTemplateKwargs: request.chatTemplateKwargs,
        tools: tools,
        toolChoice: request.toolChoice,
        toolGrammar: request.toolGrammar
    )
}

public func responsesRequestByMergingMCPTools(
    _ request: ResponsesRequest,
    manager: MCPManager?
) async -> ResponsesRequest {
    guard let manager else {
        return request
    }
    let tools = await manager.mergedOpenAITools(
        userTools: request.tools,
        toolChoice: request.toolChoice
    )
    return responsesRequestByReplacingTools(request, tools: tools)
}

public func responsesRequestByReplacingTools(
    _ request: ResponsesRequest,
    tools: [OpenAIJSONValue]?
) -> ResponsesRequest {
    ResponsesRequest(
        model: request.model,
        inputMessages: request.inputMessages,
        temperature: request.temperature,
        topP: request.topP,
        maxOutputTokens: request.maxOutputTokens,
        stream: request.stream,
        text: request.text,
        previousResponseID: request.previousResponseID,
        store: request.store,
        metadata: request.metadata,
        seed: request.seed,
        thinkingBudget: request.thinkingBudget,
        chatTemplateKwargs: request.chatTemplateKwargs,
        tools: tools,
        toolChoice: request.toolChoice,
        toolGrammar: request.toolGrammar
    )
}

private actor MCPStdioClient {
    private let config: MCPServerConfig
    // The stdio transport spawns a child process — macOS-only. iOS builds keep
    // the HTTP transports (sse-endpoint, streamable-http) and refuse stdio.
    #if os(macOS)
    private var process: Process?
    #endif
    private var input: FileHandle?
    private var output: FileHandle?
    private var sseEndpointURL: URL?
    private var nextID = 1
    private var state = "disconnected"
    private var error: String?
    private var discoveredTools: [MCPTool] = []
    private var readBuffer = Data()
    private var streamError: Error?
    private var pendingRequests: [Int: PendingRequest] = [:]

    init(config: MCPServerConfig) {
        self.config = config
    }

    var tools: [MCPTool] {
        discoveredTools
    }

    var status: MCPServerStatus {
        MCPServerStatus(
            name: config.name,
            state: state,
            transport: config.transport,
            toolsCount: discoveredTools.count,
            error: error
        )
    }

    func markError(_ message: String) {
        state = "error"
        error = message
        discoveredTools = []
    }

    func shutdown() {
        closeProcess()
        #if os(macOS)
        process = nil
        #endif
        sseEndpointURL = nil
        state = "disconnected"
    }

    func hasTool(named name: String) -> Bool {
        discoveredTools.contains { $0.name == name || $0.fullName == name }
    }

    func connect() async throws {
        guard state != "connected" else { return }
        state = "connecting"
        error = nil
        streamError = nil
        readBuffer = Data()

        switch config.transport {
        case "stdio":
            try connectStdio()
        case "sse-endpoint":
            sseEndpointURL = try await discoverSSEEndpoint()
        case "streamable-http":
            guard config.url.flatMap(URL.init(string:)) != nil else {
                throw MCPStdioError.invalidURL(config.url ?? "")
            }
        default:
            throw MCPStdioError.unsupportedTransport(config.transport)
        }

        _ = try await sendRequest(
            method: "initialize",
            params: [
                "protocolVersion": "2024-11-05",
                "capabilities": [:],
                "clientInfo": ["name": "MLXCat", "version": "0"],
            ]
        )
        try await sendNotification(method: "notifications/initialized", params: [:])
        let toolsResult = try await sendRequest(method: "tools/list", params: [:])
        discoveredTools = parseTools(from: toolsResult)
        state = "connected"
    }

    #if os(macOS)
    private func connectStdio() throws {
        guard process == nil else { return }
        let process = Process()
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        if config.command.contains("/") {
            process.executableURL = URL(fileURLWithPath: config.command)
            process.arguments = config.args
        } else {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [config.command] + config.args
        }
        var environment = ProcessInfo.processInfo.environment
        for (key, value) in config.env {
            environment[key] = value
        }
        process.environment = environment
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = FileHandle.nullDevice
        try process.run()

        self.process = process
        input = stdinPipe.fileHandleForWriting
        output = stdoutPipe.fileHandleForReading
        output?.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            Task { await self?.receiveOutputData(data) }
        }
    }
    #else
    private func connectStdio() throws {
        throw MCPStdioError.stdioUnavailableOnPlatform
    }
    #endif

    func callTool(fullName: String, localName: String, arguments: OpenAIJSONValue) async -> MCPToolExecutionResult {
        do {
            if state != "connected" {
                try await connect()
            }
            let result = try await sendRequest(
                method: "tools/call",
                params: [
                    "name": localName,
                    "arguments": jsonObject(from: arguments),
                ]
            )
            let resultObject = result as? [String: Any] ?? [:]
            let isError = resultObject["isError"] as? Bool ?? false
            let contentValue = resultObject["content"] ?? result
            let content = OpenAIJSONValue(contentValue) ?? .null
            return MCPToolExecutionResult(
                toolName: fullName,
                content: content,
                isError: isError,
                errorMessage: isError ? mcpErrorMessage(from: content) : nil
            )
        } catch {
            return MCPToolExecutionResult(
                toolName: fullName,
                content: .null,
                isError: true,
                errorMessage: String(describing: error)
            )
        }
    }

    private func sendRequest(method: String, params: [String: Any]) async throws -> Any {
        let id = nextID
        nextID += 1
        guard config.transport == "stdio" else {
            return try await sendHTTPRequest(id: id, method: method, params: params)
        }
        return try await sendStdioRequest(id: id, method: method, params: params)
    }

    private func sendStdioRequest(id: Int, method: String, params: [String: Any]) async throws -> Any {
        try writeMessage(
            [
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            ]
        )

        let timeoutMs = config.timeoutMs

        do {
            let responseLine: Data = try await withTaskCancellationHandler(
                operation: {
                    try await withCheckedThrowingContinuation { continuation in
                        let timeoutTask = Task {
                            do {
                                try await Task.sleep(nanoseconds: UInt64(timeoutMs) * 1_000_000)
                            } catch {
                                return
                            }
                            self.timeoutRequest(id: id, timeoutMs: timeoutMs)
                        }
                        pendingRequests[id] = PendingRequest(
                            continuation: continuation,
                            timeoutTask: timeoutTask
                        )
                    }
                },
                onCancel: {
                    Task { await self.cancelRequest(id: id) }
                }
            )
            let response = try parseMessageLine(responseLine)
            if let error = response["error"] as? [String: Any] {
                throw MCPStdioError.rpcError(error["message"] as? String ?? String(describing: error))
            }
            return response["result"] ?? [:]
        } catch let error as MCPStdioError {
            if case .timeout = error {
                resetAfterTimeout(error.description)
            }
            throw error
        }
    }

    private func sendHTTPRequest(id: Int?, method: String, params: [String: Any]) async throws -> Any {
        let message: [String: Any]
        if let id {
            message = [
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            ]
        } else {
            message = [
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            ]
        }

        if id == nil {
            _ = try await postHTTPRequest(message: message)
            return [:]
        }

        var lastError: Error?
        for attempt in 0 ..< 2 {
            do {
                let data = try await postHTTPRequest(message: message)
                let response = try parseHTTPResponseData(data)
                if let error = response["error"] as? [String: Any] {
                    throw MCPStdioError.rpcError(error["message"] as? String ?? String(describing: error))
                }
                return response["result"] ?? [:]
            } catch {
                lastError = error
                guard attempt == 0, !isTimeout(error) else { break }
                if config.transport == "sse-endpoint" {
                    sseEndpointURL = nil
                    try? await Task.sleep(nanoseconds: 100_000_000)
                    sseEndpointURL = try await discoverSSEEndpoint()
                } else {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
            }
        }
        throw lastError ?? MCPStdioError.invalidMessage
    }

    private func postHTTPRequest(message: [String: Any]) async throws -> Data {
        let url = try await requestURL()
        let timeoutMs = config.timeoutMs
        let headers = config.headers
        let body = try JSONSerialization.data(withJSONObject: message, options: [])
        return try await withTimeout(milliseconds: timeoutMs) {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.timeoutInterval = TimeInterval(timeoutMs) / 1000.0
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
            for (name, value) in headers {
                request.setValue(value, forHTTPHeaderField: name)
            }
            request.httpBody = body
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, 200 ..< 300 ~= http.statusCode else {
                let status = (response as? HTTPURLResponse)?.statusCode ?? -1
                throw MCPStdioError.httpStatus(status)
            }
            return data
        }
    }

    private func requestURL() async throws -> URL {
        switch config.transport {
        case "sse-endpoint":
            if let sseEndpointURL {
                return sseEndpointURL
            }
            let endpoint = try await discoverSSEEndpoint()
            sseEndpointURL = endpoint
            return endpoint
        case "streamable-http":
            guard let urlString = config.url, let url = URL(string: urlString) else {
                throw MCPStdioError.invalidURL(config.url ?? "")
            }
            return url
        default:
            throw MCPStdioError.unsupportedTransport(config.transport)
        }
    }

    private func discoverSSEEndpoint() async throws -> URL {
        guard let urlString = config.url, let url = URL(string: urlString) else {
            throw MCPStdioError.invalidURL(config.url ?? "")
        }
        let timeoutMs = config.timeoutMs
        let headers = config.headers
        return try await withTimeout(milliseconds: timeoutMs) {
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = TimeInterval(timeoutMs) / 1000.0
            request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
            for (name, value) in headers {
                request.setValue(value, forHTTPHeaderField: name)
            }
            let (bytes, response) = try await URLSession.shared.bytes(for: request)
            guard let http = response as? HTTPURLResponse, 200 ..< 300 ~= http.statusCode else {
                let status = (response as? HTTPURLResponse)?.statusCode ?? -1
                throw MCPStdioError.httpStatus(status)
            }
            var buffer = Data()
            for try await byte in bytes {
                buffer.append(byte)
                if buffer.containsSSEEventTerminator,
                    let endpoint = try Self.endpointURL(fromSSEData: buffer, baseURL: url)
                {
                    return endpoint
                }
            }
            if let endpoint = try Self.endpointURL(fromSSEData: buffer, baseURL: url) {
                return endpoint
            }
            throw MCPStdioError.invalidMessage
        }
    }

    private func parseHTTPResponseData(_ data: Data) throws -> [String: Any] {
        if let direct = try? parseMessageLine(data) {
            return direct
        }
        guard let payload = Self.firstSSEData(in: data) else {
            throw MCPStdioError.invalidMessage
        }
        return try parseMessageLine(payload)
    }

    private nonisolated static func firstSSEData(in data: Data) -> Data? {
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        for event in text.components(separatedBy: "\n\n") {
            let dataLines = event
                .split(separator: "\n")
                .map(String.init)
                .filter { $0.hasPrefix("data:") }
                .map { String($0.dropFirst("data:".count)).trimmingCharacters(in: .whitespaces) }
            guard !dataLines.isEmpty else { continue }
            let payload = dataLines.joined(separator: "\n")
            if payload == "[DONE]" { continue }
            return Data(payload.utf8)
        }
        return nil
    }

    private nonisolated static func endpointURL(fromSSEData data: Data, baseURL: URL) throws -> URL? {
        guard let payload = firstSSEData(in: data),
            let endpoint = String(data: payload, encoding: .utf8),
            !endpoint.isEmpty
        else {
            return nil
        }
        guard let url = URL(string: endpoint, relativeTo: baseURL)?.absoluteURL else {
            throw MCPStdioError.invalidURL(endpoint)
        }
        return url
    }

    private func withTimeout<T: Sendable>(
        milliseconds: Int,
        operation: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await operation()
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(milliseconds) * 1_000_000)
                throw MCPStdioError.timeout(milliseconds)
            }
            guard let result = try await group.next() else {
                throw MCPStdioError.timeout(milliseconds)
            }
            group.cancelAll()
            return result
        }
    }

    private func isTimeout(_ error: Error) -> Bool {
        if case MCPStdioError.timeout = error {
            return true
        }
        return false
    }

    private func timeoutRequest(id: Int, timeoutMs: Int) {
        guard pendingRequests[id] != nil else { return }
        let error = MCPStdioError.timeout(timeoutMs)
        streamError = error
        failAllPendingRequests(error)
        resetAfterTimeout(error.description)
    }

    private func sendNotification(method: String, params: [String: Any]) async throws {
        guard config.transport == "stdio" else {
            _ = try await sendHTTPRequest(id: nil, method: method, params: params)
            return
        }
        try writeMessage(
            [
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            ]
        )
    }

    private func writeMessage(_ message: [String: Any]) throws {
        guard let input else {
            throw MCPStdioError.notConnected
        }
        let body = try JSONSerialization.data(withJSONObject: message, options: [])
        input.write(body + Data("\n".utf8))
    }

    private func receiveOutputData(_ data: Data) {
        guard !data.isEmpty else {
            streamError = MCPStdioError.closed
            failAllPendingRequests(MCPStdioError.closed)
            return
        }

        readBuffer.append(data)
        while let newline = readBuffer.firstIndex(of: UInt8(ascii: "\n")) {
            let lineRange = readBuffer.startIndex..<newline
            var line = Data(readBuffer[lineRange])
            let nextIndex = readBuffer.index(after: newline)
            readBuffer.removeSubrange(readBuffer.startIndex..<nextIndex)
            if line.last == UInt8(ascii: "\r") {
                line.removeLast()
            }
            guard !line.isEmpty else {
                continue
            }
            routeMessageLine(line)
        }
    }

    private func routeMessageLine(_ line: Data) {
        do {
            let response = try parseMessageLine(line)
            guard let responseID = response["id"] as? Int else {
                return
            }
            guard let pending = pendingRequests.removeValue(forKey: responseID) else {
                return
            }
            pending.timeoutTask.cancel()
            pending.continuation.resume(returning: line)
        } catch {
            failAllPendingRequests(error)
        }
    }

    private func cancelRequest(id: Int) {
        guard let pending = pendingRequests.removeValue(forKey: id) else { return }
        pending.timeoutTask.cancel()
        pending.continuation.resume(throwing: CancellationError())
    }

    private func failAllPendingRequests(_ error: Error) {
        let pending = pendingRequests
        pendingRequests.removeAll()
        for request in pending.values {
            request.timeoutTask.cancel()
            request.continuation.resume(throwing: error)
        }
    }

    private func resetAfterTimeout(_ message: String) {
        closeProcess()
        #if os(macOS)
        process = nil
        #endif
        readBuffer = Data()
        streamError = nil
        state = "error"
        error = message
    }

    private func closeProcess() {
        output?.readabilityHandler = nil
        input?.closeFile()
        output?.closeFile()
        #if os(macOS)
        if let process, process.isRunning {
            process.terminate()
        }
        #endif
        input = nil
        output = nil
        failAllPendingRequests(MCPStdioError.closed)
    }

    private func parseMessageLine(_ line: Data) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: line) as? [String: Any] else {
            throw MCPStdioError.invalidMessage
        }
        return object
    }

    private func parseTools(from result: Any) -> [MCPTool] {
        guard let object = result as? [String: Any], let tools = object["tools"] as? [[String: Any]] else {
            return []
        }
        return tools.compactMap { tool in
            guard let name = tool["name"] as? String, !name.isEmpty else {
                return nil
            }
            let description = tool["description"] as? String ?? ""
            let rawSchema = tool["inputSchema"] ?? [
                "type": "object",
                "properties": [:],
            ]
            let inputSchema = OpenAIJSONValue(rawSchema) ?? .object(
                [
                    "type": .string("object"),
                    "properties": .object([:]),
                ]
            )
            return MCPTool(
                serverName: config.name,
                name: name,
                description: description,
                inputSchema: inputSchema
            )
        }
    }

    private func mcpErrorMessage(from content: OpenAIJSONValue) -> String? {
        if case .string(let string) = content {
            return string
        }
        if case .array(let parts) = content {
            for part in parts {
                guard case .object(let object) = part, case .string(let text)? = object["text"] else {
                    continue
                }
                return text
            }
        }
        return nil
    }
}

private struct PendingRequest {
    let continuation: CheckedContinuation<Data, Error>
    let timeoutTask: Task<Void, Never>
}

private enum MCPStdioError: Error, CustomStringConvertible {
    case notConnected
    case closed
    case invalidMessage
    case rpcError(String)
    case timeout(Int)
    case invalidURL(String)
    case httpStatus(Int)
    case unsupportedTransport(String)
    case stdioUnavailableOnPlatform

    var description: String {
        switch self {
        case .notConnected:
            return "MCP stdio client is not connected"
        case .closed:
            return "MCP stdio server closed its output"
        case .invalidMessage:
            return "invalid MCP stdio message"
        case .rpcError(let message):
            return "MCP JSON-RPC error: \(message)"
        case .timeout(let timeoutMs):
            return "tool failed: timeout after \(timeoutMs)ms"
        case .invalidURL(let url):
            return "invalid MCP URL: \(url)"
        case .httpStatus(let status):
            return "MCP HTTP transport returned status \(status)"
        case .unsupportedTransport(let transport):
            return "unsupported MCP transport: \(transport)"
        case .stdioUnavailableOnPlatform:
            return "MCP stdio transport is unavailable on this platform (no subprocesses); use streamable-http"
        }
    }
}

private extension Data {
    var containsSSEEventTerminator: Bool {
        contains(Data("\n\n".utf8)) || contains(Data("\r\n\r\n".utf8))
    }
}

func jsonObject(from value: OpenAIJSONValue) -> Any {
    switch value {
    case .string(let string):
        return string
    case .number(let number):
        return number
    case .bool(let bool):
        return bool
    case .object(let object):
        return object.mapValues { jsonObject(from: $0) }
    case .array(let array):
        return array.map { jsonObject(from: $0) }
    case .null:
        return NSNull()
    }
}
