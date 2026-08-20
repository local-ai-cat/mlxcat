import Foundation
import Network

struct MCPHandler {
    let manager: MCPManager?

    func handleTools(connection: NWConnection) async throws {
        let response = await toolsResponse()
        try await sendJSON(response.body, status: response.status, connection: connection)
    }

    func handleServers(connection: NWConnection) async throws {
        let response = await serversResponse()
        try await sendJSON(response.body, status: response.status, connection: connection)
    }

    func handleExecute(_ request: HTTPRequest, connection: NWConnection) async throws {
        let response = await executeResponse(body: request.body)
        try await sendJSON(response.body, status: response.status, connection: connection)
    }

    func toolsResponse() async -> HTTPJSONResponse {
        guard let manager else {
            return HTTPJSONResponse(status: 200, body: ["tools": [], "count": 0])
        }
        let tools = await manager.allTools()
        return HTTPJSONResponse(
            status: 200,
            body: [
                "tools": tools.map(toolInfo),
                "count": tools.count,
            ]
        )
    }

    func serversResponse() async -> HTTPJSONResponse {
        guard let manager else {
            return HTTPJSONResponse(status: 200, body: ["servers": []])
        }
        let statuses = await manager.serverStatuses()
        return HTTPJSONResponse(
            status: 200,
            body: [
                "servers": statuses.map(serverInfo),
            ]
        )
    }

    func executeResponse(body: Data) async -> HTTPJSONResponse {
        guard let manager else {
            return HTTPJSONResponse(
                status: 503,
                body: openAIErrorBody(message: "MCP not configured. Start server with --mcp-config", status: 503)
            )
        }
        do {
            guard let object = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                throw OpenAIServerError.invalidJSON
            }
            let toolName = object["tool_name"] as? String ?? object["tool"] as? String
            guard let toolName, !toolName.isEmpty else {
                throw OpenAIServerError.missingField("tool_name")
            }
            let arguments = OpenAIJSONValue(object["arguments"] ?? [:]) ?? .object([:])
            let result = await manager.execute(toolName: toolName, arguments: arguments)
            return HTTPJSONResponse(
                status: 200,
                body: [
                    "tool_name": result.toolName,
                    "content": jsonObject(from: result.content),
                    "is_error": result.isError,
                    "error_message": result.errorMessage as Any? ?? NSNull(),
                ]
            )
        } catch {
            let status = (error as? OpenAIServerError)?.httpStatus ?? 500
            return HTTPJSONResponse(
                status: status,
                body: openAIErrorBody(message: String(describing: error), status: status)
            )
        }
    }

    private func toolInfo(_ tool: MCPTool) -> [String: Any] {
        [
            "name": tool.fullName,
            "description": tool.description,
            "server": tool.serverName,
            "parameters": jsonObject(from: tool.inputSchema),
        ]
    }

    private func serverInfo(_ status: MCPServerStatus) -> [String: Any] {
        [
            "name": status.name,
            "state": status.state,
            "transport": status.transport,
            "tools_count": status.toolsCount,
            "error": status.error as Any? ?? NSNull(),
        ]
    }

    private func sendJSON(_ object: [String: Any], status: Int, connection: NWConnection) async throws {
        let body = try JSONSerialization.data(withJSONObject: object, options: [])
        let reason = mcpReasonPhrase(status)
        let header = "HTTP/1.1 \(status) \(reason)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.count)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try await connection.sendFinal(data: Data(header.utf8) + body)
    }
}

private func mcpReasonPhrase(_ status: Int) -> String {
    switch status {
    case 200:
        return "OK"
    case 400:
        return "Bad Request"
    case 404:
        return "Not Found"
    case 422:
        return "Unprocessable Entity"
    case 500:
        return "Internal Server Error"
    case 503:
        return "Service Unavailable"
    default:
        return "OK"
    }
}
