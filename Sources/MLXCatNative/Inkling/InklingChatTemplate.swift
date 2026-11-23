import Foundation

/// Renders the publisher's Inkling 2 Mini chat_template.jinja. The generic
/// Jinja tojson filter cannot accept the template's separators and sort_keys.
enum InklingChatTemplate {
    enum RenderError: Error {
        case unsupportedRole
        case unsupportedPart
        case invalidReasoningEffort
        case invalidTool
    }

    static func render(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        reasoningEffort: (any Sendable)?
    ) throws -> String {
        let model = "<|message_model|>"
        let system = "<|message_system|>"
        let end = "<|end_message|>"
        let text = "<|content_text|>"
        let roles = [
            "system": system, "user": "<|message_user|>", "assistant": model, "tool": "<|message_tool|>",
        ]
        let effort = try effortText(reasoningEffort)
        let effortFrame = system + text + "Thinking effort level: " + effort + end
        var output = ""
        if let tools, !tools.isEmpty {
            let specifications = try tools.map { tool -> [String: any Sendable] in
                let function = tool["function"] as? [String: any Sendable] ?? tool
                guard let name = function["name"] as? String else { throw RenderError.invalidTool }
                return [
                    "name": name, "description": function["description"] as? String ?? "",
                    "parameters": function["parameters"] as? [String: any Sendable] ?? [:],
                    "type": tool["type"] as? String ?? "function",
                ]
            }
            output += system + "tool_declare<|content_xml|>" + (try json(specifications)) + end
        }

        var declaredNames: [String: String] = [:]
        for message in messages {
            for call in message["tool_calls"] as? [[String: any Sendable]] ?? [] {
                if let id = call["id"] as? String, let function = call["function"] as? [String: any Sendable],
                    let name = function["name"] as? String
                {
                    declaredNames[id] = name
                }
            }
        }

        var effortEmitted = false
        for message in messages {
            guard let role = message["role"] as? String, let roleToken = roles[role] else {
                throw RenderError.unsupportedRole
            }
            if !effortEmitted, role != "system" {
                output += effortFrame
                effortEmitted = true
            }
            let name = message["name"] as? String ?? ""
            var header = roleToken
            if role == "assistant" {
                header += name
            } else if role == "tool" {
                let resultName = (message["tool_call_id"] as? String).flatMap { declaredNames[$0] }
                header += name.isEmpty ? resultName ?? "" : name
            }
            let reasoning = message["reasoning_content"] as? String ?? ""
            let calls = message["tool_calls"] as? [[String: any Sendable]] ?? []
            if role == "assistant", !reasoning.isEmpty {
                output += header + "<|content_thinking|>" + reasoning + end
            }
            if let content = message["content"] as? String {
                if !content.isEmpty || role != "assistant" || (calls.isEmpty && reasoning.isEmpty) {
                    output += header + text + content + end
                }
            } else if let parts = message["content"] as? [any Sendable] {
                for part in parts {
                    if let string = part as? String {
                        output += header + text + string + end
                    } else if let part = part as? [String: any Sendable] {
                        switch part["type"] as? String ?? "text" {
                        case "text", "input_text":
                            output += header + text + (part["text"] as? String ?? "") + end
                        case "image", "input_image", "image_url":
                            guard role == "user" || role == "tool" else { throw RenderError.unsupportedPart }
                            output += header + "<|content_image|><|unused_200054|>" + end
                        default:
                            throw RenderError.unsupportedPart
                        }
                    }
                }
            }
            if role == "assistant" {
                for call in calls {
                    guard let function = call["function"] as? [String: any Sendable],
                        let recipient = function["name"] as? String, name.isEmpty || name == recipient
                    else { throw RenderError.invalidTool }
                    let arguments = function["arguments"] as? [String: any Sendable] ?? [:]
                    output += roleToken + recipient + "<|content_invoke_tool_json|>"
                        + "{\"name\":" + (try json(recipient)) + ",\"args\":" + (try json(arguments)) + "}" + end
                }
                output += "<|content_model_end_sampling|>"
            }
        }
        if !effortEmitted { output += effortFrame }
        return output
    }

    private static func effortText(_ effort: (any Sendable)?) throws -> String {
        guard let effort else { return "0.9" }
        let table = [
            "none": "0", "minimal": "0.1", "low": "0.2", "medium": "0.7",
            "high": "0.9", "xhigh": "0.99", "max": "0.99",
        ]
        if let name = effort as? String, let value = table[name.trimmingCharacters(in: .whitespacesAndNewlines)] {
            return value
        }
        if let number = effort as? Double, (0...0.99).contains(number) {
            return number == 0 ? "0" : String(number)
        }
        throw RenderError.invalidReasoningEffort
    }

    private static func json(_ value: Any) throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes, .fragmentsAllowed])
        return String(decoding: data, as: UTF8.self)
    }
}
