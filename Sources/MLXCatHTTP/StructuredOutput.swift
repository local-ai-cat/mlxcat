import Foundation

/// Structured output modes observable at the HTTP layer.
///
/// `.choice`, `.jsonObject`, `.jsonSchema`, `.regex`, and `.grammar` all run as true
/// constrained decoding in the native sampler.
public enum StructuredOutputSpec: Sendable, Equatable {
    case none
    case jsonObject
    case jsonSchema(name: String?, schema: [String: OpenAIJSONValue])
    case choice([String])
    case regex(pattern: String)
    case grammar(String)
}

enum StructuredOutputParser {
    static func parse(from object: [String: Any]) throws -> StructuredOutputSpec {
        if hasNonNullValue(object["guided_grammar"]) {
            guard let grammar = object["guided_grammar"] as? String, !grammar.isEmpty else {
                throw OpenAIServerError.invalidJSON
            }
            return .grammar(grammar)
        }

        // omlx-specific guided controls are more explicit than OpenAI response_format.
        if let rawStructuredOutputs = object["structured_outputs"], !(rawStructuredOutputs is NSNull) {
            return try parseStructuredOutputs(rawStructuredOutputs)
        }

        if let rawResponseFormat = object["response_format"], !(rawResponseFormat is NSNull) {
            return try parseResponseFormat(rawResponseFormat)
        }

        return .none
    }

    private static func parseResponseFormat(_ value: Any) throws -> StructuredOutputSpec {
        guard let object = value as? [String: Any],
            let type = object["type"] as? String
        else {
            throw OpenAIServerError.invalidJSON
        }

        switch type {
        case "text":
            return .none
        case "json_object":
            return .jsonObject
        case "json_schema":
            guard let jsonSchema = object["json_schema"] as? [String: Any],
                let rawSchema = jsonSchema["schema"] as? [String: Any]
            else {
                throw OpenAIServerError.invalidJSON
            }
            return .jsonSchema(
                name: jsonSchema["name"] as? String,
                schema: try convertJSONObject(rawSchema)
            )
        default:
            throw OpenAIServerError.invalidJSON
        }
    }

    private static func parseStructuredOutputs(_ value: Any) throws -> StructuredOutputSpec {
        guard let object = value as? [String: Any],
            let type = object["type"] as? String
        else {
            return try parseStructuredOutputAliases(value)
        }

        switch type {
        case "json":
            return .jsonObject
        case "choice":
            guard let choices = object["choices"] as? [String] else {
                throw OpenAIServerError.invalidJSON
            }
            guard !choices.isEmpty else {
                throw OpenAIServerError.invalidStructuredOutput("structured_outputs.choice requires at least one choice")
            }
            return .choice(choices)
        case "regex":
            guard let pattern = object["pattern"] as? String ?? object["regex"] as? String,
                !pattern.isEmpty
            else {
                throw OpenAIServerError.invalidJSON
            }
            return .regex(pattern: pattern)
        case "grammar":
            guard let grammar = object["grammar"] as? String ?? object["gbnf"] as? String,
                !grammar.isEmpty
            else {
                throw OpenAIServerError.invalidJSON
            }
            return .grammar(grammar)
        default:
            throw OpenAIServerError.invalidJSON
        }
    }

    private static func parseStructuredOutputAliases(_ value: Any) throws -> StructuredOutputSpec {
        guard let object = value as? [String: Any] else {
            throw OpenAIServerError.invalidJSON
        }

        if let rawJSON = object["json"] ?? object["json_schema"] {
            guard let schema = rawJSON as? [String: Any] else {
                throw OpenAIServerError.invalidJSON
            }
            return .jsonSchema(name: nil, schema: try convertJSONObject(schema))
        }

        if let choices = object["choice"] as? [String] ?? object["choices"] as? [String] {
            guard !choices.isEmpty else {
                throw OpenAIServerError.invalidStructuredOutput("structured_outputs.choice requires at least one choice")
            }
            return .choice(choices)
        }

        if let pattern = object["regex"] as? String, !pattern.isEmpty {
            return .regex(pattern: pattern)
        }

        if let grammar = object["grammar"] as? String ?? object["gbnf"] as? String, !grammar.isEmpty {
            return .grammar(grammar)
        }

        throw OpenAIServerError.invalidJSON
    }

    private static func convertJSONObject(_ object: [String: Any]) throws -> [String: OpenAIJSONValue] {
        var converted: [String: OpenAIJSONValue] = [:]
        for (key, value) in object {
            guard let jsonValue = OpenAIJSONValue(value) else {
                throw OpenAIServerError.invalidJSON
            }
            converted[key] = jsonValue
        }
        return converted
    }

    private static func hasNonNullValue(_ value: Any?) -> Bool {
        guard let value else { return false }
        return !(value is NSNull)
    }
}
