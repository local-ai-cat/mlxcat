import Foundation
import MLXCat
import MLXCatHTTP

extension JSONSchemaNode {
    /// Best-effort translation of an OpenAI `json_schema` payload into the native grammar
    /// schema. Unsupported keywords degrade to *permissive* nodes, never stricter ones —
    /// over-constraining could mask tokens a compliant response needs, so the invariant is:
    /// output is always valid JSON, and every constraint we do enforce is one the request
    /// asked for. (omlx likewise enforces a supported subset via xgrammar.)
    init(openAISchema schema: [String: OpenAIJSONValue]) {
        self = Self.node(from: .object(schema))
    }

    private static func node(from value: OpenAIJSONValue) -> JSONSchemaNode {
        guard case .object(let schema) = value else { return .any }

        let allowedTypes = kinds(from: schema["type"])

        var properties: [String: JSONSchemaNode] = [:]
        if case .object(let rawProperties)? = schema["properties"] {
            for (key, subschema) in rawProperties {
                properties[key] = node(from: subschema)
            }
        }

        var required: Set<String> = []
        if case .array(let rawRequired)? = schema["required"] {
            for item in rawRequired {
                if case .string(let name) = item {
                    required.insert(name)
                }
            }
        }

        var items: JSONSchemaNode?
        if let rawItems = schema["items"] {
            items = node(from: rawItems)
        }

        var additionalPropertiesAllowed = true
        if case .bool(false)? = schema["additionalProperties"] {
            // Keywords we don't implement can widen the allowed key set beyond
            // `properties` (patternProperties keys, combinator branches, $ref targets).
            // Enforcing additionalProperties:false alongside them would forbid keys the
            // schema allows — keep the key set open whenever any are present.
            let keySetWideningKeywords = [
                "patternProperties", "propertyNames", "unevaluatedProperties",
                "anyOf", "oneOf", "allOf", "if", "$ref",
            ]
            if keySetWideningKeywords.allSatisfy({ schema[$0] == nil }) {
                additionalPropertiesAllowed = false
            }
        }

        return .schema(
            JSONSchema(
                allowedTypes: allowedTypes,
                properties: properties,
                required: required,
                enumValues: enumValues(from: schema),
                items: items,
                additionalPropertiesAllowed: additionalPropertiesAllowed
            )
        )
    }

    private static func kinds(from value: OpenAIJSONValue?) -> Set<JSONValueKind>? {
        let names: [String]
        switch value {
        case .string(let name):
            names = [name]
        case .array(let rawNames):
            names = rawNames.compactMap { item in
                if case .string(let name) = item { return name }
                return nil
            }
        default:
            return nil
        }

        var kinds: Set<JSONValueKind> = []
        for name in names {
            switch name {
            case "object": kinds.insert(.object)
            case "array": kinds.insert(.array)
            case "string": kinds.insert(.string)
            case "number", "integer": kinds.insert(.number)
            case "boolean": kinds.insert(.boolean)
            case "null": kinds.insert(.null)
            default:
                // Unknown type name — stay permissive rather than over-constrain.
                return nil
            }
        }
        return kinds.isEmpty ? nil : kinds
    }

    private static func enumValues(from schema: [String: OpenAIJSONValue]) -> Set<JSONLiteral>? {
        if let constValue = schema["const"] {
            guard let literal = literal(from: constValue) else { return nil }
            return [literal]
        }
        guard case .array(let rawEnum)? = schema["enum"], !rawEnum.isEmpty else { return nil }
        var literals: Set<JSONLiteral> = []
        for item in rawEnum {
            guard let literal = literal(from: item) else {
                // An enum member we can't represent (object/array literal): enforcing the
                // rest would forbid that member — drop the enum constraint entirely.
                return nil
            }
            literals.insert(literal)
        }
        return literals
    }

    private static func literal(from value: OpenAIJSONValue) -> JSONLiteral? {
        switch value {
        case .string(let string): return .string(string)
        case .number(let number): return .number(number)
        case .bool(let bool): return .bool(bool)
        case .null: return .null
        case .object, .array: return nil
        }
    }
}
