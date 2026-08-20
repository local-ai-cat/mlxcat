@testable import MLXCat
@testable import MLXCatHTTP
@testable import MLXCatNative
@testable import MLXCatHTTPServer
import XCTest

final class JSONGrammarBridgeTests: XCTestCase {
    func testConvertsTypePropertiesRequiredAndAdditionalProperties() {
        let node = JSONSchemaNode(openAISchema: [
            "type": .string("object"),
            "properties": .object([
                "answer": .object(["type": .string("string")]),
                "count": .object(["type": .string("integer")]),
            ]),
            "required": .array([.string("answer")]),
            "additionalProperties": .bool(false),
        ])

        XCTAssertEqual(
            node,
            .schema(
                JSONSchema(
                    allowedTypes: [.object],
                    properties: [
                        "answer": .schema(JSONSchema(allowedTypes: [.string])),
                        "count": .schema(JSONSchema(allowedTypes: [.number])),
                    ],
                    required: ["answer"],
                    additionalPropertiesAllowed: false
                )
            )
        )
    }

    func testConvertsEnumConstAndItems() {
        let enumNode = JSONSchemaNode(openAISchema: [
            "type": .string("string"),
            "enum": .array([.string("done"), .string("fail")]),
        ])
        XCTAssertEqual(
            enumNode,
            .schema(
                JSONSchema(
                    allowedTypes: [.string],
                    enumValues: [.string("done"), .string("fail")]
                )
            )
        )

        let constNode = JSONSchemaNode(openAISchema: ["const": .number(3)])
        XCTAssertEqual(constNode, .schema(JSONSchema(enumValues: [.number(3)])))

        let arrayNode = JSONSchemaNode(openAISchema: [
            "type": .string("array"),
            "items": .object(["type": .string("null")]),
        ])
        XCTAssertEqual(
            arrayNode,
            .schema(
                JSONSchema(
                    allowedTypes: [.array],
                    items: .schema(JSONSchema(allowedTypes: [.null]))
                )
            )
        )
    }

    func testUnsupportedConstraintsDegradePermissivelyNeverStricter() {
        // Unknown type name: drop the type constraint rather than forbid everything.
        XCTAssertEqual(
            JSONSchemaNode(openAISchema: ["type": .string("timestamp")]),
            .schema(JSONSchema())
        )

        // Enum containing a non-literal member: enforcing the rest would forbid that
        // member, so the enum constraint is dropped entirely.
        XCTAssertEqual(
            JSONSchemaNode(openAISchema: [
                "enum": .array([.string("a"), .object(["nested": .bool(true)])])
            ]),
            .schema(JSONSchema())
        )

        // patternProperties can allow keys beyond `properties`; enforcing
        // additionalProperties:false alongside it would forbid them.
        XCTAssertEqual(
            JSONSchemaNode(openAISchema: [
                "type": .string("object"),
                "patternProperties": .object(["^x_": .object([:])]),
                "additionalProperties": .bool(false),
            ]),
            .schema(JSONSchema(allowedTypes: [.object]))
        )

        // Non-object subschema (e.g. boolean "items": true) degrades to .any.
        XCTAssertEqual(
            JSONSchemaNode(openAISchema: [
                "type": .string("array"),
                "items": .bool(true),
            ]),
            .schema(JSONSchema(allowedTypes: [.array], items: .any))
        )
    }
}
