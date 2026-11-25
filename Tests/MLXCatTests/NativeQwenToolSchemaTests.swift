import Jinja
import MLXCatNative
import XCTest

final class NativeQwenToolSchemaTests: XCTestCase {
    func testRendersWeatherDeclarationInReferenceOrder() throws {
        let city: [String: any Sendable] = ["type": "string"]
        let parameters: [String: any Sendable] = [
            "required": ["city"],
            "properties": ["city": city],
            "type": "object",
        ]
        let function: [String: any Sendable] = [
            "parameters": parameters,
            "description": "Get weather for a city.",
            "name": "get_weather",
        ]
        let tool: [String: any Sendable] = ["function": function, "type": "function"]
        let template = try Template("{{ tool | tojson }}")
        let rendered = try template.render(["tool": NativeQwenToolSchema.value(tool)])
        XCTAssertEqual(
            rendered,
            #"{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#
        )
    }
}
