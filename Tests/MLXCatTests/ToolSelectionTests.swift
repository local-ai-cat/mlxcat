@testable import MLXCatHTTP
import XCTest

final class ToolSelectionTests: XCTestCase {
    func testNilToolsSelectsNil() {
        XCTAssertNil(selectOpenAITools(tools: nil, toolChoice: nil))
    }

    func testEmptyToolsSelectsNil() {
        XCTAssertNil(selectOpenAITools(tools: [], toolChoice: .auto))
    }

    func testToolChoiceNoneSelectsNil() {
        XCTAssertNil(selectOpenAITools(tools: sampleTools(), toolChoice: OpenAIToolChoice.none))
    }

    func testNilToolChoiceSelectsAllTools() {
        XCTAssertEqual(selectOpenAITools(tools: sampleTools(), toolChoice: nil), sampleTools())
    }

    func testAutoToolChoiceSelectsAllTools() {
        XCTAssertEqual(selectOpenAITools(tools: sampleTools(), toolChoice: .auto), sampleTools())
    }

    func testRequiredToolChoiceSelectsAllTools() {
        XCTAssertEqual(selectOpenAITools(tools: sampleTools(), toolChoice: .required), sampleTools())
    }

    func testFunctionToolChoiceFiltersToNamedTool() {
        XCTAssertEqual(selectOpenAITools(tools: sampleTools(), toolChoice: .function("get_weather")), [weatherTool])
    }

    func testMissingFunctionToolChoiceSelectsNil() {
        XCTAssertNil(selectOpenAITools(tools: sampleTools(), toolChoice: .function("missing")))
    }

    func testOpenAIToolFunctionNameReadsNestedName() {
        XCTAssertEqual(openAIToolFunctionName(weatherTool), "get_weather")
    }

    func testOpenAIToolFunctionNameReturnsNilForMissingName() {
        let tool = OpenAIJSONValue.object([
            "type": .string("function"),
            "function": .object(["description": .string("No name")]),
        ])

        XCTAssertNil(openAIToolFunctionName(tool))
    }

    private let weatherTool = OpenAIJSONValue.object([
        "type": .string("function"),
        "function": .object([
            "name": .string("get_weather"),
            "description": .string("Get weather"),
            "parameters": .object(["type": .string("object")]),
        ]),
    ])

    private let searchTool = OpenAIJSONValue.object([
        "type": .string("function"),
        "function": .object([
            "name": .string("search"),
            "description": .string("Search docs"),
            "parameters": .object(["type": .string("object")]),
        ]),
    ])

    private func sampleTools() -> [OpenAIJSONValue] {
        [weatherTool, searchTool]
    }
}
