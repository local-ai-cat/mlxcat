import Foundation

public func selectOpenAITools(
    tools: [OpenAIJSONValue]?,
    toolChoice: OpenAIToolChoice?
) -> [OpenAIJSONValue]? {
    guard let tools, !tools.isEmpty else { return nil }

    switch toolChoice {
    case .some(OpenAIToolChoice.none):
        return nil
    case .some(.function(let name)):
        // Prompt-level enforcement for now; constrained decoding is the future upgrade.
        let selectedTools = tools.filter { openAIToolFunctionName($0) == name }
        return selectedTools.isEmpty ? nil : selectedTools
    case .some(.auto), .some(.required), nil:
        return tools
    }
}

public func openAIToolFunctionName(_ tool: OpenAIJSONValue) -> String? {
    guard case .object(let toolObject) = tool,
        case .object(let functionObject)? = toolObject["function"],
        case .string(let name)? = functionObject["name"],
        !name.isEmpty
    else {
        return nil
    }
    return name
}
