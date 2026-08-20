import Foundation

public enum OpenAIToolChoice: Sendable, Equatable {
    case none
    case auto
    case required
    case function(String)

    public static func parse(_ value: Any?) throws -> OpenAIToolChoice? {
        guard let value else { return nil }
        if value is NSNull {
            return nil
        }

        if let string = value as? String {
            switch string {
            case "none":
                // Explicit enum type: a bare `.none` would resolve to Optional.none (nil).
                return OpenAIToolChoice.none
            case "auto":
                return .auto
            case "required":
                return .required
            default:
                throw OpenAIServerError.invalidJSON
            }
        }

        guard let object = value as? [String: Any],
            object["type"] as? String == "function",
            let function = object["function"] as? [String: Any],
            let name = function["name"] as? String,
            !name.isEmpty
        else {
            throw OpenAIServerError.invalidJSON
        }

        return .function(name)
    }
}
