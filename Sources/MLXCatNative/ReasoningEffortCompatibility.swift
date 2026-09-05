import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXVLM
import Tokenizers

enum ReasoningEffortTemplateValue: Equatable {
    case value(OpenAIJSONValue)
    case omitted
}

enum ReasoningEffortCompatibility {
    private static let aliases: [String: String] = [
        "off": "low",
        "none": "low",
        "minimal": "low",
        "moderate": "medium",
        "medium": "high",
        "high": "xhigh",
        "xhigh": "max",
        "max": "xhigh",
        "maximum": "max",
        "ultra": "max"
    ]

    private static let harmonyAliases: [String: String] = [
        "off": "low",
        "none": "low",
        "minimal": "low",
        "low": "low",
        "moderate": "medium",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "high",
        "maximum": "high",
        "ultra": "high"
    ]

    private static let namedLevels = Set(aliases.keys)

    static func candidates(
        for value: OpenAIJSONValue?,
        isHarmony: Bool
    ) -> [ReasoningEffortTemplateValue] {
        guard let value else { return [.omitted] }

        if isHarmony {
            guard case .string(let rawValue) = value,
                  let mapped = harmonyAliases[rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()]
            else {
                return [.omitted]
            }
            return [.value(.string(mapped))]
        }

        let normalized = normalized(value)
        var candidates: [ReasoningEffortTemplateValue] = [.value(normalized)]
        if let fallback = fallback(for: normalized), fallback != normalized {
            candidates.append(.value(fallback))
        }
        candidates.append(.omitted)
        return candidates
    }

    private static func normalized(_ value: OpenAIJSONValue) -> OpenAIJSONValue {
        guard case .string(let rawValue) = value else { return value }
        let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if namedLevels.contains(normalized) || finiteNumber(normalized) != nil {
            return .string(normalized)
        }
        return value
    }

    private static func fallback(for value: OpenAIJSONValue) -> OpenAIJSONValue? {
        guard case .string(let string) = value else { return nil }
        if let numeric = finiteNumber(string) {
            return .number(numeric)
        }
        return aliases[string].map(OpenAIJSONValue.string)
    }

    private static func finiteNumber(_ value: String) -> Double? {
        guard let number = Double(value), number.isFinite else { return nil }
        return number
    }
}
