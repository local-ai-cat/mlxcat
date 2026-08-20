import Foundation

/// vLLM-shaped tool-call parsing: a `ToolParser` per model family, registered in
/// a `ToolParserManager`, with a buffered `extractToolCalls` and an incremental
/// `extractToolCallsStreaming`. Ported from vLLM's `vllm/tool_parsers/`
/// (`abstract_tool_parser.py` `ToolParser` + `ToolParserManager`).
///
/// Scope note: vLLM's methods also receive token-id sequences; our engine works
/// on decoded text, and the vLLM parsers we translate key off text (the token
/// ids are only used by DeepSeek's streaming tag-counting, which we replace with
/// the text-based re-parse pattern hermes uses). So the Swift surface is
/// text-only.

/// Result of a buffered parse (vLLM `ExtractedToolCallInformation`).
public struct ExtractedToolCallInformation: Sendable, Equatable {
    public let toolsCalled: Bool
    public let toolCalls: [ParsedToolCall]
    /// Text preceding the tool call(s); `nil` when there is none.
    public let content: String?

    public init(toolsCalled: Bool, toolCalls: [ParsedToolCall], content: String?) {
        self.toolsCalled = toolsCalled
        self.toolCalls = toolCalls
        self.content = content
    }

    static func noTools(_ content: String) -> ExtractedToolCallInformation {
        ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: content)
    }
}

/// One streamed tool-call delta (vLLM `DeltaToolCall`). `id`/`name` are sent
/// once; `argumentsDelta` carries incremental argument text.
public struct StreamedToolCall: Sendable, Equatable {
    public let index: Int
    public var id: String?
    public var name: String?
    public var argumentsDelta: String?

    public init(index: Int, id: String? = nil, name: String? = nil, argumentsDelta: String? = nil) {
        self.index = index
        self.id = id
        self.name = name
        self.argumentsDelta = argumentsDelta
    }
}

/// One streaming step's output (vLLM `DeltaMessage`): plain content and/or
/// tool-call deltas. `nil` from a parser step means "nothing new yet".
public struct ToolCallStreamDelta: Sendable, Equatable {
    public var content: String?
    public var toolCalls: [StreamedToolCall]

    public init(content: String? = nil, toolCalls: [StreamedToolCall] = []) {
        self.content = content
        self.toolCalls = toolCalls
    }

    var isEmpty: Bool { (content == nil || content?.isEmpty == true) && toolCalls.isEmpty }
}

/// Base class for family parsers. Holds the per-request streaming state that
/// vLLM keeps on the parser instance (`current_tool_id`, `streamed_args_for_tool`,
/// …). Subclasses override `extractToolCalls`; the base provides a robust
/// "emit each tool call as it completes" streaming default that subclasses with
/// a canonical incremental algorithm (Hermes) override.
public class ToolParser {
    /// vLLM sets this False for formats where guided-JSON `tool_choice=required`
    /// output doesn't match the native syntax (GLM/Gemma/DeepSeek). We don't run
    /// grammar-forced decoding, so it's informational parity metadata only.
    public class var supportsRequiredAndNamed: Bool { true }

    let idGenerator: () -> String

    // Streaming state (vLLM ToolParser.__init__).
    var currentToolIndex: Int = -1
    var streamedArgsForTool: [String] = []
    var streamedNameForTool: [String] = []

    public required init(idGenerator: @escaping () -> String = defaultToolCallID) {
        self.idGenerator = idGenerator
    }

    /// Buffered extraction from a complete model output.
    public func extractToolCalls(_ modelOutput: String) -> ExtractedToolCallInformation {
        .noTools(modelOutput)
    }

    /// Incremental extraction. Default: re-parse `currentText`, and for each tool
    /// call that has newly completed, emit its name then its full arguments once.
    /// Content preceding the first tool call streams through as `deltaText`.
    public func extractToolCallsStreaming(
        previousText: String,
        currentText: String,
        deltaText: String
    ) -> ToolCallStreamDelta? {
        let extracted = extractToolCalls(currentText)
        guard extracted.toolsCalled else {
            return deltaText.isEmpty ? nil : ToolCallStreamDelta(content: deltaText)
        }

        var deltas: [StreamedToolCall] = []
        for (index, call) in extracted.toolCalls.enumerated() {
            while streamedNameForTool.count <= index { streamedNameForTool.append("") }
            while streamedArgsForTool.count <= index { streamedArgsForTool.append("") }

            if streamedNameForTool[index].isEmpty, !call.name.isEmpty {
                streamedNameForTool[index] = call.name
                deltas.append(StreamedToolCall(index: index, id: call.id, name: call.name))
            }
            if call.arguments != streamedArgsForTool[index],
                call.arguments.hasPrefix(streamedArgsForTool[index]) {
                let diff = String(call.arguments.dropFirst(streamedArgsForTool[index].count))
                streamedArgsForTool[index] = call.arguments
                if !diff.isEmpty {
                    deltas.append(StreamedToolCall(index: index, argumentsDelta: diff))
                }
            }
        }
        return deltas.isEmpty ? nil : ToolCallStreamDelta(toolCalls: deltas)
    }
}

/// Lock-guarded storage so the registry is safe to mutate/read across
/// concurrency domains (Swift 6 rejects nonisolated mutable global state).
private final class ToolParserStore: @unchecked Sendable {
    static let shared = ToolParserStore()
    private let lock = NSLock()
    private var parsers: [String: ToolParser.Type] = [
        "hermes": HermesToolParser.self,
        "llama3_json": Llama3JsonToolParser.self,
        "pythonic": PythonicToolParser.self,
        "gemma4": Gemma4ToolParser.self,
        "deepseek": DeepSeekToolParser.self,
    ]

    func register(_ name: String, _ type: ToolParser.Type) {
        lock.lock(); defer { lock.unlock() }
        parsers[name] = type
    }

    func parserType(named name: String) -> ToolParser.Type? {
        lock.lock(); defer { lock.unlock() }
        return parsers[name]
    }

    var registered: [String] {
        lock.lock(); defer { lock.unlock() }
        return parsers.keys.sorted()
    }
}

/// Central registry mapping a family name to its parser type (vLLM
/// `ToolParserManager`). Adding a family = one `register` call.
public enum ToolParserManager {
    public static func register(_ name: String, _ type: ToolParser.Type) {
        ToolParserStore.shared.register(name, type)
    }

    public static func parserType(named name: String) -> ToolParser.Type? {
        ToolParserStore.shared.parserType(named: name)
    }

    public static func makeParser(
        named name: String,
        idGenerator: @escaping () -> String = defaultToolCallID
    ) -> ToolParser? {
        ToolParserStore.shared.parserType(named: name).map { $0.init(idGenerator: idGenerator) }
    }

    public static var registered: [String] { ToolParserStore.shared.registered }
}
