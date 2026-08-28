import Foundation

public struct QwenXMLToolGrammarConfiguration: Sendable, Equatable {
    public let vocabulary: JSONGrammarVocabulary
    public let toolNames: [String]
    public let toolCallStartTokenID: Int
    public let toolCallEndTokenID: Int
    public let postToolAllowedTokenIDs: [Int]
    public let constrainedTokenBudget: Int

    public init(
        vocabulary: JSONGrammarVocabulary,
        toolNames: [String],
        toolCallStartTokenID: Int,
        toolCallEndTokenID: Int,
        postToolAllowedTokenIDs: [Int],
        constrainedTokenBudget: Int = 4096
    ) {
        self.vocabulary = vocabulary
        self.toolNames = Array(Set(toolNames.filter { !$0.isEmpty })).sorted()
        self.toolCallStartTokenID = toolCallStartTokenID
        self.toolCallEndTokenID = toolCallEndTokenID
        self.constrainedTokenBudget = max(1, constrainedTokenBudget)
        self.postToolAllowedTokenIDs = Array(Set(postToolAllowedTokenIDs)).sorted()
    }

    public func makeMatcher() -> QwenXMLToolBodyMatcher {
        QwenXMLToolBodyMatcher(configuration: self)
    }

    public static func isXMLWhitespace(_ character: Character) -> Bool {
        character == " " || character == "\n" || character == "\t" || character == "\r"
    }
}

public final class QwenXMLToolBodyMatcher: @unchecked Sendable {
    private let configuration: QwenXMLToolGrammarConfiguration
    private let functionOpenTags: [String]
    private var state: QwenXMLToolBodyState = .beforeFunction(prefix: "")
    private var generatedText = ""

    public init(configuration: QwenXMLToolGrammarConfiguration) {
        self.configuration = configuration
        self.functionOpenTags = configuration.toolNames.map { "<function=\($0)>" }
    }

    public func allowedTokenIDs() -> [Int] {
        makeMaskSnapshot().allowedTokenIDs()
    }

    public func accepts(tokenID: Int) -> Bool {
        guard let token = token(for: tokenID) else {
            return false
        }
        return accepts(token: token)
    }

    public func advance(tokenID: Int) {
        guard let token = token(for: tokenID), accepts(token: token) else {
            return
        }
        state = advancedState(appending: token.text, from: state) ?? .invalid
        generatedText += token.text
    }

    public var isComplete: Bool {
        state == .complete
    }

    public var text: String {
        generatedText
    }

    func makeMaskSnapshot() -> QwenXMLToolBodyMaskSnapshot {
        QwenXMLToolBodyMaskSnapshot(
            vocabulary: configuration.vocabulary,
            toolCallEndTokenID: configuration.toolCallEndTokenID,
            functionOpenTags: functionOpenTags,
            state: state
        )
    }

    private func accepts(token: JSONGrammarToken) -> Bool {
        !token.isEOS && advancedState(appending: token.text, from: state) != nil
    }

    private func token(for tokenID: Int) -> JSONGrammarToken? {
        if let token = configuration.vocabulary.tokensByID[tokenID] {
            return token
        }
        if tokenID == configuration.toolCallEndTokenID {
            return JSONGrammarToken(id: tokenID, text: qwenXMLToolCallCloseTag)
        }
        return nil
    }

    private func advancedState(appending text: String, from startState: QwenXMLToolBodyState) -> QwenXMLToolBodyState? {
        qwenXMLToolAdvancedState(
            appending: text,
            from: startState,
            functionOpenTags: functionOpenTags
        )
    }
}

struct QwenXMLToolBodyMaskSnapshot: GrammarMaskSnapshot {
    let vocabulary: JSONGrammarVocabulary
    let toolCallEndTokenID: Int
    let functionOpenTags: [String]
    let state: QwenXMLToolBodyState

    func allowedTokenIDs(isCancelled: @Sendable () -> Bool) -> [Int]? {
        var allowed = Set<Int>()
        var checked = 0
        for token in vocabulary.tokens where !token.isEOS {
            checked += 1
            if checked % 256 == 0, isCancelled() { return nil }
            if accepts(token: token) {
                allowed.insert(token.id)
            }
        }
        let closeToken = JSONGrammarToken(id: toolCallEndTokenID, text: qwenXMLToolCallCloseTag)
        if accepts(token: closeToken) {
            allowed.insert(toolCallEndTokenID)
        }
        return allowed.sorted()
    }

    private func accepts(token: JSONGrammarToken) -> Bool {
        qwenXMLToolAdvancedState(
            appending: token.text,
            from: state,
            functionOpenTags: functionOpenTags
        ) != nil
    }
}

public enum QwenXMLToolGrammarTransition: Sendable, Equatable {
    case none
    case armed
    case advanced
    case completed
    case overrun
    case abandoned
}

public struct QwenXMLToolGrammarRuntime: Sendable {
    private let configuration: QwenXMLToolGrammarConfiguration
    private var mode: QwenXMLToolGrammarMode = .normal
    private var matcher: QwenXMLToolBodyMatcher?
    private var mask: AsyncGrammarMask<QwenXMLToolBodyMaskSnapshot>?
    private var constrainedTokenCount = 0

    public init(configuration: QwenXMLToolGrammarConfiguration) {
        self.configuration = configuration
    }

    var activeMatcher: QwenXMLToolBodyMatcher? {
        mode == .tool ? matcher : nil
    }

    var activePostToolAllowedTokenIDs: [Int]? {
        mode == .postTool ? configuration.postToolAllowedTokenIDs : nil
    }

    var activeAllowedTokenIDs: [Int]? {
        switch mode {
        case .normal:
            return nil
        case .tool:
            return mask?.readyTokenIDs
        case .postTool:
            return configuration.postToolAllowedTokenIDs
        }
    }

    var hasActiveConstraint: Bool {
        mode != .normal
    }

    var requiresScalarSampling: Bool {
        hasActiveConstraint
    }

    public mutating func observeGeneratedToken(_ tokenID: Int) {
        recordGeneratedToken(tokenID)
    }

    @discardableResult
    public mutating func recordGeneratedToken(_ tokenID: Int) -> QwenXMLToolGrammarTransition {
        switch mode {
        case .normal:
            if tokenID == configuration.toolCallStartTokenID {
                armToolBody()
                return .armed
            }
            return .none
        case .postTool:
            if tokenID == configuration.toolCallStartTokenID {
                armToolBody()
                return .armed
            } else if configuration.postToolAllowedTokenIDs.contains(tokenID) {
                return .none
            } else {
                mode = .normal
                return .abandoned
            }
        case .tool:
            guard let matcher, matcher.accepts(tokenID: tokenID) else {
                dropConstrainedRegion()
                return .abandoned
            }
            matcher.advance(tokenID: tokenID)
            constrainedTokenCount += 1
            if matcher.isComplete {
                mask?.invalidate()
                self.matcher = nil
                mask = nil
                mode = .postTool
                return .completed
            } else if constrainedTokenCount >= configuration.constrainedTokenBudget {
                dropConstrainedRegion()
                return .overrun
            } else {
                mask?.prepareAdvancedState(from: matcher.makeMaskSnapshot())
                return .advanced
            }
        }
    }

    private mutating func armToolBody() {
        let matcher = configuration.makeMatcher()
        let mask = AsyncGrammarMask<QwenXMLToolBodyMaskSnapshot>()
        mask.prepareCurrentState(from: matcher.makeMaskSnapshot())
        self.matcher = matcher
        self.mask = mask
        constrainedTokenCount = 0
        mode = .tool
    }

    private mutating func dropConstrainedRegion() {
        mask?.invalidate()
        matcher = nil
        mask = nil
        mode = .normal
        constrainedTokenCount = 0
    }
}

public typealias QwenXMLToolGrammarState = QwenXMLToolGrammarRuntime

private enum QwenXMLToolGrammarMode: Sendable {
    case normal
    case tool
    case postTool
}

enum QwenXMLToolBodyState: Equatable {
    case beforeFunction(prefix: String)
    case functionBody
    case bodyTag(prefix: String)
    case parameterName(String)
    case parameterText(suffix: String)
    case afterFunction(prefix: String)
    case complete
    case invalid
}

private let qwenXMLToolParameterOpenPrefix = "<parameter="
private let qwenXMLToolParameterCloseTag = "</parameter>"
private let qwenXMLToolFunctionCloseTag = "</function>"
private let qwenXMLToolCallCloseTag = "</tool_call>"

private func qwenXMLToolAdvancedState(
    appending text: String,
    from startState: QwenXMLToolBodyState,
    functionOpenTags: [String]
) -> QwenXMLToolBodyState? {
    guard !text.isEmpty, !functionOpenTags.isEmpty else {
        return nil
    }

    var state = startState
    for character in text {
        guard let next = qwenXMLToolAdvance(
            state,
            character: character,
            functionOpenTags: functionOpenTags
        ) else {
            return nil
        }
        state = next
    }
    return state == .invalid ? nil : state
}

private func qwenXMLToolAdvance(
    _ state: QwenXMLToolBodyState,
    character: Character,
    functionOpenTags: [String]
) -> QwenXMLToolBodyState? {
    switch state {
    case .beforeFunction(let prefix):
        if prefix.isEmpty, QwenXMLToolGrammarConfiguration.isXMLWhitespace(character) {
            return .beforeFunction(prefix: "")
        }
        let next = prefix + String(character)
        guard functionOpenTags.contains(where: { $0.hasPrefix(next) }) else {
            return nil
        }
        return functionOpenTags.contains(next) ? .functionBody : .beforeFunction(prefix: next)

    case .functionBody:
        if QwenXMLToolGrammarConfiguration.isXMLWhitespace(character) {
            return .functionBody
        }
        guard character == "<" else {
            return nil
        }
        return .bodyTag(prefix: "<")

    case .bodyTag(let prefix):
        let next = prefix + String(character)
        let candidates = [qwenXMLToolParameterOpenPrefix, qwenXMLToolFunctionCloseTag]
        guard candidates.contains(where: { $0.hasPrefix(next) }) else {
            return nil
        }
        if next == qwenXMLToolParameterOpenPrefix {
            return .parameterName("")
        }
        if next == qwenXMLToolFunctionCloseTag {
            return .afterFunction(prefix: "")
        }
        return .bodyTag(prefix: next)

    case .parameterName(let name):
        if character == ">" {
            return name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? nil
                : .parameterText(suffix: "")
        }
        guard character != ">" else {
            return nil
        }
        return .parameterName(name + String(character))

    case .parameterText(let suffix):
        let nextSuffix = suffix + String(character)
        if nextSuffix.hasSuffix(qwenXMLToolParameterCloseTag) {
            return .functionBody
        }
        if nextSuffix.hasSuffix(qwenXMLToolFunctionCloseTag)
            || nextSuffix.hasSuffix(qwenXMLToolCallCloseTag)
        {
            return nil
        }
        return .parameterText(suffix: qwenXMLToolTrimmedSuffix(nextSuffix))

    case .afterFunction(let prefix):
        if prefix.isEmpty, QwenXMLToolGrammarConfiguration.isXMLWhitespace(character) {
            return .afterFunction(prefix: "")
        }
        let next = prefix + String(character)
        guard qwenXMLToolCallCloseTag.hasPrefix(next) else {
            return nil
        }
        return next == qwenXMLToolCallCloseTag ? .complete : .afterFunction(prefix: next)

    case .complete, .invalid:
        return nil
    }
}

private func qwenXMLToolTrimmedSuffix(_ value: String) -> String {
    let keep = max(
        qwenXMLToolParameterCloseTag.count,
        qwenXMLToolFunctionCloseTag.count,
        qwenXMLToolCallCloseTag.count
    ) - 1
    guard value.count > keep else {
        return value
    }
    return String(value.suffix(keep))
}
