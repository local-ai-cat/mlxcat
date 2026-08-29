import Foundation

/// The shape of a token mask.
///
/// A grammar state that admits nearly the whole vocabulary must publish the
/// small COMPLEMENT rather than enumerate ~150k admissions. Enumerating cost
/// O(vocab x token length) on every constrained token — 107 ms measured on
/// Qwen3-Coder-30B — which is not merely a slow mask: it exceeds the ~11 ms
/// inter-token budget at 88 tok/s, so an off-path precompute can never finish
/// before the state it was computed for is superseded.
public enum GrammarTokenMask: Sendable, Equatable {
    /// Only these ids may be sampled. Right for restrictive states.
    case allow([Int])
    /// Every id EXCEPT these may be sampled. Right for permissive states.
    case deny([Int])
}

/// The candidate set behind the deny-shaped mask, built once per vocabulary.
///
/// Both permissive Qwen-XML states — `.parameterName` and `.parameterText` —
/// can only reject a character, or leave the state, on ">":
///
///   * `.parameterText` rejects when the rolling suffix completes
///     `</function>` or `</tool_call>`, and leaves on `</parameter>`. All three
///     end in ">".
///   * `.parameterName` accepts every character except ">", which either opens
///     the parameter body or rejects an empty name.
///
/// So a token whose text contains no ">" is accepted unconditionally from
/// either state, and cannot change the state. That makes "tokens containing
/// >" a COMPLETE candidate set — not a heuristic — and every candidate is then
/// simulated by the same DFA the allow path uses, so the two masks cannot
/// disagree. `QwenXMLToolBodyMaskSnapshot.allowedTokenIDs` is retained as the
/// exhaustive reference implementation the equivalence tests check against.
public final class QwenXMLToolBodyDenyIndex: Sendable {
    /// Tokens that must be simulated: the only ones a permissive state can reject.
    let candidates: [JSONGrammarToken]
    /// Denied in every state, so never candidates: EOS (carries no text and is
    /// inadmissible mid-tool-call) and any empty-text token (the DFA rejects an
    /// empty advance outright). The allow path gets both for free by iterating
    /// non-EOS tokens and rejecting empty text; a complement mask has to name
    /// them explicitly or it would silently admit them.
    let alwaysDeniedTokenIDs: [Int]

    public init(vocabulary: JSONGrammarVocabulary) {
        var candidates: [JSONGrammarToken] = []
        var alwaysDenied: [Int] = []
        for token in vocabulary.tokens {
            if token.isEOS || token.text.isEmpty {
                alwaysDenied.append(token.id)
            } else if token.text.contains(">") {
                candidates.append(token)
            }
        }
        self.candidates = candidates
        self.alwaysDeniedTokenIDs = alwaysDenied
    }
}

public struct QwenXMLToolGrammarConfiguration: Sendable, Equatable {
    public let vocabulary: JSONGrammarVocabulary
    public let toolNames: [String]
    public let toolCallStartTokenID: Int
    public let toolCallEndTokenID: Int
    public let postToolAllowedTokenIDs: [Int]
    public let constrainedTokenBudget: Int
    let denyIndex: QwenXMLToolBodyDenyIndex
    let postToolAllowedTokenIDSet: Set<Int>

    /// `denyIndex` is derived purely from `vocabulary`; pass a cached one so the
    /// O(vocab) scan happens once per model rather than once per request — this
    /// initialiser runs on every chat completion that carries tools.
    public init(
        vocabulary: JSONGrammarVocabulary,
        toolNames: [String],
        toolCallStartTokenID: Int,
        toolCallEndTokenID: Int,
        postToolAllowedTokenIDs: [Int],
        constrainedTokenBudget: Int = 4096,
        denyIndex: QwenXMLToolBodyDenyIndex? = nil
    ) {
        self.vocabulary = vocabulary
        self.toolNames = Array(Set(toolNames.filter { !$0.isEmpty })).sorted()
        self.toolCallStartTokenID = toolCallStartTokenID
        self.toolCallEndTokenID = toolCallEndTokenID
        self.constrainedTokenBudget = max(1, constrainedTokenBudget)
        self.postToolAllowedTokenIDs = Array(Set(postToolAllowedTokenIDs)).sorted()
        self.postToolAllowedTokenIDSet = Set(postToolAllowedTokenIDs)
        self.denyIndex = denyIndex ?? QwenXMLToolBodyDenyIndex(vocabulary: vocabulary)
    }

    /// `denyIndex` is a pure function of `vocabulary`, so it is not part of identity.
    public static func == (
        lhs: QwenXMLToolGrammarConfiguration,
        rhs: QwenXMLToolGrammarConfiguration
    ) -> Bool {
        lhs.vocabulary == rhs.vocabulary
            && lhs.toolNames == rhs.toolNames
            && lhs.toolCallStartTokenID == rhs.toolCallStartTokenID
            && lhs.toolCallEndTokenID == rhs.toolCallEndTokenID
            && lhs.postToolAllowedTokenIDs == rhs.postToolAllowedTokenIDs
            && lhs.constrainedTokenBudget == rhs.constrainedTokenBudget
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

    /// The production mask. Deny-shaped in the permissive states, allow-shaped
    /// (bucket-pruned) elsewhere.
    public func tokenMask() -> GrammarTokenMask {
        makeMaskSnapshot().tokenMask(isCancelled: { false }) ?? .allow([])
    }

    /// Exhaustive reference implementation: every non-EOS token tested against
    /// the DFA. Retained because the equivalence tests check `tokenMask()`
    /// against it over the real vocabulary — it is NOT the sampling path.
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
            denyIndex: configuration.denyIndex,
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
    let denyIndex: QwenXMLToolBodyDenyIndex
    let toolCallEndTokenID: Int
    let functionOpenTags: [String]
    let state: QwenXMLToolBodyState

    /// True where the state admits nearly the whole vocabulary, so the mask must
    /// be published as a complement. See `QwenXMLToolBodyDenyIndex` for why ">"
    /// is the complete candidate filter for exactly these two states.
    var isPermissive: Bool {
        switch state {
        case .parameterName, .parameterText:
            return true
        case .beforeFunction, .functionBody, .bodyTag, .afterFunction, .complete, .invalid:
            return false
        }
    }

    func tokenMask(isCancelled: @Sendable () -> Bool) -> GrammarTokenMask? {
        isPermissive ? deniedTokenMask(isCancelled: isCancelled) : allowedTokenMask(isCancelled: isCancelled)
    }

    private func deniedTokenMask(isCancelled: @Sendable () -> Bool) -> GrammarTokenMask? {
        var denied = denyIndex.alwaysDeniedTokenIDs
        var sawCloseToken = false
        var checked = 0
        for token in denyIndex.candidates {
            checked += 1
            if checked % 256 == 0, isCancelled() { return nil }
            if token.id == toolCallEndTokenID { sawCloseToken = true }
            if !accepts(token: token) {
                denied.append(token.id)
            }
        }
        // The close token may be synthetic (absent from the vocabulary the
        // caller built), in which case the candidate loop never saw it.
        if !sawCloseToken {
            let closeToken = JSONGrammarToken(id: toolCallEndTokenID, text: qwenXMLToolCallCloseTag)
            if !accepts(token: closeToken) {
                denied.append(toolCallEndTokenID)
            }
        }
        return .deny(denied)
    }

    private func allowedTokenMask(isCancelled: @Sendable () -> Bool) -> GrammarTokenMask? {
        var allowed: [Int] = []
        var checked = 0
        for (firstCharacter, bucket) in vocabulary.tokensByFirstCharacter {
            // Prefix validity is monotone: the DFA rejects at the first bad
            // character, so if a one-character extension is invalid then every
            // token starting with that character is invalid too — skip the
            // bucket. Same pruning the JSON/regex/GBNF grammars already use.
            guard accepts(text: String(firstCharacter)) else { continue }
            for token in bucket {
                checked += 1
                if checked % 256 == 0, isCancelled() { return nil }
                if accepts(token: token) {
                    allowed.append(token.id)
                }
            }
        }
        let closeToken = JSONGrammarToken(id: toolCallEndTokenID, text: qwenXMLToolCallCloseTag)
        if accepts(token: closeToken), !allowed.contains(toolCallEndTokenID) {
            allowed.append(toolCallEndTokenID)
        }
        return .allow(allowed)
    }

    /// Exhaustive reference: the pre-optimisation behaviour, kept as the oracle
    /// the equivalence tests compare `tokenMask()` against.
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
        accepts(text: token.text)
    }

    private func accepts(text: String) -> Bool {
        qwenXMLToolAdvancedState(
            appending: text,
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

    /// Only the post-tool allowlist is published as ids. The tool body's mask is
    /// computed synchronously at sample time (see `QwenXMLToolBodyMatcher.tokenMask`):
    /// it is now cheap enough that off-path precomputation buys nothing, and an
    /// `AsyncGrammarMask` here could never be ready anyway — every accepted token
    /// advances the state and invalidates the compute in flight.
    var activeAllowedTokenIDs: [Int]? {
        switch mode {
        case .normal, .tool:
            return nil
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
            } else if configuration.postToolAllowedTokenIDSet.contains(tokenID) {
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
                self.matcher = nil
                mode = .postTool
                return .completed
            } else if constrainedTokenCount >= configuration.constrainedTokenBudget {
                dropConstrainedRegion()
                return .overrun
            } else {
                return .advanced
            }
        }
    }

    private mutating func armToolBody() {
        matcher = configuration.makeMatcher()
        constrainedTokenCount = 0
        mode = .tool
    }

    private mutating func dropConstrainedRegion() {
        matcher = nil
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
