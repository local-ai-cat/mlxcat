import Foundation

public enum RegexGrammarError: Error, CustomStringConvertible, Equatable, Sendable {
    case unsupported(String)

    public var description: String {
        switch self {
        case .unsupported(let message):
            return message
        }
    }
}

public struct RegexGrammarConfiguration: Sendable, Equatable {
    public let vocabulary: JSONGrammarVocabulary
    public let pattern: String
    fileprivate let nfa: RegexNFA

    public init(vocabulary: JSONGrammarVocabulary, pattern: String) throws {
        var parser = RegexParser(pattern: pattern)
        let ast = try parser.parse()
        self.vocabulary = vocabulary
        self.pattern = pattern
        self.nfa = RegexNFABuilder.compile(ast)
    }

    public init(tokens: [JSONGrammarToken], pattern: String) throws {
        try self.init(vocabulary: JSONGrammarVocabulary(tokens: tokens), pattern: pattern)
    }

    public func makeMatcher() -> RegexGrammarMatcher {
        RegexGrammarMatcher(configuration: self)
    }

    public static func == (lhs: RegexGrammarConfiguration, rhs: RegexGrammarConfiguration) -> Bool {
        lhs.vocabulary == rhs.vocabulary && lhs.pattern == rhs.pattern
    }
}

public final class RegexGrammarMatcher {
    private let vocabulary: JSONGrammarVocabulary
    private let nfa: RegexNFA
    private var currentStates: Set<Int>
    private var generatedText = ""

    public init(configuration: RegexGrammarConfiguration) {
        self.vocabulary = configuration.vocabulary
        self.nfa = configuration.nfa
        self.currentStates = configuration.nfa.epsilonClosure([configuration.nfa.start])
    }

    public func allowedTokenIDs() -> [Int] {
        var allowed: [Int] = []
        for (firstCharacter, bucket) in vocabulary.tokensByFirstCharacter {
            guard nfa.isViable(nfa.advance(currentStates, over: String(firstCharacter))) else {
                continue
            }
            for token in bucket where accepts(token: token) {
                allowed.append(token.id)
            }
        }
        if isComplete {
            allowed.append(contentsOf: vocabulary.eosTokenIDs)
        }
        return allowed
    }

    public func accepts(tokenID: Int) -> Bool {
        guard let token = vocabulary.tokensByID[tokenID] else {
            return false
        }
        return accepts(token: token)
    }

    public func advance(tokenID: Int) {
        guard let token = vocabulary.tokensByID[tokenID], !token.isEOS else {
            return
        }
        let nextStates = nfa.advance(currentStates, over: token.text)
        guard nfa.isViable(nextStates) else {
            return
        }
        currentStates = nextStates
        generatedText += token.text
    }

    public var isComplete: Bool {
        nfa.isAccepting(currentStates)
    }

    public var text: String {
        generatedText
    }

    func makeMaskSnapshot() -> RegexGrammarMaskSnapshot {
        RegexGrammarMaskSnapshot(
            vocabulary: vocabulary,
            nfa: nfa,
            currentStates: currentStates
        )
    }

    private func accepts(token: JSONGrammarToken) -> Bool {
        if token.isEOS {
            return isComplete
        }
        guard !token.text.isEmpty else {
            return false
        }
        return nfa.isViable(nfa.advance(currentStates, over: token.text))
    }
}

struct RegexGrammarMaskSnapshot: GrammarMaskSnapshot {
    let vocabulary: JSONGrammarVocabulary
    fileprivate let nfa: RegexNFA
    let currentStates: Set<Int>

    func allowedTokenIDs(isCancelled: @Sendable () -> Bool) -> [Int]? {
        var allowed: [Int] = []
        for (firstCharacter, bucket) in vocabulary.tokensByFirstCharacter {
            if isCancelled() { return nil }
            guard nfa.isViable(nfa.advance(currentStates, over: String(firstCharacter))) else {
                continue
            }
            var checked = 0
            for token in bucket {
                checked += 1
                if checked % 256 == 0, isCancelled() { return nil }
                if accepts(token: token) {
                    allowed.append(token.id)
                }
            }
        }
        if isComplete {
            allowed.append(contentsOf: vocabulary.eosTokenIDs)
        }
        return allowed
    }

    private var isComplete: Bool {
        nfa.isAccepting(currentStates)
    }

    private func accepts(token: JSONGrammarToken) -> Bool {
        if token.isEOS {
            return isComplete
        }
        guard !token.text.isEmpty else {
            return false
        }
        return nfa.isViable(nfa.advance(currentStates, over: token.text))
    }
}

private indirect enum RegexAST: Sendable, Equatable {
    case empty
    case predicate(RegexCharacterPredicate)
    case concat([RegexAST])
    case alternate([RegexAST])
    case repeatNode(RegexAST, min: Int, max: Int?)
}

private struct RegexNFA: Sendable, Equatable {
    struct State: Sendable, Equatable {
        var epsilon: [Int] = []
        var transitions: [Transition] = []
        var accepting = false
    }

    struct Transition: Sendable, Equatable {
        let predicate: RegexCharacterPredicate
        let target: Int
    }

    let states: [State]
    let start: Int
    let reachableAcceptingStates: Set<Int>

    func epsilonClosure(_ states: some Sequence<Int>) -> Set<Int> {
        var closure = Set(states)
        var stack = Array(closure)
        while let state = stack.popLast() {
            for target in self.states[state].epsilon where !closure.contains(target) {
                closure.insert(target)
                stack.append(target)
            }
        }
        return closure
    }

    func advance(_ states: Set<Int>, over text: String) -> Set<Int> {
        var current = epsilonClosure(states)
        for character in text {
            var next: Set<Int> = []
            for state in current {
                for transition in self.states[state].transitions
                    where transition.predicate.matches(character)
                {
                    next.insert(transition.target)
                }
            }
            guard !next.isEmpty else {
                return []
            }
            current = epsilonClosure(next)
        }
        return current
    }

    func isAccepting(_ states: Set<Int>) -> Bool {
        states.contains { self.states[$0].accepting }
    }

    func isViable(_ states: Set<Int>) -> Bool {
        !states.isDisjoint(with: reachableAcceptingStates)
    }
}

private struct RegexCharacterPredicate: Sendable, Equatable {
    let ranges: [ClosedRange<Unicode.Scalar>]
    let inverted: Bool
    let any: Bool

    static let anyCharacter = RegexCharacterPredicate(ranges: [], inverted: false, any: true)

    static func literal(_ character: Character) throws -> RegexCharacterPredicate {
        guard let scalar = singleScalar(character) else {
            throw RegexGrammarError.unsupported("regex literals must be single Unicode scalars")
        }
        return RegexCharacterPredicate(ranges: [scalar...scalar], inverted: false, any: false)
    }

    static func characterClass(
        ranges: [ClosedRange<Unicode.Scalar>],
        inverted: Bool = false
    ) -> RegexCharacterPredicate {
        RegexCharacterPredicate(ranges: ranges, inverted: inverted, any: false)
    }

    func matches(_ character: Character) -> Bool {
        guard let scalar = Self.singleScalar(character) else {
            return false
        }
        if any {
            return scalar != "\n"
        }
        let contained = ranges.contains { $0.contains(scalar) }
        return inverted ? !contained : contained
    }

    private static func singleScalar(_ character: Character) -> Unicode.Scalar? {
        let scalars = String(character).unicodeScalars
        guard scalars.count == 1 else { return nil }
        return scalars.first
    }
}

private struct RegexParser {
    private let pattern: [Character]
    private var index = 0

    init(pattern: String) {
        self.pattern = Array(pattern)
    }

    mutating func parse() throws -> RegexAST {
        let ast = try parseAlternation()
        guard isAtEnd else {
            throw RegexGrammarError.unsupported("unsupported regex syntax near '\(current)'")
        }
        return ast
    }

    private mutating func parseAlternation() throws -> RegexAST {
        var branches = [try parseConcatenation()]
        while consume("|") {
            branches.append(try parseConcatenation())
        }
        return branches.count == 1 ? branches[0] : .alternate(branches)
    }

    private mutating func parseConcatenation() throws -> RegexAST {
        var parts: [RegexAST] = []
        while !isAtEnd, current != ")", current != "|" {
            parts.append(try parseRepetition())
        }
        if parts.isEmpty {
            return .empty
        }
        return parts.count == 1 ? parts[0] : .concat(parts)
    }

    private mutating func parseRepetition() throws -> RegexAST {
        let atom = try parseAtom()
        if consume("*") {
            return try ensureNoRepeatedQuantifier(.repeatNode(atom, min: 0, max: nil))
        }
        if consume("+") {
            return try ensureNoRepeatedQuantifier(.repeatNode(atom, min: 1, max: nil))
        }
        if consume("?") {
            return try ensureNoRepeatedQuantifier(.repeatNode(atom, min: 0, max: 1))
        }
        if consume("{") {
            let (min, max) = try parseBounds()
            return try ensureNoRepeatedQuantifier(.repeatNode(atom, min: min, max: max))
        }
        return atom
    }

    private mutating func ensureNoRepeatedQuantifier(_ ast: RegexAST) throws -> RegexAST {
        if ["*", "+", "?", "{"].contains(current) {
            throw RegexGrammarError.unsupported("repeated regex quantifiers are unsupported")
        }
        return ast
    }

    private mutating func parseAtom() throws -> RegexAST {
        if consume("(") {
            let ast = try parseAlternation()
            guard consume(")") else {
                throw RegexGrammarError.unsupported("unclosed regex group")
            }
            return ast
        }
        if consume("[") {
            return .predicate(try parseCharacterClass())
        }
        if consume(".") {
            return .predicate(.anyCharacter)
        }
        if current == "^" || current == "$" {
            throw RegexGrammarError.unsupported(Self.anchorUnsupportedMessage)
        }
        if consume("\\") {
            return .predicate(try parseEscape(inClass: false))
        }
        guard !isAtEnd, !["*", "+", "?", "{", "}", ")"].contains(current) else {
            throw RegexGrammarError.unsupported("unsupported regex atom near '\(current)'")
        }
        return .predicate(try .literal(advance()))
    }

    private mutating func parseCharacterClass() throws -> RegexCharacterPredicate {
        if current == "^" {
            throw RegexGrammarError.unsupported(Self.anchorUnsupportedMessage)
        }
        let inverted = false
        var ranges: [ClosedRange<Unicode.Scalar>] = []
        var previous: Unicode.Scalar?

        while !isAtEnd, current != "]" {
            let start = try parseClassScalar()
            if consume("-"), !isAtEnd, current != "]" {
                let end = try parseClassScalar()
                guard start.value <= end.value else {
                    throw RegexGrammarError.unsupported("regex character class range is reversed")
                }
                ranges.append(start...end)
                previous = nil
            } else {
                ranges.append(start...start)
                previous = start
            }
        }

        guard consume("]") else {
            throw RegexGrammarError.unsupported("unclosed regex character class")
        }
        _ = previous
        return .characterClass(ranges: ranges, inverted: inverted)
    }

    private mutating func parseClassScalar() throws -> Unicode.Scalar {
        if current == "^" || current == "$" {
            throw RegexGrammarError.unsupported(Self.anchorUnsupportedMessage)
        }
        if consume("\\") {
            let predicate = try parseEscape(inClass: true)
            guard predicate.ranges.count == 1,
                predicate.ranges[0].lowerBound == predicate.ranges[0].upperBound,
                !predicate.inverted,
                !predicate.any
            else {
                throw RegexGrammarError.unsupported("regex shorthand classes are unsupported inside character ranges")
            }
            return predicate.ranges[0].lowerBound
        }
        guard let scalar = String(advance()).unicodeScalars.first else {
            throw RegexGrammarError.unsupported("regex class scalar could not be parsed")
        }
        return scalar
    }

    private mutating func parseEscape(inClass: Bool) throws -> RegexCharacterPredicate {
        guard !isAtEnd else {
            throw RegexGrammarError.unsupported("dangling regex escape")
        }
        let character = advance()
        switch character {
        case "d":
            return .characterClass(ranges: [scalar("0")...scalar("9")])
        case "w":
            return .characterClass(ranges: [
                scalar("0")...scalar("9"),
                scalar("A")...scalar("Z"),
                scalar("_")...scalar("_"),
                scalar("a")...scalar("z"),
            ])
        case "s":
            return .characterClass(ranges: [
                scalar("\t")...scalar("\t"),
                scalar("\n")...scalar("\n"),
                scalar("\r")...scalar("\r"),
                scalar(" ")...scalar(" "),
            ])
        case "n":
            return try .literal("\n")
        case "r":
            return try .literal("\r")
        case "t":
            return try .literal("\t")
        default:
            if Self.escapedLiteralMetacharacters.contains(character) {
                return try .literal(character)
            }
            throw RegexGrammarError.unsupported("unsupported regex escape '\\\(character)'")
        }
    }

    private mutating func parseBounds() throws -> (Int, Int?) {
        let min = try parseInteger()
        if consume("}") {
            return (min, min)
        }
        guard consume(",") else {
            throw RegexGrammarError.unsupported("invalid regex bounded repetition")
        }
        if consume("}") {
            return (min, nil)
        }
        let max = try parseInteger()
        guard consume("}") else {
            throw RegexGrammarError.unsupported("invalid regex bounded repetition")
        }
        guard max >= min else {
            throw RegexGrammarError.unsupported("regex bounded repetition max is smaller than min")
        }
        return (min, max)
    }

    private mutating func parseInteger() throws -> Int {
        var digits = ""
        while !isAtEnd, current.isNumber {
            digits.append(advance())
        }
        guard let value = Int(digits) else {
            throw RegexGrammarError.unsupported("regex repetition bound must be an integer")
        }
        guard value <= 256 else {
            throw RegexGrammarError.unsupported("regex repetition bounds above 256 are unsupported")
        }
        return value
    }

    private func scalar(_ character: Character) -> Unicode.Scalar {
        String(character).unicodeScalars.first!
    }

    private static let anchorUnsupportedMessage =
        "regex anchors (^ and $) are unsupported because constrained decode is implicitly anchored"

    private static let escapedLiteralMetacharacters = Set<Character>(
        [".", "\\", "(", ")", "[", "]", "{", "}", "*", "+", "?", "|", "^", "$", "-"]
    )

    private var isAtEnd: Bool {
        index >= pattern.count
    }

    private var current: Character {
        isAtEnd ? "\0" : pattern[index]
    }

    private mutating func consume(_ character: Character) -> Bool {
        guard current == character else { return false }
        index += 1
        return true
    }

    private mutating func advance() -> Character {
        let character = current
        index += 1
        return character
    }
}

private enum RegexNFABuilder {
    struct Fragment {
        let start: Int
        let accepts: Set<Int>
    }

    static func compile(_ ast: RegexAST) -> RegexNFA {
        var builder = Builder()
        let fragment = builder.build(ast)
        let accept = builder.addState()
        for state in fragment.accepts {
            builder.addEpsilon(from: state, to: accept)
        }
        builder.states[accept].accepting = true
        return builder.makeNFA(start: fragment.start)
    }

    private struct Builder {
        var states: [RegexNFA.State] = []

        mutating func build(_ ast: RegexAST) -> Fragment {
            switch ast {
            case .empty:
                let state = addState()
                return Fragment(start: state, accepts: [state])
            case .predicate(let predicate):
                let start = addState()
                let accept = addState()
                states[start].transitions.append(.init(predicate: predicate, target: accept))
                return Fragment(start: start, accepts: [accept])
            case .concat(let parts):
                guard let first = parts.first else {
                    return build(.empty)
                }
                var result = build(first)
                for part in parts.dropFirst() {
                    let next = build(part)
                    for state in result.accepts {
                        addEpsilon(from: state, to: next.start)
                    }
                    result = Fragment(start: result.start, accepts: next.accepts)
                }
                return result
            case .alternate(let branches):
                let start = addState()
                let accept = addState()
                for branch in branches {
                    let fragment = build(branch)
                    addEpsilon(from: start, to: fragment.start)
                    for state in fragment.accepts {
                        addEpsilon(from: state, to: accept)
                    }
                }
                return Fragment(start: start, accepts: [accept])
            case .repeatNode(let ast, let min, let max):
                return buildRepeat(ast, min: min, max: max)
            }
        }

        private mutating func buildRepeat(_ ast: RegexAST, min: Int, max: Int?) -> Fragment {
            if max == 0 {
                return build(.empty)
            }

            if min == 0, max == nil {
                return buildStar(ast)
            }

            var result: Fragment?
            for _ in 0..<min {
                append(build(ast), to: &result)
            }
            if let max {
                for _ in 0..<(max - min) {
                    append(buildOptional(ast), to: &result)
                }
            } else {
                append(buildStar(ast), to: &result)
            }
            return result ?? build(.empty)
        }

        private mutating func buildOptional(_ ast: RegexAST) -> Fragment {
            let fragment = build(ast)
            let start = addState()
            let accept = addState()
            addEpsilon(from: start, to: fragment.start)
            addEpsilon(from: start, to: accept)
            for state in fragment.accepts {
                addEpsilon(from: state, to: accept)
            }
            return Fragment(start: start, accepts: [accept])
        }

        private mutating func buildStar(_ ast: RegexAST) -> Fragment {
            let fragment = build(ast)
            let start = addState()
            let accept = addState()
            addEpsilon(from: start, to: fragment.start)
            addEpsilon(from: start, to: accept)
            for state in fragment.accepts {
                addEpsilon(from: state, to: fragment.start)
                addEpsilon(from: state, to: accept)
            }
            return Fragment(start: start, accepts: [accept])
        }

        private mutating func append(_ next: Fragment, to result: inout Fragment?) {
            guard let current = result else {
                result = next
                return
            }
            for state in current.accepts {
                addEpsilon(from: state, to: next.start)
            }
            result = Fragment(start: current.start, accepts: next.accepts)
        }

        mutating func addState() -> Int {
            states.append(RegexNFA.State())
            return states.count - 1
        }

        mutating func addEpsilon(from: Int, to: Int) {
            states[from].epsilon.append(to)
        }

        func makeNFA(start: Int) -> RegexNFA {
            let reachable = reachableAcceptingStates()
            return RegexNFA(states: states, start: start, reachableAcceptingStates: reachable)
        }

        private func reachableAcceptingStates() -> Set<Int> {
            var reverse: [Int: [Int]] = [:]
            for (state, value) in states.enumerated() {
                for target in value.epsilon {
                    reverse[target, default: []].append(state)
                }
                for transition in value.transitions {
                    reverse[transition.target, default: []].append(state)
                }
            }

            var reachable = Set(states.indices.filter { states[$0].accepting })
            var stack = Array(reachable)
            while let state = stack.popLast() {
                for previous in reverse[state, default: []] where !reachable.contains(previous) {
                    reachable.insert(previous)
                    stack.append(previous)
                }
            }
            return reachable
        }
    }
}
