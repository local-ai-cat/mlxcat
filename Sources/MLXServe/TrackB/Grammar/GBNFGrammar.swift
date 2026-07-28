import Foundation

public enum GBNFGrammarError: Error, CustomStringConvertible, Equatable, Sendable {
    case unsupported(String)

    public var description: String {
        switch self {
        case .unsupported(let message):
            return message
        }
    }
}

public struct GBNFGrammarConfiguration: Sendable, Equatable {
    public let vocabulary: JSONGrammarVocabulary
    public let grammar: String
    fileprivate let rules: [String: GBNFExpr]

    public init(vocabulary: JSONGrammarVocabulary, grammar: String) throws {
        let rules = try GBNFParser(grammar: grammar).parse()
        try GBNFValidator.validate(rules: rules)
        self.vocabulary = vocabulary
        self.grammar = grammar
        self.rules = rules
    }

    public init(tokens: [JSONGrammarToken], grammar: String) throws {
        try self.init(vocabulary: JSONGrammarVocabulary(tokens: tokens), grammar: grammar)
    }

    public func makeMatcher() -> GBNFGrammarMatcher {
        GBNFGrammarMatcher(configuration: self)
    }
}

public final class GBNFGrammarMatcher {
    private let vocabulary: JSONGrammarVocabulary
    private let machine: GBNFMachine
    private var generatedText = ""
    private var memoizedCompletionState: GBNFValidationResult?

    public init(configuration: GBNFGrammarConfiguration) {
        self.vocabulary = configuration.vocabulary
        self.machine = GBNFMachine(rules: configuration.rules)
    }

    public func allowedTokenIDs() -> [Int] {
        var allowed: [Int] = []
        for (firstCharacter, bucket) in vocabulary.tokensByFirstCharacter {
            let probe = machine.validate(generatedText + String(firstCharacter))
            guard probe.isAllowed else { continue }
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
        generatedText += token.text
        memoizedCompletionState = nil
    }

    public var isComplete: Bool {
        completionState == .complete
    }

    public var text: String {
        generatedText
    }

    func makeMaskSnapshot() -> GBNFGrammarMaskSnapshot {
        GBNFGrammarMaskSnapshot(
            vocabulary: vocabulary,
            machine: machine,
            generatedText: generatedText
        )
    }

    private var completionState: GBNFValidationResult {
        if let memoizedCompletionState {
            return memoizedCompletionState
        }
        let state = machine.validate(generatedText)
        memoizedCompletionState = state
        return state
    }

    private func accepts(token: JSONGrammarToken) -> Bool {
        if token.isEOS {
            return isComplete
        }
        guard !token.text.isEmpty else {
            return false
        }
        return machine.validate(generatedText + token.text).isAllowed
    }
}

struct GBNFGrammarMaskSnapshot: GrammarMaskSnapshot {
    let vocabulary: JSONGrammarVocabulary
    fileprivate let machine: GBNFMachine
    let generatedText: String

    func allowedTokenIDs(isCancelled: @Sendable () -> Bool) -> [Int]? {
        var allowed: [Int] = []
        for (firstCharacter, bucket) in vocabulary.tokensByFirstCharacter {
            if isCancelled() { return nil }
            let probe = machine.validate(generatedText + String(firstCharacter))
            guard probe.isAllowed else { continue }
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
        machine.validate(generatedText) == .complete
    }

    private func accepts(token: JSONGrammarToken) -> Bool {
        if token.isEOS {
            return isComplete
        }
        guard !token.text.isEmpty else {
            return false
        }
        return machine.validate(generatedText + token.text).isAllowed
    }
}

private enum GBNFValidationResult {
    case complete
    case incomplete
    case invalid

    var isAllowed: Bool {
        switch self {
        case .complete, .incomplete:
            return true
        case .invalid:
            return false
        }
    }
}

private struct GBNFScalarRange: Sendable, Equatable, Hashable {
    let lower: UInt32
    let upper: UInt32

    func contains(_ scalar: Unicode.Scalar) -> Bool {
        lower <= scalar.value && scalar.value <= upper
    }
}

private struct GBNFCharacterClass: Sendable, Equatable, Hashable {
    let ranges: [GBNFScalarRange]
    let inverted: Bool

    func matches(_ character: Character) -> Bool {
        guard let scalar = String(character).unicodeScalars.only else {
            return false
        }
        let contained = ranges.contains { $0.contains(scalar) }
        return inverted ? !contained : contained
    }
}

private indirect enum GBNFExpr: Sendable, Equatable, Hashable {
    case empty
    case literal(Character)
    case any
    case characterClass(GBNFCharacterClass)
    case rule(String)
    case sequence([GBNFExpr])
    case alternate([GBNFExpr])
    case repeatNode(GBNFExpr, min: Int, max: Int?)
}

private struct GBNFState: Sendable, Equatable, Hashable {
    var stack: [GBNFExpr]
}

private struct GBNFMachine {
    let rules: [String: GBNFExpr]

    func validate(_ text: String) -> GBNFValidationResult {
        var states = epsilonClosure([GBNFState(stack: [.rule("root")])])
        for character in text {
            var next: Set<GBNFState> = []
            for state in states {
                guard let top = state.stack.last else { continue }
                var rest = state.stack
                _ = rest.popLast()
                switch top {
                case .literal(let expected) where expected == character:
                    next.insert(GBNFState(stack: rest))
                case .any where character != "\n":
                    next.insert(GBNFState(stack: rest))
                case .characterClass(let characterClass) where characterClass.matches(character):
                    next.insert(GBNFState(stack: rest))
                default:
                    continue
                }
            }
            states = epsilonClosure(next)
            if states.isEmpty {
                return .invalid
            }
        }
        return states.contains { $0.stack.isEmpty } ? .complete : .incomplete
    }

    private func epsilonClosure(_ seeds: some Sequence<GBNFState>) -> Set<GBNFState> {
        var result: Set<GBNFState> = []
        var stack = Array(seeds)
        while let state = stack.popLast() {
            guard result.insert(state).inserted else { continue }
            guard let top = state.stack.last else { continue }
            var rest = state.stack
            _ = rest.popLast()

            switch top {
            case .empty:
                stack.append(GBNFState(stack: rest))
            case .rule(let name):
                if let rule = rules[name] {
                    stack.append(GBNFState(stack: rest + [rule]))
                }
            case .sequence(let parts):
                stack.append(GBNFState(stack: rest + parts.reversed()))
            case .alternate(let branches):
                for branch in branches {
                    stack.append(GBNFState(stack: rest + [branch]))
                }
            case .repeatNode(let atom, let min, let max):
                if min == 0 {
                    stack.append(GBNFState(stack: rest))
                }
                guard max == nil || max ?? 0 > 0 else { continue }
                let remainingMin = Swift.max(0, min - 1)
                let remainingMax = max.map { $0 - 1 }
                var repeated = rest
                if remainingMax != 0 {
                    repeated.append(.repeatNode(atom, min: remainingMin, max: remainingMax))
                }
                repeated.append(atom)
                stack.append(GBNFState(stack: repeated))
            case .literal, .any, .characterClass:
                continue
            }
        }
        return result
    }
}

private enum GBNFValidator {
    static func validate(rules: [String: GBNFExpr]) throws {
        guard rules["root"] != nil else {
            throw GBNFGrammarError.unsupported("GBNF grammar requires a root rule")
        }
        for (name, expression) in rules {
            for reference in references(in: expression) where rules[reference] == nil {
                throw GBNFGrammarError.unsupported("GBNF rule '\(name)' references unknown rule '\(reference)'")
            }
        }
        let nullableRules = computeNullableRules(rules)
        let graph = rules.mapValues { nullableFirstReferences(in: $0, nullableRules: nullableRules) }
        if let cycle = leftRecursiveRule(in: graph) {
            throw GBNFGrammarError.unsupported("left-recursive GBNF rule '\(cycle)' is unsupported")
        }
    }

    private static func references(in expression: GBNFExpr) -> Set<String> {
        switch expression {
        case .rule(let name):
            return [name]
        case .sequence(let parts), .alternate(let parts):
            return parts.reduce(into: Set<String>()) { $0.formUnion(references(in: $1)) }
        case .repeatNode(let atom, _, _):
            return references(in: atom)
        case .empty, .literal, .any, .characterClass:
            return []
        }
    }

    private static func computeNullableRules(_ rules: [String: GBNFExpr]) -> Set<String> {
        var nullableRules: Set<String> = []
        var changed = true
        while changed {
            changed = false
            for (name, expression) in rules where !nullableRules.contains(name) {
                if isNullable(expression, nullableRules: nullableRules) {
                    nullableRules.insert(name)
                    changed = true
                }
            }
        }
        return nullableRules
    }

    private static func nullableFirstReferences(
        in expression: GBNFExpr,
        nullableRules: Set<String>
    ) -> Set<String> {
        switch expression {
        case .rule(let name):
            return [name]
        case .sequence(let parts):
            var result: Set<String> = []
            for part in parts {
                result.formUnion(nullableFirstReferences(in: part, nullableRules: nullableRules))
                if !isNullable(part, nullableRules: nullableRules) { break }
            }
            return result
        case .alternate(let branches):
            return branches.reduce(into: Set<String>()) {
                $0.formUnion(nullableFirstReferences(in: $1, nullableRules: nullableRules))
            }
        case .repeatNode(let atom, _, let max):
            guard max != 0 else { return [] }
            return nullableFirstReferences(in: atom, nullableRules: nullableRules)
        case .empty, .literal, .any, .characterClass:
            return []
        }
    }

    private static func isNullable(_ expression: GBNFExpr, nullableRules: Set<String>) -> Bool {
        switch expression {
        case .empty:
            return true
        case .alternate(let branches):
            return branches.contains { isNullable($0, nullableRules: nullableRules) }
        case .sequence(let parts):
            return parts.allSatisfy { isNullable($0, nullableRules: nullableRules) }
        case .repeatNode(let atom, let min, _):
            return min == 0 || isNullable(atom, nullableRules: nullableRules)
        case .rule(let name):
            return nullableRules.contains(name)
        case .literal, .any, .characterClass:
            return false
        }
    }

    private static func leftRecursiveRule(in graph: [String: Set<String>]) -> String? {
        let cyclicRules = graph.keys.sorted().filter { reaches($0, from: $0, graph: graph) }
        if cyclicRules.contains("root") {
            return "root"
        }
        return cyclicRules.first
    }

    private static func reaches(
        _ target: String,
        from start: String,
        graph: [String: Set<String>]
    ) -> Bool {
        var visited: Set<String> = []
        var stack = Array(graph[start] ?? [])
        while let node = stack.popLast() {
            if node == target {
                return true
            }
            guard visited.insert(node).inserted else { continue }
            stack.append(contentsOf: graph[node] ?? [])
        }
        return false
    }
}

private struct GBNFParser {
    private let grammar: String

    init(grammar: String) {
        self.grammar = grammar
    }

    func parse() throws -> [String: GBNFExpr] {
        var rules: [String: GBNFExpr] = [:]
        for rawLine in grammar.components(separatedBy: .newlines) {
            let line = stripComment(rawLine).trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            guard let separator = line.range(of: "::=") else {
                throw GBNFGrammarError.unsupported("malformed GBNF rule: \(line)")
            }
            let name = line[..<separator.lowerBound].trimmingCharacters(in: .whitespaces)
            guard isIdentifier(String(name)) else {
                throw GBNFGrammarError.unsupported("malformed GBNF rule name: \(name)")
            }
            let rhs = String(line[separator.upperBound...])
            var parser = GBNFExpressionParser(text: rhs)
            let expression = try parser.parseExpression()
            try parser.expectEnd()
            rules[String(name)] = expression
        }
        guard !rules.isEmpty else {
            throw GBNFGrammarError.unsupported("GBNF grammar is empty")
        }
        return rules
    }

    private func stripComment(_ line: String) -> String {
        var inString = false
        var inClass = false
        var escaped = false
        var result = ""
        for character in line {
            if escaped {
                result.append(character)
                escaped = false
                continue
            }
            if character == "\\" {
                result.append(character)
                escaped = true
                continue
            }
            if character == "\"" && !inClass {
                inString.toggle()
            } else if character == "[" && !inString {
                inClass = true
            } else if character == "]" && !inString {
                inClass = false
            } else if character == "#" && !inString && !inClass {
                break
            }
            result.append(character)
        }
        return result
    }

    private func isIdentifier(_ text: String) -> Bool {
        guard let first = text.first, first.isLetter || first == "_" else {
            return false
        }
        return text.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
    }
}

private struct GBNFExpressionParser {
    private let characters: [Character]
    private var index = 0

    init(text: String) {
        self.characters = Array(text)
    }

    mutating func parseExpression() throws -> GBNFExpr {
        try parseAlternation()
    }

    mutating func expectEnd() throws {
        skipWhitespace()
        guard isAtEnd else {
            throw GBNFGrammarError.unsupported("unsupported GBNF syntax near '\(current)'")
        }
    }

    private mutating func parseAlternation() throws -> GBNFExpr {
        var branches = [try parseSequence()]
        while true {
            skipWhitespace()
            guard consume("|") else { break }
            branches.append(try parseSequence())
        }
        return branches.count == 1 ? branches[0] : .alternate(branches)
    }

    private mutating func parseSequence() throws -> GBNFExpr {
        var parts: [GBNFExpr] = []
        while true {
            skipWhitespace()
            guard !isAtEnd, current != ")", current != "|" else { break }
            parts.append(try parsePostfix())
        }
        if parts.isEmpty { return .empty }
        return parts.count == 1 ? parts[0] : .sequence(parts)
    }

    private mutating func parsePostfix() throws -> GBNFExpr {
        let atom = try parseAtom()
        skipWhitespace()
        if consume("*") {
            return .repeatNode(atom, min: 0, max: nil)
        }
        if consume("+") {
            return .repeatNode(atom, min: 1, max: nil)
        }
        if consume("?") {
            return .repeatNode(atom, min: 0, max: 1)
        }
        if consume("{") {
            let bounds = try parseBounds()
            return .repeatNode(atom, min: bounds.min, max: bounds.max)
        }
        return atom
    }

    private mutating func parseAtom() throws -> GBNFExpr {
        skipWhitespace()
        if consume("(") {
            let expression = try parseAlternation()
            skipWhitespace()
            guard consume(")") else {
                throw GBNFGrammarError.unsupported("unclosed GBNF group")
            }
            return expression
        }
        if consume("\"") {
            return try parseStringLiteral()
        }
        if consume("[") {
            return .characterClass(try parseCharacterClass())
        }
        if consume(".") {
            return .any
        }
        if isIdentifierStart(current) {
            return .rule(parseIdentifier())
        }
        throw GBNFGrammarError.unsupported("unsupported GBNF atom near '\(current)'")
    }

    private mutating func parseStringLiteral() throws -> GBNFExpr {
        var parts: [GBNFExpr] = []
        while !isAtEnd, current != "\"" {
            parts.append(.literal(try parseEscapedCharacter()))
        }
        guard consume("\"") else {
            throw GBNFGrammarError.unsupported("unclosed GBNF string literal")
        }
        if parts.isEmpty { return .empty }
        return parts.count == 1 ? parts[0] : .sequence(parts)
    }

    private mutating func parseCharacterClass() throws -> GBNFCharacterClass {
        let inverted = consume("^")
        var ranges: [GBNFScalarRange] = []
        while !isAtEnd, current != "]" {
            let start = try parseClassScalar()
            if consume("-"), !isAtEnd, current != "]" {
                let end = try parseClassScalar()
                guard start.value <= end.value else {
                    throw GBNFGrammarError.unsupported("GBNF character class range is reversed")
                }
                ranges.append(GBNFScalarRange(lower: start.value, upper: end.value))
            } else {
                ranges.append(GBNFScalarRange(lower: start.value, upper: start.value))
            }
        }
        guard consume("]") else {
            throw GBNFGrammarError.unsupported("unclosed GBNF character class")
        }
        guard !ranges.isEmpty else {
            throw GBNFGrammarError.unsupported("GBNF character class is empty")
        }
        return GBNFCharacterClass(ranges: ranges, inverted: inverted)
    }

    private mutating func parseClassScalar() throws -> Unicode.Scalar {
        let character = try parseEscapedCharacter()
        guard let scalar = String(character).unicodeScalars.only else {
            throw GBNFGrammarError.unsupported("GBNF character class entries must be single Unicode scalars")
        }
        return scalar
    }

    private mutating func parseEscapedCharacter() throws -> Character {
        if consume("\\") {
            guard !isAtEnd else {
                throw GBNFGrammarError.unsupported("dangling GBNF escape")
            }
            let escaped = advance()
            switch escaped {
            case "b": return "\u{08}"
            case "f": return "\u{0C}"
            case "n": return "\n"
            case "r": return "\r"
            case "t": return "\t"
            case "x":
                return try parseHexEscapedCharacter(width: 2, escape: "x")
            case "u":
                return try parseHexEscapedCharacter(width: 4, escape: "u")
            case "U":
                return try parseHexEscapedCharacter(width: 8, escape: "U")
            case "\"", "\\", "[", "]", "(", ")", "{", "}", "-", "|", "*", "+", "?":
                return escaped
            default:
                throw GBNFGrammarError.unsupported("unsupported GBNF escape '\\\(escaped)'")
            }
        }
        return advance()
    }

    private mutating func parseHexEscapedCharacter(width: Int, escape: Character) throws -> Character {
        var digits = ""
        for _ in 0 ..< width {
            guard !isAtEnd, current.isHexDigit else {
                throw GBNFGrammarError.unsupported("invalid GBNF escape '\\\(escape)'")
            }
            digits.append(advance())
        }
        guard let value = UInt32(digits, radix: 16),
            let scalar = Unicode.Scalar(value)
        else {
            throw GBNFGrammarError.unsupported("invalid GBNF escape '\\\(escape)'")
        }
        return Character(scalar)
    }

    private mutating func parseBounds() throws -> (min: Int, max: Int?) {
        let min = try parseInteger()
        if consume("}") {
            return (min, min)
        }
        guard consume(",") else {
            throw GBNFGrammarError.unsupported("invalid GBNF repetition bound")
        }
        if consume("}") {
            return (min, nil)
        }
        let max = try parseInteger()
        guard consume("}") else {
            throw GBNFGrammarError.unsupported("invalid GBNF repetition bound")
        }
        guard max >= min else {
            throw GBNFGrammarError.unsupported("GBNF repetition max is smaller than min")
        }
        return (min, max)
    }

    private mutating func parseInteger() throws -> Int {
        var digits = ""
        while !isAtEnd, current.isNumber {
            digits.append(advance())
        }
        guard let value = Int(digits) else {
            throw GBNFGrammarError.unsupported("GBNF repetition bound must be an integer")
        }
        guard value <= 256 else {
            throw GBNFGrammarError.unsupported("GBNF repetition bounds above 256 are unsupported")
        }
        return value
    }

    private mutating func parseIdentifier() -> String {
        var name = ""
        while !isAtEnd, current.isLetter || current.isNumber || current == "_" || current == "-" {
            name.append(advance())
        }
        return name
    }

    private mutating func skipWhitespace() {
        while !isAtEnd, current == " " || current == "\t" {
            index += 1
        }
    }

    private func isIdentifierStart(_ character: Character) -> Bool {
        character.isLetter || character == "_"
    }

    private var isAtEnd: Bool {
        index >= characters.count
    }

    private var current: Character {
        isAtEnd ? "\0" : characters[index]
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

private extension String.UnicodeScalarView {
    var only: Unicode.Scalar? {
        guard count == 1 else { return nil }
        return first
    }
}
