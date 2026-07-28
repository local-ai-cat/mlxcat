import Foundation

public struct JSONGrammarToken: Sendable, Equatable {
    public let id: Int
    public let text: String
    public let isEOS: Bool

    public init(id: Int, text: String, isEOS: Bool = false) {
        self.id = id
        self.text = text
        self.isEOS = isEOS
    }
}

/// Immutable vocabulary index shared by every matcher for a model. Built once per tokenizer
/// (O(vocab)); matchers are created per request row and must stay cheap to construct.
public final class JSONGrammarVocabulary: Sendable, Equatable {
    public let tokens: [JSONGrammarToken]
    let tokensByID: [Int: JSONGrammarToken]
    let tokensByFirstCharacter: [Character: [JSONGrammarToken]]
    let eosTokenIDs: [Int]

    public init(tokens: [JSONGrammarToken]) {
        self.tokens = tokens
        var byID: [Int: JSONGrammarToken] = Dictionary(minimumCapacity: tokens.count)
        var byFirstCharacter: [Character: [JSONGrammarToken]] = [:]
        var eosIDs: [Int] = []
        for token in tokens {
            byID[token.id] = token
            if token.isEOS {
                eosIDs.append(token.id)
            } else if let firstCharacter = token.text.first {
                byFirstCharacter[firstCharacter, default: []].append(token)
            }
        }
        self.tokensByID = byID
        self.tokensByFirstCharacter = byFirstCharacter
        self.eosTokenIDs = eosIDs
    }

    public static func == (lhs: JSONGrammarVocabulary, rhs: JSONGrammarVocabulary) -> Bool {
        lhs === rhs || lhs.tokens == rhs.tokens
    }
}

public struct JSONGrammarConfiguration: Sendable, Equatable {
    public let vocabulary: JSONGrammarVocabulary
    public let schema: JSONSchemaNode

    public var tokens: [JSONGrammarToken] { vocabulary.tokens }

    public init(vocabulary: JSONGrammarVocabulary, schema: JSONSchemaNode) {
        self.vocabulary = vocabulary
        self.schema = schema
    }

    public init(tokens: [JSONGrammarToken], schema: JSONSchemaNode) {
        self.init(vocabulary: JSONGrammarVocabulary(tokens: tokens), schema: schema)
    }

    public func makeMatcher() -> JSONGrammarMatcher {
        JSONGrammarMatcher(configuration: self)
    }
}

public enum JSONValueKind: String, Sendable, Hashable {
    case object
    case array
    case string
    case number
    case boolean
    case null
}

public enum JSONLiteral: Sendable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
}

public indirect enum JSONSchemaNode: Sendable, Equatable {
    case any
    case schema(JSONSchema)

    public static var jsonObject: JSONSchemaNode {
        .schema(
            JSONSchema(
                allowedTypes: [.object],
                properties: [:],
                required: [],
                enumValues: nil,
                items: nil,
                additionalPropertiesAllowed: true
            )
        )
    }
}

public struct JSONSchema: Sendable, Equatable {
    public let allowedTypes: Set<JSONValueKind>?
    public let properties: [String: JSONSchemaNode]
    public let required: Set<String>
    public let enumValues: Set<JSONLiteral>?
    public let items: JSONSchemaNode?
    public let additionalPropertiesAllowed: Bool

    public init(
        allowedTypes: Set<JSONValueKind>? = nil,
        properties: [String: JSONSchemaNode] = [:],
        required: Set<String> = [],
        enumValues: Set<JSONLiteral>? = nil,
        items: JSONSchemaNode? = nil,
        additionalPropertiesAllowed: Bool = true
    ) {
        self.allowedTypes = allowedTypes
        self.properties = properties
        self.required = required
        self.enumValues = enumValues
        self.items = items
        self.additionalPropertiesAllowed = additionalPropertiesAllowed
    }
}

public final class JSONGrammarMatcher {
    private let vocabulary: JSONGrammarVocabulary
    private let schema: JSONSchemaNode
    private var generatedText = ""
    private var memoizedCompletionState: JSONPrefixValidationResult?

    public init(configuration: JSONGrammarConfiguration) {
        self.vocabulary = configuration.vocabulary
        self.schema = configuration.schema
    }

    public func allowedTokenIDs() -> [Int] {
        var allowed: [Int] = []
        for (firstCharacter, bucket) in vocabulary.tokensByFirstCharacter {
            // Prefix validity is monotone: once a one-character extension is invalid, every
            // longer token starting with that character is invalid too — skip the bucket.
            let probe = JSONPrefixValidator.validate(
                generatedText + String(firstCharacter),
                schema: schema
            )
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
        guard let token = vocabulary.tokensByID[tokenID] else {
            return
        }
        guard !token.isEOS else { return }
        generatedText += token.text
        memoizedCompletionState = nil
    }

    public var isComplete: Bool {
        completionState == .complete
    }

    public var text: String {
        generatedText
    }

    func makeMaskSnapshot() -> JSONGrammarMaskSnapshot {
        JSONGrammarMaskSnapshot(
            vocabulary: vocabulary,
            schema: schema,
            generatedText: generatedText
        )
    }

    private var completionState: JSONPrefixValidationResult {
        if let memoizedCompletionState {
            return memoizedCompletionState
        }
        let state = JSONPrefixValidator.validate(generatedText, schema: schema)
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
        return JSONPrefixValidator.validate(generatedText + token.text, schema: schema).isAllowed
    }
}

struct JSONGrammarMaskSnapshot: GrammarMaskSnapshot {
    let vocabulary: JSONGrammarVocabulary
    let schema: JSONSchemaNode
    let generatedText: String

    func allowedTokenIDs(isCancelled: @Sendable () -> Bool) -> [Int]? {
        var allowed: [Int] = []
        for (firstCharacter, bucket) in vocabulary.tokensByFirstCharacter {
            if isCancelled() { return nil }
            let probe = JSONPrefixValidator.validate(
                generatedText + String(firstCharacter),
                schema: schema
            )
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
        JSONPrefixValidator.validate(generatedText, schema: schema) == .complete
    }

    private func accepts(token: JSONGrammarToken) -> Bool {
        if token.isEOS {
            return isComplete
        }
        guard !token.text.isEmpty else {
            return false
        }
        return JSONPrefixValidator.validate(generatedText + token.text, schema: schema).isAllowed
    }
}

private enum JSONPrefixValidationResult {
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

private enum JSONParseResult<T> {
    case complete(T, String.Index)
    case incomplete
    case invalid
}

private struct JSONParsedValue: Equatable {
    let kind: JSONValueKind
    let literal: JSONLiteral?
}

private enum JSONPrefixValidator {
    static func validate(_ text: String, schema: JSONSchemaNode) -> JSONPrefixValidationResult {
        let parser = JSONPrefixParser(text: text)
        let start = parser.skipWhitespace(from: text.startIndex)
        switch parser.parseValue(from: start, schema: schema) {
        case .complete(_, let index):
            let end = parser.skipWhitespace(from: index)
            return end == text.endIndex ? .complete : .invalid
        case .incomplete:
            return .incomplete
        case .invalid:
            return .invalid
        }
    }
}

private struct JSONPrefixParser {
    let text: String

    func parseValue(from index: String.Index, schema: JSONSchemaNode) -> JSONParseResult<JSONParsedValue> {
        let index = skipWhitespace(from: index)
        guard index < text.endIndex else {
            return .incomplete
        }

        switch text[index] {
        case "{":
            guard schemaAllows(.object, schema: schema) else { return .invalid }
            return parseObject(from: index, schema: schema)
        case "[":
            guard schemaAllows(.array, schema: schema) else { return .invalid }
            return parseArray(from: index, schema: schema)
        case "\"":
            guard schemaAllows(.string, schema: schema) else { return .invalid }
            return parseStringValue(from: index, schema: schema)
        case "-", "0"..."9":
            guard schemaAllows(.number, schema: schema) else { return .invalid }
            return parseNumber(from: index, schema: schema)
        case "t":
            guard schemaAllows(.boolean, schema: schema) else { return .invalid }
            return parseLiteral("true", value: .bool(true), kind: .boolean, from: index, schema: schema)
        case "f":
            guard schemaAllows(.boolean, schema: schema) else { return .invalid }
            return parseLiteral("false", value: .bool(false), kind: .boolean, from: index, schema: schema)
        case "n":
            guard schemaAllows(.null, schema: schema) else { return .invalid }
            return parseLiteral("null", value: .null, kind: .null, from: index, schema: schema)
        default:
            return .invalid
        }
    }

    func parseObject(from index: String.Index, schema: JSONSchemaNode) -> JSONParseResult<JSONParsedValue> {
        let objectSchema = schema.objectSchema
        var cursor = text.index(after: index)
        var seenKeys = Set<String>()

        cursor = skipWhitespace(from: cursor)
        guard cursor < text.endIndex else {
            return .incomplete
        }
        if text[cursor] == "}" {
            guard objectSchema.required.isSubset(of: seenKeys) else {
                return .invalid
            }
            return completeContainer(kind: .object, literal: nil, end: text.index(after: cursor), schema: schema)
        }

        while true {
            let keyResult = parseObjectKey(from: cursor, schema: objectSchema, seenKeys: seenKeys)
            let key: String
            switch keyResult {
            case .complete(let parsedKey, let nextIndex):
                key = parsedKey
                cursor = skipWhitespace(from: nextIndex)
            case .incomplete:
                return .incomplete
            case .invalid:
                return .invalid
            }

            guard cursor < text.endIndex else {
                return .incomplete
            }
            guard text[cursor] == ":" else {
                return .invalid
            }
            cursor = text.index(after: cursor)

            let propertySchema = objectSchema.properties[key] ?? .any
            switch parseValue(from: cursor, schema: propertySchema) {
            case .complete(_, let nextIndex):
                seenKeys.insert(key)
                cursor = skipWhitespace(from: nextIndex)
            case .incomplete:
                return .incomplete
            case .invalid:
                return .invalid
            }

            guard cursor < text.endIndex else {
                return .incomplete
            }
            if text[cursor] == "," {
                cursor = skipWhitespace(from: text.index(after: cursor))
                guard cursor < text.endIndex else {
                    return .incomplete
                }
                continue
            }
            if text[cursor] == "}" {
                guard objectSchema.required.isSubset(of: seenKeys) else {
                    return .invalid
                }
                return completeContainer(kind: .object, literal: nil, end: text.index(after: cursor), schema: schema)
            }
            return .invalid
        }
    }

    func parseArray(from index: String.Index, schema: JSONSchemaNode) -> JSONParseResult<JSONParsedValue> {
        let itemSchema = schema.arrayItemSchema
        var cursor = skipWhitespace(from: text.index(after: index))
        guard cursor < text.endIndex else {
            return .incomplete
        }
        if text[cursor] == "]" {
            return completeContainer(kind: .array, literal: nil, end: text.index(after: cursor), schema: schema)
        }

        while true {
            switch parseValue(from: cursor, schema: itemSchema) {
            case .complete(_, let nextIndex):
                cursor = skipWhitespace(from: nextIndex)
            case .incomplete:
                return .incomplete
            case .invalid:
                return .invalid
            }

            guard cursor < text.endIndex else {
                return .incomplete
            }
            if text[cursor] == "," {
                cursor = skipWhitespace(from: text.index(after: cursor))
                guard cursor < text.endIndex else {
                    return .incomplete
                }
                continue
            }
            if text[cursor] == "]" {
                return completeContainer(kind: .array, literal: nil, end: text.index(after: cursor), schema: schema)
            }
            return .invalid
        }
    }

    func parseStringValue(from index: String.Index, schema: JSONSchemaNode) -> JSONParseResult<JSONParsedValue> {
        switch parseString(from: index, allowedPrefixes: schema.stringEnumValues) {
        case .complete(let value, let nextIndex):
            let literal = JSONLiteral.string(value)
            guard schemaAllowsEnum(literal, schema: schema) else {
                return .invalid
            }
            return .complete(JSONParsedValue(kind: .string, literal: literal), nextIndex)
        case .incomplete:
            return .incomplete
        case .invalid:
            return .invalid
        }
    }

    func parseNumber(from index: String.Index, schema: JSONSchemaNode) -> JSONParseResult<JSONParsedValue> {
        let parsed = scanNumber(from: index)
        switch parsed {
        case .complete(let raw, let nextIndex):
            guard let number = Double(raw) else {
                return .invalid
            }
            let literal = JSONLiteral.number(number)
            guard schemaAllowsEnum(literal, schema: schema) else {
                return .invalid
            }
            return .complete(JSONParsedValue(kind: .number, literal: literal), nextIndex)
        case .incomplete:
            return .incomplete
        case .invalid:
            return .invalid
        }
    }

    func parseLiteral(
        _ expected: String,
        value: JSONLiteral,
        kind: JSONValueKind,
        from index: String.Index,
        schema: JSONSchemaNode
    ) -> JSONParseResult<JSONParsedValue> {
        var cursor = index
        for expectedCharacter in expected {
            guard cursor < text.endIndex else {
                return .incomplete
            }
            guard text[cursor] == expectedCharacter else {
                return .invalid
            }
            cursor = text.index(after: cursor)
        }
        guard schemaAllowsEnum(value, schema: schema) else {
            return .invalid
        }
        return .complete(JSONParsedValue(kind: kind, literal: value), cursor)
    }

    func parseObjectKey(
        from index: String.Index,
        schema: JSONSchema,
        seenKeys: Set<String>
    ) -> JSONParseResult<String> {
        let allowedKeys: Set<String>?
        if schema.additionalPropertiesAllowed {
            allowedKeys = nil
        } else {
            allowedKeys = Set(schema.properties.keys).subtracting(seenKeys)
        }
        return parseString(from: index, allowedPrefixes: allowedKeys)
    }

    func parseString(
        from index: String.Index,
        allowedPrefixes: Set<String>?
    ) -> JSONParseResult<String> {
        guard index < text.endIndex, text[index] == "\"" else {
            return .invalid
        }

        var cursor = text.index(after: index)
        var value = ""
        while cursor < text.endIndex {
            let character = text[cursor]
            if character == "\"" {
                if let allowedPrefixes, !allowedPrefixes.contains(value) {
                    return .invalid
                }
                return .complete(value, text.index(after: cursor))
            }
            if character == "\\" {
                cursor = text.index(after: cursor)
                guard cursor < text.endIndex else {
                    return prefixAllowed(value, allowedPrefixes: allowedPrefixes) ? .incomplete : .invalid
                }
                switch escapedCharacter(at: cursor, cursor: &cursor) {
                case .complete(let escaped, _):
                    value.append(escaped)
                case .incomplete:
                    return prefixAllowed(value, allowedPrefixes: allowedPrefixes) ? .incomplete : .invalid
                case .invalid:
                    return .invalid
                }
            } else {
                guard character >= " " else {
                    return .invalid
                }
                value.append(character)
                cursor = text.index(after: cursor)
            }
            guard prefixAllowed(value, allowedPrefixes: allowedPrefixes) else {
                return .invalid
            }
        }
        return prefixAllowed(value, allowedPrefixes: allowedPrefixes) ? .incomplete : .invalid
    }

    func scanNumber(from index: String.Index) -> JSONParseResult<String> {
        var cursor = index
        var raw = ""

        if cursor < text.endIndex, text[cursor] == "-" {
            raw.append(text[cursor])
            cursor = text.index(after: cursor)
            guard cursor < text.endIndex else {
                return .incomplete
            }
        }

        guard cursor < text.endIndex else {
            return .incomplete
        }
        if text[cursor] == "0" {
            raw.append(text[cursor])
            cursor = text.index(after: cursor)
            if cursor < text.endIndex, text[cursor].isDigit {
                return .invalid
            }
        } else if text[cursor].isOneThroughNine {
            while cursor < text.endIndex, text[cursor].isDigit {
                raw.append(text[cursor])
                cursor = text.index(after: cursor)
            }
        } else {
            return .invalid
        }

        if cursor < text.endIndex, text[cursor] == "." {
            raw.append(text[cursor])
            cursor = text.index(after: cursor)
            guard cursor < text.endIndex else {
                return .incomplete
            }
            guard text[cursor].isDigit else {
                return .invalid
            }
            while cursor < text.endIndex, text[cursor].isDigit {
                raw.append(text[cursor])
                cursor = text.index(after: cursor)
            }
        }

        if cursor < text.endIndex, text[cursor] == "e" || text[cursor] == "E" {
            raw.append(text[cursor])
            cursor = text.index(after: cursor)
            guard cursor < text.endIndex else {
                return .incomplete
            }
            if text[cursor] == "+" || text[cursor] == "-" {
                raw.append(text[cursor])
                cursor = text.index(after: cursor)
                guard cursor < text.endIndex else {
                    return .incomplete
                }
            }
            guard text[cursor].isDigit else {
                return .invalid
            }
            while cursor < text.endIndex, text[cursor].isDigit {
                raw.append(text[cursor])
                cursor = text.index(after: cursor)
            }
        }

        guard cursor == text.endIndex || text[cursor].isJSONValueDelimiter else {
            return .invalid
        }
        return .complete(raw, cursor)
    }

    func skipWhitespace(from index: String.Index) -> String.Index {
        var cursor = index
        while cursor < text.endIndex, text[cursor].isJSONWhitespace {
            cursor = text.index(after: cursor)
        }
        return cursor
    }

    private func completeContainer(
        kind: JSONValueKind,
        literal: JSONLiteral?,
        end: String.Index,
        schema: JSONSchemaNode
    ) -> JSONParseResult<JSONParsedValue> {
        if let literal {
            guard schemaAllowsEnum(literal, schema: schema) else {
                return .invalid
            }
        }
        return .complete(JSONParsedValue(kind: kind, literal: literal), end)
    }

    private func escapedCharacter(at index: String.Index, cursor: inout String.Index) -> JSONParseResult<Character> {
        let character = text[index]
        cursor = text.index(after: index)
        switch character {
        case "\"", "\\", "/":
            return .complete(character, cursor)
        case "b":
            return .complete("\u{08}", cursor)
        case "f":
            return .complete("\u{0C}", cursor)
        case "n":
            return .complete("\n", cursor)
        case "r":
            return .complete("\r", cursor)
        case "t":
            return .complete("\t", cursor)
        case "u":
            switch hexEscapeValue(cursor: &cursor) {
            case .complete(let value, _):
                if (0xD800...0xDBFF).contains(value) {
                    return lowSurrogate(after: value, cursor: &cursor)
                }
                guard !(0xDC00...0xDFFF).contains(value), let unicodeScalar = UnicodeScalar(value) else {
                    return .invalid
                }
                return .complete(Character(unicodeScalar), cursor)
            case .incomplete:
                return .incomplete
            case .invalid:
                return .invalid
            }
        default:
            return .invalid
        }
    }

    /// Parses the four hex digits of a `\u` escape. Truncation by the end of text is a
    /// valid *prefix* — the missing digits can still arrive in a later token; treating it
    /// as invalid would (via bucket pruning) mask every token that continues the escape.
    private func hexEscapeValue(cursor: inout String.Index) -> JSONParseResult<UInt32> {
        var scalar = ""
        for _ in 0..<4 {
            guard cursor < text.endIndex else {
                return .incomplete
            }
            guard text[cursor].isHexDigit else {
                return .invalid
            }
            scalar.append(text[cursor])
            cursor = text.index(after: cursor)
        }
        guard let value = UInt32(scalar, radix: 16) else {
            return .invalid
        }
        return .complete(value, cursor)
    }

    /// A high surrogate must be followed by a `\uDC00`–`\uDFFF` escape (JSON encodes
    /// non-BMP characters as surrogate pairs). Truncation anywhere inside the second
    /// escape is an incomplete prefix, not an error.
    private func lowSurrogate(after highSurrogate: UInt32, cursor: inout String.Index) -> JSONParseResult<Character> {
        for expected in ["\\", "u"] as [Character] {
            guard cursor < text.endIndex else {
                return .incomplete
            }
            guard text[cursor] == expected else {
                return .invalid
            }
            cursor = text.index(after: cursor)
        }
        switch hexEscapeValue(cursor: &cursor) {
        case .complete(let lowValue, _):
            guard (0xDC00...0xDFFF).contains(lowValue) else {
                return .invalid
            }
            let combined = 0x10000 + ((highSurrogate - 0xD800) << 10) + (lowValue - 0xDC00)
            guard let unicodeScalar = UnicodeScalar(combined) else {
                return .invalid
            }
            return .complete(Character(unicodeScalar), cursor)
        case .incomplete:
            return .incomplete
        case .invalid:
            return .invalid
        }
    }

    private func schemaAllows(_ kind: JSONValueKind, schema: JSONSchemaNode) -> Bool {
        switch schema {
        case .any:
            return true
        case .schema(let schema):
            if let allowedTypes = schema.allowedTypes, !allowedTypes.contains(kind) {
                return false
            }
            guard let enumValues = schema.enumValues else {
                return true
            }
            return enumValues.contains { $0.kind == kind }
        }
    }

    private func schemaAllowsEnum(_ literal: JSONLiteral, schema: JSONSchemaNode) -> Bool {
        switch schema {
        case .any:
            return true
        case .schema(let schema):
            guard let enumValues = schema.enumValues else {
                return true
            }
            return enumValues.contains(literal)
        }
    }

    private func prefixAllowed(_ value: String, allowedPrefixes: Set<String>?) -> Bool {
        guard let allowedPrefixes else {
            return true
        }
        return allowedPrefixes.contains { $0.hasPrefix(value) }
    }
}

private extension JSONSchemaNode {
    var objectSchema: JSONSchema {
        switch self {
        case .any:
            return JSONSchema(allowedTypes: [.object])
        case .schema(let schema):
            return schema
        }
    }

    var arrayItemSchema: JSONSchemaNode {
        switch self {
        case .any:
            return .any
        case .schema(let schema):
            return schema.items ?? .any
        }
    }

    var stringEnumValues: Set<String>? {
        switch self {
        case .any:
            return nil
        case .schema(let schema):
            guard let enumValues = schema.enumValues else {
                return nil
            }
            let strings = enumValues.compactMap { literal -> String? in
                if case .string(let value) = literal {
                    return value
                }
                return nil
            }
            return strings.count == enumValues.count ? Set(strings) : nil
        }
    }
}

private extension JSONLiteral {
    var kind: JSONValueKind {
        switch self {
        case .string:
            return .string
        case .number:
            return .number
        case .bool:
            return .boolean
        case .null:
            return .null
        }
    }
}

private extension Character {
    var isJSONWhitespace: Bool {
        self == " " || self == "\n" || self == "\r" || self == "\t"
    }

    var isJSONValueDelimiter: Bool {
        isJSONWhitespace || self == "," || self == "]" || self == "}"
    }

    var isDigit: Bool {
        self >= "0" && self <= "9"
    }

    var isOneThroughNine: Bool {
        self >= "1" && self <= "9"
    }

    var isHexDigit: Bool {
        isDigit || (self >= "a" && self <= "f") || (self >= "A" && self <= "F")
    }
}
