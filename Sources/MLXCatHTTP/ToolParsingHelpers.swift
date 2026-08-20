import Foundation

// Shared, format-agnostic helpers used by the per-family tool parsers.

// MARK: - JSON

func parseJSONObject(_ text: String) -> [String: Any]? {
    guard let data = text.data(using: .utf8), !data.isEmpty else { return nil }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

func parseJSONFragmentValue(_ text: String) -> Any? {
    guard let data = text.data(using: .utf8), !data.isEmpty else { return nil }
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}

func isCompleteJSON(_ text: String) -> Bool {
    parseJSONFragmentValue(text.trimmingCharacters(in: .whitespacesAndNewlines)) != nil
}

/// Serialize a JSON-compatible value to a compact string. Objects use sorted
/// keys for deterministic output (JSONSerialization does not preserve key order
/// through a parse, so re-sorting is the only stable choice).
func serializeJSONValue(_ value: Any) -> String {
    if let object = value as? [String: Any] {
        return serializeJSONObject(object)
    }
    if JSONSerialization.isValidJSONObject([value]),
        let data = try? JSONSerialization.data(withJSONObject: [value]),
        let array = String(data: data, encoding: .utf8) {
        // Strip the wrapping [ ] we added to make a fragment serializable.
        return String(array.dropFirst().dropLast())
    }
    return "{}"
}

func serializeJSONObject(_ object: [String: Any]) -> String {
    guard JSONSerialization.isValidJSONObject(object),
        let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
        let string = String(data: data, encoding: .utf8)
    else {
        return "{}"
    }
    return string
}

/// vLLM `json.dumps(arguments)`: a dict/array is re-serialized, a string is
/// returned verbatim, `nil` becomes `{}`.
func argumentsJSONString(from value: Any?) -> String {
    guard let value else { return "{}" }
    if let string = value as? String { return string }
    if let object = value as? [String: Any] { return serializeJSONObject(object) }
    return serializeJSONValue(value)
}

/// Parse a raw argument string to a compact JSON object string; a valid JSON
/// object is re-serialized with sorted keys, otherwise the trimmed text is
/// returned (`{}` when empty).
func normalizedArgumentsJSON(from text: String) -> String {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return "{}" }
    if let object = parseJSONObject(trimmed) {
        return serializeJSONObject(object)
    }
    return trimmed
}

// MARK: - Tag / span scanning

/// Longest suffix of `text` that is a prefix of `tag` (vLLM `partial_tag_overlap`).
/// Lets a streaming parser hold back a partial closing/opening tag.
func partialTagOverlap(_ text: String, _ tag: String) -> Int {
    let maxLength = min(text.count, tag.count - 1)
    guard maxLength > 0 else { return 0 }
    for length in stride(from: maxLength, through: 1, by: -1) {
        if text.hasSuffix(String(tag.prefix(length))) {
            return length
        }
    }
    return 0
}

/// Index of the `}` that closes the `{` at `openBrace`, honoring nested braces
/// and double-quoted strings (Gemma's `<|"|>` delimiter opens with `"`, so a
/// plain double-quote scan balances it).
func matchingBraceEnd(in text: String, openBrace: String.Index) -> String.Index? {
    matchingCloser(in: text, open: openBrace, openChar: "{", closeChar: "}")
}

/// Index of the `)` that closes the `(` at `openParen`, honoring nested
/// brackets/braces/parens and quotes.
func matchingParenEnd(in text: String, openParen: String.Index) -> String.Index? {
    matchingCloser(in: text, open: openParen, openChar: "(", closeChar: ")")
}

private func matchingCloser(
    in text: String,
    open: String.Index,
    openChar: Character,
    closeChar: Character
) -> String.Index? {
    var depth = 0
    var inString = false
    var stringDelimiter: Character = "\""
    var escaped = false
    var index = open
    while index < text.endIndex {
        let character = text[index]
        if inString {
            if escaped {
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == stringDelimiter {
                inString = false
            }
        } else if character == "\"" || character == "'" {
            inString = true
            stringDelimiter = character
        } else if character == "(" || character == "[" || character == "{" {
            depth += 1
        } else if character == ")" || character == "]" || character == "}" {
            depth -= 1
            if depth == 0, character == closeChar {
                return index
            }
        }
        index = text.index(after: index)
    }
    return nil
}

/// Split `text` on `separator` at brace/bracket/paren/quote depth 0.
func splitTopLevel(_ text: String, separator: Character) -> [String] {
    var parts: [String] = []
    var current = ""
    var depth = 0
    var inString = false
    var stringDelimiter: Character = "\""
    var index = text.startIndex
    while index < text.endIndex {
        let character = text[index]
        if inString {
            if character == stringDelimiter { inString = false }
            current.append(character)
        } else if character == "\"" || character == "'" {
            inString = true
            stringDelimiter = character
            current.append(character)
        } else if character == "{" || character == "[" || character == "(" {
            depth += 1
            current.append(character)
        } else if character == "}" || character == "]" || character == ")" {
            depth -= 1
            current.append(character)
        } else if character == separator, depth == 0 {
            parts.append(current)
            current = ""
        } else {
            current.append(character)
        }
        index = text.index(after: index)
    }
    if !current.isEmpty {
        parts.append(current)
    }
    return parts
}

/// First `target` at brace/bracket/paren/quote depth 0.
func topLevelIndex(of target: Character, in text: String) -> String.Index? {
    var depth = 0
    var inString = false
    var stringDelimiter: Character = "\""
    var index = text.startIndex
    while index < text.endIndex {
        let character = text[index]
        if inString {
            if character == stringDelimiter { inString = false }
        } else if character == "\"" || character == "'" {
            inString = true
            stringDelimiter = character
        } else if character == "{" || character == "[" || character == "(" {
            depth += 1
        } else if character == "}" || character == "]" || character == ")" {
            depth -= 1
        } else if character == target, depth == 0 {
            return index
        }
        index = text.index(after: index)
    }
    return nil
}

// MARK: - Python literal values (pythonic parser)

/// Convert a Python/JSON literal token to a JSON-compatible Swift value.
func pythonLiteralValue(_ raw: String) -> Any {
    let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { return "" }

    if (text.hasPrefix("\"") && text.hasSuffix("\"") && text.count >= 2)
        || (text.hasPrefix("'") && text.hasSuffix("'") && text.count >= 2) {
        return String(text.dropFirst().dropLast())
    }
    switch text {
    case "True", "true": return true
    case "False", "false": return false
    case "None", "null": return NSNull()
    default: break
    }
    if let intValue = Int(text) { return intValue }
    if let doubleValue = Double(text) { return doubleValue }
    if text.hasPrefix("[") && text.hasSuffix("]") {
        let inner = String(text.dropFirst().dropLast())
        return splitTopLevel(inner, separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map(pythonLiteralValue)
    }
    if text.hasPrefix("{") && text.hasSuffix("}") {
        let inner = String(text.dropFirst().dropLast())
        var dict: [String: Any] = [:]
        for entry in splitTopLevel(inner, separator: ",") {
            guard let colon = topLevelIndex(of: ":", in: entry) else { continue }
            let keyRaw = String(entry[entry.startIndex..<colon]).trimmingCharacters(in: .whitespacesAndNewlines)
            let key = (keyRaw.hasPrefix("\"") || keyRaw.hasPrefix("'")) && keyRaw.count >= 2
                ? String(keyRaw.dropFirst().dropLast())
                : keyRaw
            let value = String(entry[entry.index(after: colon)...])
            if !key.isEmpty { dict[key] = pythonLiteralValue(value) }
        }
        return dict
    }
    return text
}

// MARK: - Gemma 4 argument grammar

private let gemmaDelim = Array("<|\"|>")

/// Port of vLLM `parser/gemma4._parse_gemma4_args` (non-partial path): parse
/// Gemma's `key:<|"|>value<|"|>` / bare / nested `{}` / `[]` argument grammar.
func parseGemma4Args(_ argsStr: String) -> [String: Any] {
    let s = Array(argsStr)
    let n = s.count
    let dlen = gemmaDelim.count
    var result: [String: Any] = [:]
    var i = 0

    func matchesDelim(at index: Int) -> Bool {
        index + dlen <= n && Array(s[index..<index + dlen]) == gemmaDelim
    }
    func findDelim(from index: Int) -> Int {
        var j = index
        while j + dlen <= n {
            if Array(s[j..<j + dlen]) == gemmaDelim { return j }
            j += 1
        }
        return -1
    }

    while i < n {
        while i < n, s[i] == " " || s[i] == "," || s[i] == "\n" || s[i] == "\t" { i += 1 }
        if i >= n { break }

        let keyStart = i
        while i < n, s[i] != ":" { i += 1 }
        if i >= n { break }
        var key = String(s[keyStart..<i]).trimmingCharacters(in: .whitespacesAndNewlines)
        if key.hasPrefix("<|\"|>") && key.hasSuffix("<|\"|>") && key.count >= 2 * dlen {
            key = String(key.dropFirst(dlen).dropLast(dlen))
        }
        i += 1 // skip ':'

        if i >= n { result[key] = ""; break }
        while i < n, s[i] == " " || s[i] == "\n" || s[i] == "\t" { i += 1 }
        if i >= n { result[key] = ""; break }

        if matchesDelim(at: i) {
            i += dlen
            let valStart = i
            let end = findDelim(from: i)
            if end == -1 {
                result[key] = String(s[valStart..<n])
                break
            }
            result[key] = String(s[valStart..<end])
            i = end + dlen
        } else if s[i] == "{" {
            var depth = 1
            let objStart = i + 1
            i += 1
            while i < n, depth > 0 {
                if matchesDelim(at: i) {
                    i += dlen
                    let next = findDelim(from: i)
                    i = next == -1 ? n : next + dlen
                    continue
                }
                if s[i] == "{" { depth += 1 } else if s[i] == "}" { depth -= 1 }
                i += 1
            }
            let end = depth > 0 ? i : i - 1
            result[key] = parseGemma4Args(String(s[objStart..<max(objStart, end)]))
        } else if s[i] == "[" {
            var depth = 1
            let arrStart = i + 1
            i += 1
            while i < n, depth > 0 {
                if matchesDelim(at: i) {
                    i += dlen
                    let next = findDelim(from: i)
                    i = next == -1 ? n : next + dlen
                    continue
                }
                if s[i] == "[" { depth += 1 } else if s[i] == "]" { depth -= 1 }
                i += 1
            }
            let end = depth > 0 ? i : i - 1
            result[key] = parseGemma4Array(String(s[arrStart..<max(arrStart, end)]))
        } else {
            let valStart = i
            while i < n, s[i] != ",", s[i] != "}", s[i] != "]" { i += 1 }
            if i == valStart { break }
            result[key] = String(s[valStart..<i]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
    return result
}

/// Port of vLLM `parser/gemma4._parse_gemma4_array` (non-partial path).
func parseGemma4Array(_ arrStr: String) -> [Any] {
    let s = Array(arrStr)
    let n = s.count
    let dlen = gemmaDelim.count
    var items: [Any] = []
    var i = 0

    func matchesDelim(at index: Int) -> Bool {
        index + dlen <= n && Array(s[index..<index + dlen]) == gemmaDelim
    }
    func findDelim(from index: Int) -> Int {
        var j = index
        while j + dlen <= n {
            if Array(s[j..<j + dlen]) == gemmaDelim { return j }
            j += 1
        }
        return -1
    }

    while i < n {
        while i < n, s[i] == " " || s[i] == "," || s[i] == "\n" || s[i] == "\t" { i += 1 }
        if i >= n { break }

        if matchesDelim(at: i) {
            i += dlen
            let end = findDelim(from: i)
            if end == -1 { items.append(String(s[i..<n])); break }
            items.append(String(s[i..<end]))
            i = end + dlen
        } else if s[i] == "{" {
            var depth = 1
            let objStart = i + 1
            i += 1
            while i < n, depth > 0 {
                if matchesDelim(at: i) {
                    i += dlen
                    let next = findDelim(from: i)
                    i = next == -1 ? n : next + dlen
                    continue
                }
                if s[i] == "{" { depth += 1 } else if s[i] == "}" { depth -= 1 }
                i += 1
            }
            let end = depth > 0 ? i : i - 1
            items.append(parseGemma4Args(String(s[objStart..<max(objStart, end)])))
        } else {
            let valStart = i
            while i < n, s[i] != ",", s[i] != "]" { i += 1 }
            let value = String(s[valStart..<i]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { items.append(value) }
        }
    }
    return items
}
