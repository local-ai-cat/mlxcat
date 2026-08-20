import MLX
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

final class GBNFGrammarMatcherTests: XCTestCase {
    func testAllowedTokenIDsMatchesBruteForceAcceptanceAcrossStates() throws {
        let matcher = try Self.configuration().makeMatcher()

        for step in [nil, "12", "+", "3"] as [String?] {
            if let step {
                matcher.advance(tokenID: Self.id(step))
            }
            let bruteForce = Set(
                Self.tokens.filter { matcher.accepts(tokenID: $0.id) }.map(\.id)
            )
            XCTAssertEqual(Set(matcher.allowedTokenIDs()), bruteForce)
        }
        XCTAssertTrue(matcher.allowedTokenIDs().contains(Self.eosID))
    }

    func testMaskSnapshotIsIsolatedFromMatcherAdvances() throws {
        let configuration = try Self.configuration()
        let matcher = configuration.makeMatcher()
        let beforeAdvance = configuration.makeMatcher()
        let afterAdvance = configuration.makeMatcher()
        afterAdvance.advance(tokenID: Self.id("12"))
        let snapshot = matcher.makeMaskSnapshot()

        matcher.advance(tokenID: Self.id("12"))

        XCTAssertEqual(Set(snapshot.allowedTokenIDs()), Set(beforeAdvance.allowedTokenIDs()))
        XCTAssertEqual(Set(matcher.allowedTokenIDs()), Set(afterAdvance.allowedTokenIDs()))
    }

    func testArithmeticGrammarAllowsPrefixesAndGatesEOSUntilAcceptState() throws {
        let matcher = try Self.configuration().makeMatcher()

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("1")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("12")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("+")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("x")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.eosID))

        matcher.advance(tokenID: Self.id("12"))
        XCTAssertTrue(matcher.isComplete)
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("+")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))

        matcher.advance(tokenID: Self.id("+"))
        XCTAssertFalse(matcher.isComplete)
        XCTAssertFalse(matcher.accepts(tokenID: Self.eosID))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("3")))

        matcher.advance(tokenID: Self.id("3"))
        XCTAssertTrue(matcher.isComplete)
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
    }

    func testGrammarSupportsAlternationGroupingRepetitionAndComments() throws {
        let grammar = """
        root ::= ("cat" | "dog")+ # a comment
        """
        let matcher = try GBNFGrammarConfiguration(tokens: Self.tokens, grammar: grammar).makeMatcher()

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("cat")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("dog")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("ca")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("x")))

        matcher.advance(tokenID: Self.id("cat"))
        XCTAssertTrue(matcher.isComplete)
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("dog")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
    }

    func testGrammarSupportsHexAndUnicodeEscapes() throws {
        let grammar = """
        root ::= "\\u0041" [\\x30-\\x39] "\\U0000005A"
        """
        let matcher = try GBNFGrammarConfiguration(tokens: Self.tokens, grammar: grammar).makeMatcher()

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("A1Z")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("A1")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("x")))

        matcher.advance(tokenID: Self.id("A1Z"))
        XCTAssertTrue(matcher.isComplete)
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
    }

    func testGrammarRejectsMalformedRules() {
        assertUnsupportedGrammar(
            "number ::= [0-9]+",
            message: "GBNF grammar requires a root rule"
        )
        assertUnsupportedGrammar(
            "root ::= number\n",
            message: "GBNF rule 'root' references unknown rule 'number'"
        )
        assertUnsupportedGrammar(
            "root ::= [z-a]",
            message: "GBNF character class range is reversed"
        )
        assertUnsupportedGrammar(
            "root ::= root \"a\"",
            message: "left-recursive GBNF rule 'root' is unsupported"
        )
        assertUnsupportedGrammar(
            """
            root ::= a
            a ::= root "x"
            """,
            message: "left-recursive GBNF rule 'root' is unsupported"
        )
        assertUnsupportedGrammar(
            """
            root ::= prefix a
            prefix ::= "" | "p"
            a ::= root "x"
            """,
            message: "left-recursive GBNF rule 'root' is unsupported"
        )
        assertUnsupportedGrammar(
            "root ::= \"abc",
            message: "unclosed GBNF string literal"
        )
    }

    func testGrammarMaskConstrainsGreedyArgmaxWhenCandidateIsInvalid() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = try Self.configuration().makeMatcher()
        var logits = [Float](repeating: -10, count: Self.logitCount)
        logits[Self.id("x")] = 99
        logits[Self.id("12")] = 5

        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0),
            gbnfGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), Self.id("12"))
    }

    func testGrammarWithTruncationFiltersMasksBeforeSampling() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = try Self.configuration().makeMatcher()
        var logits = [Float](repeating: -10, count: Self.logitCount)
        logits[Self.id("x")] = 99
        logits[Self.id("12")] = 5

        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0.7, topK: 1),
            gbnfGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), Self.id("12"))
    }

    func testArithmeticGrammarConstrainedDecodeEndToEnd() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = try Self.configuration().makeMatcher()
        let forcedValidPath = ["12", "+", "3", Self.eosText]
        var decoded = ""
        var generated: [Int] = []

        for validText in forcedValidPath {
            var logits = [Float](repeating: -10, count: Self.logitCount)
            logits[Self.id("x")] = 99
            logits[Self.id(validText)] = 5
            let sampled = TokenSampler.sample(
                logits: MLXArray(logits),
                parameters: SamplingParameters(temperature: 0),
                generatedTokens: generated,
                gbnfGrammarMatcher: matcher
            )
            let tokenID = sampled.item(Int.self)

            XCTAssertEqual(tokenID, Self.id(validText))
            XCTAssertTrue(matcher.accepts(tokenID: tokenID))
            matcher.advance(tokenID: tokenID)
            generated.append(tokenID)
            if validText != Self.eosText {
                decoded += validText
            }
        }

        XCTAssertEqual(decoded, "12+3")
        XCTAssertTrue(matcher.isComplete)
    }

    private static let grammar = """
    root ::= number (operator number)*
    number ::= [0-9]+
    operator ::= [+-]
    """

    private static let eosID = 49
    private static let eosText = "<eos>"
    private static let logitCount = 60

    private static let tokenTable: [(String, Int)] = [
        ("1", 1),
        ("12", 2),
        ("3", 3),
        ("+", 4),
        ("-", 5),
        ("x", 6),
        ("cat", 7),
        ("dog", 8),
        ("ca", 9),
        ("A1", 10),
        ("A1Z", 11),
        (eosText, eosID),
    ]

    private static let tokens = tokenTable.map { text, id in
        text == eosText
            ? JSONGrammarToken(id: id, text: "", isEOS: true)
            : JSONGrammarToken(id: id, text: text)
    }

    private static func configuration() throws -> GBNFGrammarConfiguration {
        try GBNFGrammarConfiguration(tokens: tokens, grammar: grammar)
    }

    private static func id(_ text: String) -> Int {
        tokenTable.first { $0.0 == text }!.1
    }

    private func assertUnsupportedGrammar(
        _ grammar: String,
        message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try GBNFGrammarConfiguration(tokens: Self.tokens, grammar: grammar),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(error as? GBNFGrammarError, .unsupported(message), file: file, line: line)
            let httpError = OpenAIServerError.invalidStructuredOutput(String(describing: error))
            XCTAssertEqual(httpError.httpStatus, 400, file: file, line: line)
            XCTAssertEqual(httpError.description, message, file: file, line: line)
        }
    }
}
