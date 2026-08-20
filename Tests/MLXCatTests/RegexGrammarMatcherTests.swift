import MLX
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

final class RegexGrammarMatcherTests: XCTestCase {
    func testRegexAllowsPrefixesAndGatesEOSUntilAcceptState() throws {
        let matcher = try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "[A-Z]{2}\\d?")
            .makeMatcher()

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("A")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("AB")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("a")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.eosID))

        matcher.advance(tokenID: Self.id("AB"))
        XCTAssertTrue(matcher.isComplete)
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("1")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("A")))
    }

    func testMaskSnapshotIsIsolatedFromMatcherAdvances() throws {
        let configuration = try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "[A-Z]{2}\\d?")
        let matcher = configuration.makeMatcher()
        let beforeAdvance = configuration.makeMatcher()
        let afterAdvance = configuration.makeMatcher()
        afterAdvance.advance(tokenID: Self.id("AB"))
        let snapshot = matcher.makeMaskSnapshot()

        matcher.advance(tokenID: Self.id("AB"))

        XCTAssertEqual(Set(snapshot.allowedTokenIDs()), Set(beforeAdvance.allowedTokenIDs()))
        XCTAssertEqual(Set(matcher.allowedTokenIDs()), Set(afterAdvance.allowedTokenIDs()))
    }

    func testRegexSupportsGroupsAlternationAndPlus() throws {
        let matcher = try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "(cat|dog)+")
            .makeMatcher()

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("cat")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("dog")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("ca")))

        matcher.advance(tokenID: Self.id("cat"))
        XCTAssertTrue(matcher.isComplete)
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("dog")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
    }

    func testRegexRejectsUnsupportedRange() {
        XCTAssertThrowsError(
            try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "[z-a]")
        ) { error in
            XCTAssertEqual(error as? RegexGrammarError, .unsupported("regex character class range is reversed"))
        }
    }

    func testRegexRejectsAnchorsBecauseDecodeIsImplicitlyAnchored() {
        assertUnsupportedRegex(
            pattern: "a$b",
            message: "regex anchors (^ and $) are unsupported because constrained decode is implicitly anchored"
        )
        assertUnsupportedRegex(
            pattern: "^ab",
            message: "regex anchors (^ and $) are unsupported because constrained decode is implicitly anchored"
        )
        assertUnsupportedRegex(
            pattern: "[^A]",
            message: "regex anchors (^ and $) are unsupported because constrained decode is implicitly anchored"
        )
    }

    func testRegexRejectsUnknownEscapes() {
        assertUnsupportedRegex(
            pattern: "\\D",
            message: "unsupported regex escape '\\D'"
        )
        assertUnsupportedRegex(
            pattern: "\\q",
            message: "unsupported regex escape '\\q'"
        )
    }

    func testRegexSupportedEscapesStillParse() throws {
        _ = try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "\\d\\w\\s\\.\\\\\\n\\t\\r")
        _ = try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "\\$\\^\\[\\]\\{\\}\\(\\)\\*\\+\\?\\|\\-")
    }

    func testRegexMaskConstrainsGreedyArgmaxWhenCandidateIsInvalid() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "[A-Z]{2}")
            .makeMatcher()
        var logits = [Float](repeating: -10, count: 40)
        logits[Self.id("a")] = 99
        logits[Self.id("AB")] = 5

        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0),
            regexGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), Self.id("AB"))
    }

    func testRegexFastPathKeepsValidGreedyCandidate() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "[A-Z]{2}")
            .makeMatcher()
        var logits = [Float](repeating: -10, count: 40)
        logits[Self.id("AB")] = 99

        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0),
            regexGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), Self.id("AB"))
    }

    func testRegexWithTruncationFiltersMasksBeforeSampling() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = try RegexGrammarConfiguration(tokens: Self.tokens, pattern: "[A-Z]{2}")
            .makeMatcher()
        var logits = [Float](repeating: -10, count: 40)
        logits[Self.id("a")] = 99
        logits[Self.id("AB")] = 5

        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0.7, topK: 1),
            regexGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), Self.id("AB"))
    }

    private static let eosID = 39

    private static let tokenTable: [(String, Int)] = [
        ("A", 1),
        ("B", 2),
        ("AB", 3),
        ("C", 4),
        ("1", 5),
        ("9", 6),
        ("a", 7),
        ("cat", 8),
        ("dog", 9),
        ("ca", 10),
    ]

    private static let tokens = tokenTable.map { text, id in
        JSONGrammarToken(id: id, text: text)
    } + [JSONGrammarToken(id: eosID, text: "", isEOS: true)]

    private static func id(_ text: String) -> Int {
        tokenTable.first { $0.0 == text }!.1
    }

    private func assertUnsupportedRegex(
        pattern: String,
        message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try RegexGrammarConfiguration(tokens: Self.tokens, pattern: pattern),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(error as? RegexGrammarError, .unsupported(message), file: file, line: line)
            let httpError = OpenAIServerError.invalidStructuredOutput(String(describing: error))
            XCTAssertEqual(httpError.httpStatus, 400, file: file, line: line)
            XCTAssertEqual(httpError.description, message, file: file, line: line)
        }
    }
}
