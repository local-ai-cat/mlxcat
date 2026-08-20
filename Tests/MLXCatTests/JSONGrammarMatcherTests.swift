import MLX
@testable import MLXCat
import XCTest

final class JSONGrammarMatcherTests: XCTestCase {
    func testAllowedTokenIDsMatchesBruteForceAcceptanceAcrossStates() {
        let configuration = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
        let matcher = configuration.makeMatcher()

        // Walk through empty → open → complete states; at each state the bucket-pruned
        // allowed set must equal a brute-force accepts() sweep over the whole vocabulary.
        for step in [nil, "{", "}"] as [String?] {
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

    func testMaskSnapshotIsIsolatedFromMatcherAdvances() {
        let configuration = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
        let matcher = configuration.makeMatcher()
        let beforeAdvance = configuration.makeMatcher()
        let afterAdvance = configuration.makeMatcher()
        afterAdvance.advance(tokenID: Self.id("{"))
        let snapshot = matcher.makeMaskSnapshot()

        matcher.advance(tokenID: Self.id("{"))

        XCTAssertEqual(Set(snapshot.allowedTokenIDs()), Set(beforeAdvance.allowedTokenIDs()))
        XCTAssertEqual(Set(matcher.allowedTokenIDs()), Set(afterAdvance.allowedTokenIDs()))
    }

    func testTruncatedUnicodeEscapeIsIncompleteNotInvalid() {
        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
            .makeMatcher()
        for token in ["{", "\"name\"", ":"] {
            matcher.advance(tokenID: Self.id(token))
        }

        // Token ends mid-escape: `"\` — must be a legal prefix state.
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"\\")))
        matcher.advance(tokenID: Self.id("\"\\"))

        // Continuing the escape (`u0041"`) must stay reachable — and, critically, must
        // survive bucket pruning: the one-character probe `…"\u` is an incomplete escape,
        // not an invalid one.
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("u0041\"")))
        XCTAssertTrue(matcher.allowedTokenIDs().contains(Self.id("u0041\"")))

        matcher.advance(tokenID: Self.id("u0041\""))
        matcher.advance(tokenID: Self.id("}"))
        XCTAssertTrue(matcher.isComplete)
    }

    func testSurrogatePairEscapesSpanTokensAndSurviveBucketPruning() {
        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
            .makeMatcher()
        for token in ["{", "\"name\"", ":"] {
            matcher.advance(tokenID: Self.id(token))
        }

        // `"\uD83D` — a complete high surrogate awaiting its low half — is a legal prefix.
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"\\uD83D")))
        matcher.advance(tokenID: Self.id("\"\\uD83D"))

        // The continuation `\uDE00"` (low surrogate + close quote) must be reachable and
        // must survive bucket pruning (probe `…\uD83D\` is incomplete, not invalid).
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\\uDE00\"")))
        XCTAssertTrue(matcher.allowedTokenIDs().contains(Self.id("\\uDE00\"")))

        // A lone low surrogate is never a valid escape.
        matcher.advance(tokenID: Self.id("\\uDE00\""))
        matcher.advance(tokenID: Self.id("}"))
        XCTAssertTrue(matcher.isComplete)
    }

    func testGrammarWithTruncationFiltersMasksBeforeSampling() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
            .makeMatcher()
        // topK=1 with an invalid argmax: the fast path must NOT run (filters active);
        // masking first leaves "{" as the sole survivor of the truncation.
        var logits = [Float](repeating: -10, count: 20)
        logits[3] = 99
        logits[1] = 5
        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0.7, topK: 1),
            jsonGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), 1)
    }

    func testGrammarMaskConstrainsGreedyArgmaxWhenCandidateIsInvalid() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
            .makeMatcher()
        // Highest logit is "[" (id 3) — invalid for a top-level object; the rejection
        // path must fall back to the mask and pick "{" (id 1), the valid runner-up.
        var logits = [Float](repeating: -10, count: 20)
        logits[3] = 99
        logits[1] = 5
        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0),
            jsonGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), 1)
    }

    func testGrammarFastPathKeepsValidGreedyCandidate() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
            .makeMatcher()
        var logits = [Float](repeating: -10, count: 20)
        logits[1] = 99
        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0),
            jsonGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), 1)
    }

    func testThinkingBudgetForcedTokenDefersToJSONGrammarMask() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
            .makeMatcher()
        var state: ThinkingBudgetState? = ThinkingBudgetState(
            configuration: ThinkingBudgetConfiguration(
                budget: 0,
                closeTokenIDs: [Self.id("}")],
                startsInThinking: true
            )
        )
        var logits = [Float](repeating: -10, count: 24)
        logits[Self.id("[")] = 99
        logits[Self.id("{")] = 5

        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0),
            jsonGrammarMatcher: matcher,
            thinkingBudgetState: &state
        )
        let tokenID = sampled.item(Int.self)

        XCTAssertEqual(tokenID, Self.id("{"))
        XCTAssertTrue(matcher.accepts(tokenID: tokenID))
        matcher.advance(tokenID: tokenID)
        state?.advance(tokenID: tokenID)
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("{")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("}")))
    }

    func testJSONObjectAllowsOnlyObjectPrefixesAndGatesEOS() {
        let matcher = JSONGrammarConfiguration(
            tokens: Self.tokens,
            schema: .jsonObject
        ).makeMatcher()

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("{")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("[")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.eosID))

        matcher.advance(tokenID: Self.id("{"))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("}")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.eosID))

        matcher.advance(tokenID: Self.id("}"))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id(" ")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id(",")))
    }

    func testSchemaRequiresKeysAndConstrainsTypesAndEnums() {
        let schema = JSONSchemaNode.schema(
            JSONSchema(
                allowedTypes: [.object],
                properties: [
                    "status": .schema(
                        JSONSchema(
                            allowedTypes: [.string],
                            enumValues: [.string("done"), .string("fail")]
                        )
                    )
                ],
                required: ["status"],
                additionalPropertiesAllowed: false
            )
        )
        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: schema).makeMatcher()

        matcher.advance(tokenID: Self.id("{"))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"status\"")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("\"age\"")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("}")))

        matcher.advance(tokenID: Self.id("\"status\""))
        matcher.advance(tokenID: Self.id(":"))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"done\"")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"fail\"")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("\"blue\"")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("1")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("true")))

        matcher.advance(tokenID: Self.id("\"done\""))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("}")))
    }

    func testTokenCanSpanStringCloseAndComma() {
        let schema = JSONSchemaNode.schema(
            JSONSchema(
                allowedTypes: [.object],
                properties: [
                    "name": .schema(JSONSchema(allowedTypes: [.string])),
                    "age": .schema(JSONSchema(allowedTypes: [.number])),
                ],
                required: ["name", "age"],
                additionalPropertiesAllowed: false
            )
        )
        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: schema).makeMatcher()

        for token in ["{", "\"name\"", ":"] {
            matcher.advance(tokenID: Self.id(token))
        }

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"x\",")))
        matcher.advance(tokenID: Self.id("\"x\","))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"age\"")))

        matcher.advance(tokenID: Self.id("\"age\""))
        matcher.advance(tokenID: Self.id(":"))
        matcher.advance(tokenID: Self.id("1"))
        matcher.advance(tokenID: Self.id("}"))
        XCTAssertTrue(matcher.isComplete)
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
    }

    func testSchemaArraysAndNullsAreAcceptedWhenDeclared() {
        let schema = JSONSchemaNode.schema(
            JSONSchema(
                allowedTypes: [.object],
                properties: [
                    "items": .schema(
                        JSONSchema(
                            allowedTypes: [.array],
                            items: .schema(JSONSchema(allowedTypes: [.null]))
                        )
                    )
                ],
                required: ["items"],
                additionalPropertiesAllowed: false
            )
        )
        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: schema).makeMatcher()

        for token in ["{", "\"items\"", ":", "[", "null", "]", "}"] {
            XCTAssertTrue(matcher.accepts(tokenID: Self.id(token)), "expected token \(token) to be accepted")
            matcher.advance(tokenID: Self.id(token))
        }
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
    }

    private static let eosID = 999

    private static let tokenTable: [(String, Int)] = [
        ("{", 1),
        ("}", 2),
        ("[", 3),
        ("]", 4),
        (":", 5),
        (",", 6),
        (" ", 7),
        ("\"name\"", 8),
        ("\"age\"", 9),
        ("\"status\"", 10),
        ("\"items\"", 11),
        ("\"done\"", 12),
        ("\"fail\"", 13),
        ("\"blue\"", 14),
        ("\"x\",", 15),
        ("1", 16),
        ("true", 17),
        ("null", 18),
        ("\"\\", 19),
        ("u0041\"", 20),
        ("\"\\uD83D", 21),
        ("\\uDE00\"", 22),
    ]

    private static let tokens = tokenTable.map { text, id in
        JSONGrammarToken(id: id, text: text)
    } + [
        JSONGrammarToken(id: eosID, text: "", isEOS: true)
    ]

    private static func id(_ token: String) -> Int {
        guard let match = tokenTable.first(where: { $0.0 == token }) else {
            XCTFail("unknown token \(token)")
            return -1
        }
        return match.1
    }
}
