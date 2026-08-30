@testable import MLXCat
import XCTest

/// The deny-shaped mask exists to avoid enumerating ~150k admissions in a
/// permissive state. That is only safe if it is EXACTLY equivalent to the
/// exhaustive allow-list it replaces, so these tests use
/// `QwenXMLToolBodyMaskSnapshot.allowedTokenIDs` — the untouched reference
/// implementation — as an oracle, over an adversarial vocabulary built to break
/// the "tokens containing >" candidate filter.
///
/// These run with no model and no environment gate: the real-vocabulary version
/// in `QwenXMLToolBodyMaskLatencyTests` can only run where the weights are, and
/// a gated-only equivalence check would skip forever in CI.
final class QwenXMLToolMaskEquivalenceTests: XCTestCase {
    func testMaskMatchesExhaustiveReferenceAcrossEveryState() {
        let universe = Self.universe
        for state in Self.states {
            let snapshot = Self.snapshot(state: state)
            let reference = Set(snapshot.allowedTokenIDs())
            let mask = try? XCTUnwrap(snapshot.tokenMask(isCancelled: { false }))
            guard let mask else { return XCTFail("no mask for \(state)") }
            XCTAssertEqual(
                Self.resolve(mask, universe: universe),
                reference,
                "mask disagrees with the exhaustive reference in state \(state)"
            )
        }
    }

    /// Equivalence alone cannot catch a silent regression to the slow path — an
    /// allow-shaped mask in a permissive state would still be *correct*. Pin the
    /// shape so the optimisation cannot quietly stop being taken.
    func testPermissiveStatesPublishDenyShapeAndRestrictiveStatesPublishAllowShape() {
        for state in Self.states {
            let mask = Self.snapshot(state: state).tokenMask(isCancelled: { false })
            switch (state, mask) {
            case (.parameterText, .deny), (.parameterName, .deny):
                break
            case (.parameterText, _), (.parameterName, _):
                XCTFail("permissive state \(state) must publish a deny-shaped mask")
            case (_, .allow):
                break
            default:
                XCTFail("restrictive state \(state) must publish an allow-shaped mask")
            }
        }
    }

    /// The point of the change: the complement is small. If a future edit makes
    /// the deny set proportional to the vocabulary, the latency regression comes
    /// straight back and this fails before anyone re-runs the benchmark.
    func testDenySetIsASmallFractionOfTheVocabulary() {
        guard case .deny(let denied)? = Self.snapshot(state: .parameterText(suffix: "")).tokenMask(isCancelled: { false })
        else {
            return XCTFail("expected a deny-shaped mask in a parameter body")
        }
        XCTAssertLessThan(denied.count * 4, Self.universe.count, "deny set is no longer a small complement")
    }

    func testEOSAndEmptyTextTokensAreDeniedInsideAParameterBody() {
        guard case .deny(let denied)? = Self.snapshot(state: .parameterText(suffix: "")).tokenMask(isCancelled: { false })
        else {
            return XCTFail("expected a deny-shaped mask in a parameter body")
        }
        // Both are rejected by the DFA but contain no ">", so neither is a
        // candidate — they have to be denied explicitly.
        XCTAssertTrue(denied.contains(Self.eosID), "EOS must not be samplable mid-tool-call")
        XCTAssertTrue(denied.contains(Self.emptyTextID), "an empty-text token must not be samplable")
    }

    /// A closing tag split across the state suffix and the token is the case the
    /// candidate filter has to get right: the ">" arrives in the token, the rest
    /// of the tag is already in the suffix.
    func testClosingTagsSplitAcrossSuffixAndTokenAreDenied() {
        let snapshot = Self.snapshot(state: .parameterText(suffix: "</functio"))
        guard case .deny(let denied)? = snapshot.tokenMask(isCancelled: { false }) else {
            return XCTFail("expected a deny-shaped mask")
        }
        XCTAssertTrue(denied.contains(Self.id("n>")), "</functio + n> completes a premature close tag")
        XCTAssertFalse(denied.contains(Self.id("nal>")), "</functio + nal> is ordinary parameter text")
    }

    /// `</parameter></function>` leaves the body legally and then closes the
    /// function — it contains a closing tag but must still be accepted. Proves
    /// the deny path simulates candidates rather than pattern-matching them.
    func testTokenLeavingTheBodyBeforeClosingTheFunctionIsAllowed() {
        let snapshot = Self.snapshot(state: .parameterText(suffix: ""))
        guard case .deny(let denied)? = snapshot.tokenMask(isCancelled: { false }) else {
            return XCTFail("expected a deny-shaped mask")
        }
        XCTAssertFalse(denied.contains(Self.id("</parameter></function>")))
        XCTAssertTrue(denied.contains(Self.id("</function>")))
    }

    /// Wiring check: the shapes above must reach the sampler through the real
    /// matcher, not only through a directly-constructed snapshot.
    func testMatcherPublishesDenyShapeAfterAdvancingIntoAParameterBody() {
        let configuration = Self.configuration
        let matcher = configuration.makeMatcher()
        if case .deny = matcher.tokenMask() {
            XCTFail("the initial state is restrictive and must publish an allow-shaped mask")
        }
        for text in ["<function=bash>", "<parameter=command>"] {
            matcher.advance(tokenID: Self.id(text))
        }
        guard case .deny = matcher.tokenMask() else {
            return XCTFail("a parameter body must publish a deny-shaped mask")
        }
        XCTAssertEqual(
            Self.resolve(matcher.tokenMask(), universe: Self.universe),
            Set(matcher.allowedTokenIDs())
        )
    }

    // MARK: - Fixtures

    private static let eosID = 0
    private static let emptyTextID = 1
    private static let toolCallOpenID = 900
    private static let toolCallCloseID = 901

    /// Deliberately hostile: every partial closing tag, a bare ">", tokens that
    /// straddle a tag boundary, and a token that closes the body and the
    /// function in one step.
    private static let tokenTexts = [
        "ordinary", " ", "\n", "x",
        "<", "</", "</f", "</functio", "n>", "nal>", ">", "=>", "a>b",
        "<function=bash>", "<function=python>", "<parameter=command>", "<parameter=",
        "</parameter>", "</parameter", "</function>", "</tool_call>",
        "</parameter></function>", "</parameter></function></tool_call>",
        #"printf "<not-a-close>""#,
        #"echo "hello world""#,
    ]

    private static let states: [QwenXMLToolBodyState] = [
        .beforeFunction(prefix: ""),
        .beforeFunction(prefix: "<f"),
        .functionBody,
        .bodyTag(prefix: "<"),
        .bodyTag(prefix: "</"),
        .parameterName(""),
        .parameterName("comm"),
        .parameterText(suffix: ""),
        .parameterText(suffix: "x"),
        .parameterText(suffix: "</"),
        .parameterText(suffix: "</functio"),
        .parameterText(suffix: "</tool_cal"),
        .parameterText(suffix: "</paramete"),
        .afterFunction(prefix: ""),
        .afterFunction(prefix: "</tool_cal"),
    ]

    private static var tokens: [JSONGrammarToken] {
        [
            JSONGrammarToken(id: eosID, text: "", isEOS: true),
            JSONGrammarToken(id: emptyTextID, text: ""),
        ]
            + tokenTexts.enumerated().map { offset, text in
                JSONGrammarToken(id: offset + 2, text: text)
            }
            + [JSONGrammarToken(id: toolCallCloseID, text: "</tool_call>")]
    }

    private static var configuration: QwenXMLToolGrammarConfiguration {
        QwenXMLToolGrammarConfiguration(
            vocabulary: JSONGrammarVocabulary(tokens: tokens),
            toolNames: ["bash"],
            toolCallStartTokenID: toolCallOpenID,
            toolCallEndTokenID: toolCallCloseID,
            postToolAllowedTokenIDs: [eosID, toolCallOpenID]
        )
    }

    private static var universe: Set<Int> {
        Set(tokens.map(\.id)).union([toolCallCloseID])
    }

    private static func snapshot(state: QwenXMLToolBodyState) -> QwenXMLToolBodyMaskSnapshot {
        let vocabulary = JSONGrammarVocabulary(tokens: tokens)
        return QwenXMLToolBodyMaskSnapshot(
            vocabulary: vocabulary,
            denyIndex: QwenXMLToolBodyDenyIndex(vocabulary: vocabulary),
            toolCallEndTokenID: toolCallCloseID,
            functionOpenTags: ["<function=bash>"],
            state: state
        )
    }

    private static func resolve(_ mask: GrammarTokenMask, universe: Set<Int>) -> Set<Int> {
        switch mask {
        case .allow(let ids):
            return Set(ids)
        case .deny(let ids):
            return universe.subtracting(ids)
        }
    }

    private static func id(_ text: String) -> Int {
        tokens.first { $0.text == text && !$0.isEOS }?.id ?? -1
    }
}
