@testable import MLXCat
import XCTest

final class QwenXMLToolBodyMatcherTests: XCTestCase {
    func testAcceptsQwenXMLBodyWithEmbeddedQuotesAndSpaces() {
        let matcher = Self.configuration().makeMatcher()
        for token in [
            "\n",
            "<function=bash>",
            "\n<parameter=command>",
            #"find on-the-app-store -name "*.xcodeproj" -o -name "*.xcworkspace""#,
            "</parameter>",
            "\n</function>",
            "</tool_call>",
        ] {
            XCTAssertTrue(matcher.accepts(tokenID: Self.id(token)), "rejected \(token)")
            matcher.advance(tokenID: Self.id(token))
        }

        XCTAssertTrue(matcher.isComplete)
    }

    func testRejectsToolNamesOutsideRequestUnion() {
        let matcher = Self.configuration(toolNames: ["bash"]).makeMatcher()

        XCTAssertFalse(matcher.accepts(tokenID: Self.id("<function=python>")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("<function=bash>")))
    }

    func testParameterTextRejectsPrematureCloseTagsOnly() {
        let matcher = Self.configuration().makeMatcher()
        for token in ["<function=bash>", "<parameter=command>"] {
            XCTAssertTrue(matcher.accepts(tokenID: Self.id(token)))
            matcher.advance(tokenID: Self.id(token))
        }

        XCTAssertTrue(matcher.accepts(tokenID: Self.id(#"printf "<not-a-close>""#)))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("</function>")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("</tool_call>")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("</parameter>")))
    }

    func testMalformedHermesJSONFixtureIsCleanlyRefusedInsideToolCall() {
        let matcher = Self.configuration().makeMatcher()
        let malformedPayload = #"{"name": "bash", "arguments": {"command":"find on-the-app-store -name "*.xcodeproj" -o -name "*.xcworkspace""#

        XCTAssertFalse(matcher.accepts(tokenID: Self.id(malformedPayload)))
        XCTAssertFalse(matcher.allowedTokenIDs().contains(Self.id(malformedPayload)))
        XCTAssertTrue(matcher.allowedTokenIDs().contains(Self.id("<function=bash>")))
    }

    func testAllowedTokenIDsMatchesBruteForceAcceptanceAcrossStates() {
        let configuration = Self.configuration()
        let matcher = configuration.makeMatcher()

        for step in [nil, "<function=bash>", "<parameter=command>"] as [String?] {
            if let step {
                matcher.advance(tokenID: Self.id(step))
            }
            let bruteForce = Set(configuration.vocabulary.tokens.filter { matcher.accepts(tokenID: $0.id) }.map(\.id))
            XCTAssertEqual(Set(matcher.allowedTokenIDs()), bruteForce)
        }
    }

    func testRuntimeArmsOnTriggerTokenIDAndReleasesToPostToolAllowlist() {
        var runtime = QwenXMLToolGrammarState(configuration: Self.configuration())

        runtime.observeGeneratedToken(Self.id("ordinary"))
        XCTAssertFalse(runtime.requiresScalarSampling)
        XCTAssertNil(runtime.activeMatcher)

        runtime.observeGeneratedToken(Self.toolCallOpenID)
        XCTAssertTrue(runtime.requiresScalarSampling)
        XCTAssertNotNil(runtime.activeMatcher)

        for token in [
            "<function=bash>",
            "<parameter=command>",
            #"echo "hello world""#,
            "</parameter>",
            "</function>",
            "</tool_call>",
        ] {
            runtime.observeGeneratedToken(Self.id(token))
        }

        XCTAssertTrue(runtime.requiresScalarSampling)
        XCTAssertNil(runtime.activeMatcher)
        XCTAssertEqual(Set(runtime.activeAllowedTokenIDs ?? []), Set([Self.eosID, Self.toolCallOpenID, Self.id(" ")]))

        runtime.observeGeneratedToken(Self.toolCallOpenID)
        XCTAssertNotNil(runtime.activeMatcher)
    }

    func testConstrainedRegionBudgetDropsGrammarRatherThanRunningForever() {
        var runtime = QwenXMLToolGrammarState(
            configuration: Self.configuration(constrainedTokenBudget: 2)
        )

        runtime.observeGeneratedToken(Self.toolCallOpenID)
        runtime.observeGeneratedToken(Self.id("<function=bash>"))
        runtime.observeGeneratedToken(Self.id("<parameter=command>"))

        XCTAssertFalse(runtime.requiresScalarSampling)
        XCTAssertNil(runtime.activeMatcher)
        XCTAssertNil(runtime.activeAllowedTokenIDs)
    }

    private static let eosID = 0
    private static let toolCallOpenID = 100
    private static let toolCallCloseID = 101

    private static let tokenTexts = [
        "ordinary",
        " ",
        "\n",
        "<function=bash>",
        "<function=python>",
        "\n<parameter=command>",
        "<parameter=command>",
        #"find on-the-app-store -name "*.xcodeproj" -o -name "*.xcworkspace""#,
        #"printf "<not-a-close>""#,
        #"echo "hello world""#,
        "</parameter>",
        "\n</function>",
        "</function>",
        #"{"name": "bash", "arguments": {"command":"find on-the-app-store -name "*.xcodeproj" -o -name "*.xcworkspace""#,
    ]

    private static func configuration(
        toolNames: [String] = ["bash"],
        constrainedTokenBudget: Int = 4096
    ) -> QwenXMLToolGrammarConfiguration {
        QwenXMLToolGrammarConfiguration(
            vocabulary: JSONGrammarVocabulary(tokens: tokens),
            toolNames: toolNames,
            toolCallStartTokenID: toolCallOpenID,
            toolCallEndTokenID: toolCallCloseID,
            postToolAllowedTokenIDs: [eosID, toolCallOpenID, id(" ")],
            constrainedTokenBudget: constrainedTokenBudget
        )
    }

    private static var tokens: [JSONGrammarToken] {
        [JSONGrammarToken(id: eosID, text: "", isEOS: true)]
            + tokenTexts.enumerated().map { offset, text in
                JSONGrammarToken(id: offset + 1, text: text)
            }
            + [JSONGrammarToken(id: toolCallCloseID, text: "</tool_call>")]
    }

    private static func id(_ text: String) -> Int {
        tokens.first { $0.text == text }?.id ?? -1
    }
}
