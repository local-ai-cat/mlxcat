@testable import MLXCat
import XCTest

/// The three-layer resolution order is the contract: operator kill switch beats
/// client request beats server default. Each test names the layer it pins.
final class QwenXMLToolGrammarPolicyTests: XCTestCase {
    func testBuiltInDefaultAppliesWhenNothingElseSpeaks() {
        XCTAssertEqual(
            QwenXMLToolGrammarPolicy.isEnabled(requestOverride: nil, environmentValue: nil),
            QwenXMLToolGrammarPolicy.defaultEnabled
        )
        XCTAssertFalse(QwenXMLToolGrammarPolicy.isEnabled(
            requestOverride: nil, environmentValue: nil, defaultEnabled: false
        ))
        XCTAssertTrue(QwenXMLToolGrammarPolicy.isEnabled(
            requestOverride: nil, environmentValue: nil, defaultEnabled: true
        ))
    }

    func testRequestOverrideBeatsTheDefaultInBothDirections() {
        XCTAssertTrue(QwenXMLToolGrammarPolicy.isEnabled(
            requestOverride: true, environmentValue: nil, defaultEnabled: false
        ))
        XCTAssertFalse(QwenXMLToolGrammarPolicy.isEnabled(
            requestOverride: false, environmentValue: nil, defaultEnabled: true
        ))
    }

    func testEnvironmentOnActsAsServerDefaultNotAMandate() {
        for value in ["1", "on", "true", "ON", "True"] {
            XCTAssertTrue(QwenXMLToolGrammarPolicy.isEnabled(
                requestOverride: nil, environmentValue: value, defaultEnabled: false
            ), value)
            // A client saying no still wins over env-on: the client may know the
            // grammar conflicts with what it is doing.
            XCTAssertFalse(QwenXMLToolGrammarPolicy.isEnabled(
                requestOverride: false, environmentValue: value, defaultEnabled: false
            ), value)
        }
    }

    func testEnvironmentOffIsAKillSwitchThatBeatsEveryone() {
        for value in ["0", "off", "false", "OFF", "False"] {
            XCTAssertFalse(QwenXMLToolGrammarPolicy.isEnabled(
                requestOverride: true, environmentValue: value, defaultEnabled: true
            ), value)
        }
    }

    func testUnrecognisedEnvironmentValueFallsThroughToTheDefault() {
        XCTAssertTrue(QwenXMLToolGrammarPolicy.isEnabled(
            requestOverride: nil, environmentValue: "banana", defaultEnabled: true
        ))
        XCTAssertFalse(QwenXMLToolGrammarPolicy.isEnabled(
            requestOverride: nil, environmentValue: "banana", defaultEnabled: false
        ))
    }

    /// Today's shipped posture, spelled out so a default flip is a deliberate
    /// edit to BOTH the policy and this line.
    func testShippedDefaultIsOff() {
        XCTAssertFalse(QwenXMLToolGrammarPolicy.defaultEnabled)
    }
}
