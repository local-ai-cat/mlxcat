import XCTest
@testable import MLXCatNative

final class ReasoningEffortCompatibilityTests: XCTestCase {
    func testQwenAliasesRetryWithCompatibleValuesThenNativeDefault() {
        XCTAssertEqual(
            ReasoningEffortCompatibility.candidates(for: .string("high"), isHarmony: false),
            [.value(.string("high")), .value(.string("xhigh")), .omitted]
        )
        XCTAssertEqual(
            ReasoningEffortCompatibility.candidates(for: .string(" MINIMAL "), isHarmony: false),
            [.value(.string("minimal")), .value(.string("low")), .omitted]
        )
        XCTAssertEqual(
            ReasoningEffortCompatibility.candidates(for: .string("maximum"), isHarmony: false),
            [.value(.string("maximum")), .value(.string("max")), .omitted]
        )
    }

    func testNumericStringRetriesAsNumber() {
        XCTAssertEqual(
            ReasoningEffortCompatibility.candidates(for: .string("0.5"), isHarmony: false),
            [.value(.string("0.5")), .value(.number(0.5)), .omitted]
        )
    }

    func testUnknownAndNonStringValuesFallBackToNativeDefault() {
        XCTAssertEqual(
            ReasoningEffortCompatibility.candidates(for: .string("bogus"), isHarmony: false),
            [.value(.string("bogus")), .omitted]
        )
        XCTAssertEqual(
            ReasoningEffortCompatibility.candidates(for: .number(1), isHarmony: false),
            [.value(.number(1)), .omitted]
        )
    }

    func testHarmonyAliasesAreNormalizedWithoutInvalidRetries() {
        XCTAssertEqual(
            ReasoningEffortCompatibility.candidates(for: .string("xhigh"), isHarmony: true),
            [.value(.string("high"))]
        )
        XCTAssertEqual(
            ReasoningEffortCompatibility.candidates(for: .string("bogus"), isHarmony: true),
            [.omitted]
        )
    }
}
