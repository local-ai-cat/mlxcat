import Foundation
import MLXLMCommon
import XCTest
@testable import MLXCatNative

/// A byte-level tokenizer in miniature: every id is a run of UTF-8 bytes, and
/// decoding joins the runs before reading them as UTF-8 — which is what makes a
/// fragment on its own come out as U+FFFD.
private struct ByteRunTokenizer: Tokenizer {
    let runs: [Int: [UInt8]]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.flatMap { runs[$0] ?? [] }, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

final class StreamingTokenTextTests: XCTestCase {
    /// "Hello! 😊 Bye 🚀" the way Qwen3 tokenizes it: the smiley is two tokens,
    /// the rocket three.
    private let tokenizer = ByteRunTokenizer(runs: [
        1: Array("Hello".utf8),
        2: Array("!".utf8),
        3: [0x20, 0xF0, 0x9F, 0x98],
        4: [0x8A],
        5: Array(" Bye".utf8),
        6: [0x20, 0xF0, 0x9F],
        7: [0x9A],
        8: [0x80],
    ])

    func testFragmentsAreHeldUntilTheCharacterIsWhole() {
        var text = StreamingTokenText(tokenizer: tokenizer)

        let pieces = [1, 2, 3, 4, 5, 6, 7, 8].map { text.text(for: $0) }

        XCTAssertEqual(pieces, ["Hello", "!", "", " 😊", " Bye", "", "", " 🚀"])
        XCTAssertFalse(pieces.joined().contains("\u{FFFD}"))
    }

    func testDecodingEachTokenAloneIsWhatWentWrong() {
        // The behavior this type replaces, pinned so the regression is legible.
        let alone = tokenizer.decode(tokenIds: [3], skipSpecialTokens: false)
        XCTAssertTrue(alone.contains("\u{FFFD}"))
    }
}
