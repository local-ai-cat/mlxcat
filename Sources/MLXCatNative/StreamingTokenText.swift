import Foundation
import MLXLMCommon

/// Turns the engine's per-token stream into text without cutting a character in
/// half.
///
/// Byte-level BPE spreads one emoji over two or three tokens, and decoding each
/// token on its own turned every fragment into U+FFFD — a Qwen3 reply ended in
/// "Hello! ��" (Local AI Cat, 2026-09-07). The detokenizer holds a fragment
/// until the character is whole, so a mid-character token yields empty text and
/// the closing token yields the character. Token ids still go out one per
/// token; only the text is deferred.
package struct StreamingTokenText {
    private var detokenizer: NaiveStreamingDetokenizer

    package init(tokenizer: any Tokenizer) {
        detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    }

    /// The text this token completes: empty while a multi-byte character is
    /// still open, the whole character (and anything after it) once it closes.
    package mutating func text(for token: Int) -> String {
        detokenizer.append(token: token)
        return detokenizer.next() ?? ""
    }
}
