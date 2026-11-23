import Foundation
import MLXHuggingFace
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXCatNative

final class InklingTokenizerTests: XCTestCase {
    func testPublishedTokenizerKeepsFrameTokensAndUnicode() async throws {
        guard let path = ProcessInfo.processInfo.environment["MLXCAT_INKLING_TOKENIZER_DIRECTORY"] else {
            throw XCTSkip("Set the directory containing the published Inkling tokenizer assets.")
        }
        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: URL(fileURLWithPath: path, isDirectory: true))
        let frames = [
            "<|message_model|>", "<|content_thinking|>", "<|content_text|>",
            "<|content_invoke_tool_json|>", "<|end_message|>", "<|unused_200054|>",
        ]
        for frame in frames {
            let token = try XCTUnwrap(tokenizer.convertTokenToId(frame), frame)
            XCTAssertEqual(tokenizer.encode(text: frame, addSpecialTokens: false), [token])
            XCTAssertEqual(tokenizer.decode(tokenIds: [token]), frame)
        }
        XCTAssertEqual(tokenizer.eosTokenId, 200006)
        let prompt = try InklingChatTemplate.render(
            messages: [["role": "user", "content": "図書館は早く開きます。"]],
            tools: nil, reasoningEffort: "none")
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
        XCTAssertEqual(tokenizer.decode(tokenIds: tokens), prompt)
    }
}
