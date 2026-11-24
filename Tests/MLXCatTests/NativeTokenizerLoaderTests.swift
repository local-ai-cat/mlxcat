import Foundation
import MLXCatNative
import Tokenizers
import XCTest

final class NativeTokenizerLoaderTests: XCTestCase {
    func testQwen35AliasUsesTheSameBPEAndTemplate() async throws {
        let directory = try tokenizerDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeConfig("Qwen2Tokenizer", to: directory)
        let known = try await NativeTokenizerLoader().load(from: directory)
        try writeConfig("Qwen3_5Tokenizer", to: directory)
        let alias = try await NativeTokenizerLoader().load(from: directory)

        let messages: [[String: any Sendable]] = [["role": "user", "content": "ab"]]
        XCTAssertEqual(known.encode(text: "ab", addSpecialTokens: false), [2])
        XCTAssertEqual(alias.encode(text: "ab", addSpecialTokens: false), [2])
        XCTAssertEqual(alias.eosTokenId, 3)
        XCTAssertEqual(
            try alias.applyChatTemplate(messages: messages),
            try known.applyChatTemplate(messages: messages))
    }

    func testUnrecognizedClassStillFailsStrictValidation() async throws {
        let directory = try tokenizerDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let unknown = "UnrecognizedTestTokenizer"
        try writeConfig(unknown, to: directory)

        do {
            _ = try await NativeTokenizerLoader().load(from: directory)
            XCTFail("An unknown tokenizer class should remain unsupported.")
        } catch TokenizerError.unsupportedTokenizer(let name) {
            XCTAssertEqual(name, unknown)
        }
    }

    private func tokenizerDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = """
            {"version":"1.0",
             "model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"<|im_end|>":3},"merges":["a b"]},
             "added_tokens":[{"id":3,"content":"<|im_end|>","special":true,"single_word":false,
             "lstrip":false,"rstrip":false,"normalized":false}]}
            """
        try Data(data.utf8).write(to: directory.appending(component: "tokenizer.json"))
        return directory
    }

    private func writeConfig(_ name: String, to directory: URL) throws {
        let config = """
            {"tokenizer_class":"\(name)","eos_token":"<|im_end|>",
             "chat_template":"{{ messages[0]['content'] }}<|im_end|>"}
            """
        try Data(config.utf8).write(to: directory.appending(component: "tokenizer_config.json"))
    }
}
