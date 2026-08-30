import Foundation
import Tokenizers
@testable import MLXCat
import XCTest

final class QwenXMLToolBodyMatcherBenchmarkTests: XCTestCase {
    func testRealQwenVocabularyMaskLatency() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["MLXCAT_TOOL_GRAMMAR_BENCH_MODEL"],
            !modelPath.isEmpty
        else {
            throw XCTSkip("Set MLXCAT_TOOL_GRAMMAR_BENCH_MODEL to a Qwen model directory.")
        }

        let modelURL = URL(fileURLWithPath: modelPath)
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)
        let vocabulary = try Self.vocabulary(modelURL: modelURL, tokenizer: tokenizer)
        let startTokenID = try XCTUnwrap(tokenizer.convertTokenToId("<tool_call>"))
        let endTokenID = try XCTUnwrap(tokenizer.convertTokenToId("</tool_call>"))
        let eosTokenIDs = try Self.eosTokenIDs(modelURL: modelURL, tokenizer: tokenizer)
        let postToolAllowedTokenIDs = Set(
            vocabulary.tokens.compactMap { token -> Int? in
                guard !token.isEOS,
                    !token.text.isEmpty,
                    token.text.allSatisfy(QwenXMLToolGrammarConfiguration.isXMLWhitespace)
                else {
                    return nil
                }
                return token.id
            }
        )
        .union([startTokenID])
        .union(eosTokenIDs)

        let configuration = QwenXMLToolGrammarConfiguration(
            vocabulary: vocabulary,
            toolNames: ["bash"],
            toolCallStartTokenID: startTokenID,
            toolCallEndTokenID: endTokenID,
            postToolAllowedTokenIDs: Array(postToolAllowedTokenIDs)
        )

        XCTAssertEqual(startTokenID, 151657)
        XCTAssertEqual(endTokenID, 151658)

        for prefixLength in [0, 64, 256, 1024] {
            let matcher = configuration.makeMatcher()
            if prefixLength > 0 {
                try Self.advance(
                    matcher,
                    text: "<function=bash><parameter=command>" + String(repeating: "x", count: prefixLength),
                    tokenizer: tokenizer
                )
            }

            // Time the SAMPLING path (`tokenMask`), not the exhaustive reference
            // (`allowedTokenIDs`) — the reference is retained only as the
            // equivalence oracle and is O(vocab) by design.
            _ = matcher.tokenMask()
            let durations = (0 ..< 80).map { _ in
                let started = DispatchTime.now().uptimeNanoseconds
                _ = matcher.tokenMask()
                return DispatchTime.now().uptimeNanoseconds - started
            }

            // A latency number means nothing without knowing which path produced
            // it: an allow-shaped mask here would be a silent regression that
            // still "passes" any equivalence check.
            let mask = matcher.tokenMask()
            let universe = Set(vocabulary.tokens.map(\.id)).union([endTokenID])
            switch mask {
            case .deny(let denied):
                XCTAssertGreaterThan(prefixLength, 0, "prefix 0 is a restrictive state")
                print("TOOLGRAMMAR_MASK_SHAPE prefix=\(prefixLength) shape=deny size=\(denied.count)")
                XCTAssertLessThan(denied.count * 10, vocabulary.tokens.count)
                XCTAssertEqual(universe.subtracting(denied), Set(matcher.allowedTokenIDs()))
            case .allow(let allowed):
                XCTAssertEqual(prefixLength, 0, "a parameter body must be deny-shaped")
                print("TOOLGRAMMAR_MASK_SHAPE prefix=\(prefixLength) shape=allow size=\(allowed.count)")
                XCTAssertEqual(Set(allowed), Set(matcher.allowedTokenIDs()))
            }
            let sorted = durations.sorted()
            let p50 = Self.percentile(sorted, 0.50)
            let p99 = Self.percentile(sorted, 0.99)
            print(
                String(
                    format: "TOOLGRAMMAR_MASK_BENCH prefix=%d vocab=%d p50_ms=%.3f p99_ms=%.3f samples=%d",
                    prefixLength,
                    vocabulary.tokens.count,
                    Double(p50) / 1_000_000,
                    Double(p99) / 1_000_000,
                    sorted.count
                )
            )
            XCTAssertLessThan(
                Double(p99) / 1_000_000,
                Self.maskLatencyCeilingMilliseconds,
                "mask p99 exceeds the ceiling at prefix \(prefixLength)"
            )
        }

        try Self.measureAdversarialSuffixes(configuration: configuration, tokenizer: tokenizer, endTokenID: endTokenID, vocabulary: vocabulary)
    }

    /// A benign parameter body denies almost nothing, so the prefix rows above
    /// measure the complement mask at its easiest. The worst case is a suffix
    /// that is ALREADY a partial closing tag: every token that could complete it
    /// has to be denied, which is the only way the deny set can grow.
    private static func measureAdversarialSuffixes(
        configuration: QwenXMLToolGrammarConfiguration,
        tokenizer: any Tokenizer,
        endTokenID: Int,
        vocabulary: JSONGrammarVocabulary
    ) throws {
        let universe = Set(vocabulary.tokens.map(\.id)).union([endTokenID])
        for suffix in ["</functio", "</tool_cal", "</paramete"] {
            let matcher = configuration.makeMatcher()
            try advance(matcher, text: "<function=bash><parameter=command>", tokenizer: tokenizer)
            try advance(matcher, text: suffix, tokenizer: tokenizer)

            _ = matcher.tokenMask()
            let durations = (0 ..< 40).map { _ in
                let started = DispatchTime.now().uptimeNanoseconds
                _ = matcher.tokenMask()
                return DispatchTime.now().uptimeNanoseconds - started
            }
            let p99 = percentile(durations.sorted(), 0.99)
            guard case .deny(let denied) = matcher.tokenMask() else {
                return XCTFail("a parameter body must stay deny-shaped with suffix \(suffix)")
            }
            print(
                String(
                    format: "TOOLGRAMMAR_MASK_ADVERSARIAL suffix=%@ deny=%d p99_ms=%.3f",
                    suffix,
                    denied.count,
                    Double(p99) / 1_000_000
                )
            )
            XCTAssertEqual(
                universe.subtracting(denied),
                Set(matcher.allowedTokenIDs()),
                "deny mask diverges from the exhaustive reference at suffix \(suffix)"
            )
            XCTAssertLessThan(Double(p99) / 1_000_000, maskLatencyCeilingMilliseconds)
        }
    }

    private static let maskLatencyCeilingMilliseconds = 10.0

    private static func advance(
        _ matcher: QwenXMLToolBodyMatcher,
        text: String,
        tokenizer: any Tokenizer
    ) throws {
        let tokenIDs = tokenizer.encode(text: text, addSpecialTokens: false)
        XCTAssertFalse(tokenIDs.isEmpty)
        for tokenID in tokenIDs {
            XCTAssertTrue(matcher.accepts(tokenID: tokenID), "rejected token \(tokenID) while advancing \(text)")
            matcher.advance(tokenID: tokenID)
        }
    }

    private static func vocabulary(
        modelURL: URL,
        tokenizer: any Tokenizer
    ) throws -> JSONGrammarVocabulary {
        let vocabularySize = try modelConfigInt(modelURL: modelURL, key: "vocab_size")
        let eosTokenIDs = try eosTokenIDs(modelURL: modelURL, tokenizer: tokenizer)
        var tokens: [JSONGrammarToken] = []
        tokens.reserveCapacity(vocabularySize)

        for id in 0 ..< vocabularySize where tokenizer.convertIdToToken(id) != nil {
            if eosTokenIDs.contains(id) {
                tokens.append(JSONGrammarToken(id: id, text: "", isEOS: true))
                continue
            }
            let text = tokenizer.decode(tokens: [id], skipSpecialTokens: true)
            guard !text.isEmpty, !text.contains("\u{FFFD}") else {
                continue
            }
            tokens.append(JSONGrammarToken(id: id, text: text))
        }
        if tokens.allSatisfy({ $0.id != 151658 }) {
            tokens.append(JSONGrammarToken(id: 151658, text: "</tool_call>"))
        }
        return JSONGrammarVocabulary(tokens: tokens)
    }

    private static func eosTokenIDs(
        modelURL: URL,
        tokenizer: any Tokenizer
    ) throws -> Set<Int> {
        var ids: Set<Int> = []
        if let eosTokenID = tokenizer.eosTokenId {
            ids.insert(eosTokenID)
        }
        if let configEOS = try modelConfigValue(modelURL: modelURL, key: "eos_token_id") {
            if let id = configEOS as? Int {
                ids.insert(id)
            } else if let values = configEOS as? [Int] {
                ids.formUnion(values)
            }
        }
        return ids
    }

    private static func modelConfigInt(modelURL: URL, key: String) throws -> Int {
        guard let value = try modelConfigValue(modelURL: modelURL, key: key) as? Int else {
            throw NSError(domain: "QwenXMLToolBodyMatcherBenchmarkTests", code: 1)
        }
        return value
    }

    private static func modelConfigValue(modelURL: URL, key: String) throws -> Any? {
        let configURL = modelURL.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return object?[key]
    }

    private static func percentile(_ sorted: [UInt64], _ percentile: Double) -> UInt64 {
        let index = min(
            sorted.count - 1,
            max(0, Int(ceil(Double(sorted.count) * percentile)) - 1)
        )
        return sorted[index]
    }
}
