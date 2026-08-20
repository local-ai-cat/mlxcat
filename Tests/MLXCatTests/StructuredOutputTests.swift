import Foundation
import MLX
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

final class StructuredOutputTests: XCTestCase {
    func testChatResponseFormatJSONObjectsParseToStructuredOutputSpec() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "response_format": {"type": "json_object"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .jsonObject)
    }

    func testChatResponseFormatJSONSchemaParsesToStructuredOutputSpec() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                      "name": "answer",
                      "schema": {
                        "type": "object",
                        "properties": {
                          "answer": {"type": "string"}
                        },
                        "required": ["answer"]
                      }
                    }
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(
            request.structuredOutput,
            .jsonSchema(
                name: "answer",
                schema: [
                    "type": .string("object"),
                    "properties": .object([
                        "answer": .object(["type": .string("string")])
                    ]),
                    "required": .array([.string("answer")]),
                ]
            )
        )
    }

    func testChatResponseFormatTextIsNoOp() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "response_format": {"type": "text"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .none)
    }

    func testStructuredOutputsChoiceParses() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {
                    "type": "choice",
                    "choices": ["yes", "no"]
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .choice(["yes", "no"]))
    }

    func testStructuredOutputsPrecedeResponseFormat() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "response_format": {"type": "json_object"},
                  "structured_outputs": {
                    "type": "choice",
                    "choices": ["yes", "no"]
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .choice(["yes", "no"]))
    }

    func testStructuredOutputsJSONMapsToJSONObjectFallback() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {"type": "json"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .jsonObject)
    }

    func testStructuredOutputsEmptyChoiceThrowsBadRequest() {
        XCTAssertThrowsError(
            try OpenAIChatRequest.parse(
                Data(
                    """
                    {
                      "model": "test-model",
                      "messages": [{"role": "user", "content": "hello"}],
                      "structured_outputs": {
                        "type": "choice",
                        "choices": []
                      }
                    }
                    """.utf8
                )
            )
        ) { error in
            XCTAssertEqual((error as? OpenAIServerError)?.httpStatus, 400)
        }
    }

    func testGuidedGrammarParses() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "guided_grammar": "root ::= \\"yes\\""
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .grammar("root ::= \"yes\""))
    }

    func testStructuredOutputsGrammarWithoutGrammarFieldThrowsInvalidJSON() {
        XCTAssertThrowsError(
            try OpenAIChatRequest.parse(
                Data(
                    """
                    {
                      "model": "test-model",
                      "messages": [{"role": "user", "content": "hello"}],
                      "structured_outputs": {"type": "grammar"}
                    }
                    """.utf8
                )
            )
        ) { error in
            XCTAssertEqual((error as? OpenAIServerError)?.httpStatus, 422)
        }
    }

    func testStructuredOutputsGrammarParsesGrammarField() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {
                    "type": "grammar",
                    "grammar": "root ::= [0-9]+"
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .grammar("root ::= [0-9]+"))
    }

    func testStructuredOutputsGrammarParsesGBNFField() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {
                    "type": "grammar",
                    "gbnf": "root ::= \\"ok\\""
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .grammar("root ::= \"ok\""))
    }

    func testStructuredOutputsRegexParsesPattern() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {"type": "regex", "pattern": "[A-Z]{2}"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .regex(pattern: "[A-Z]{2}"))
    }

    func testStructuredOutputsParsesOMLXRegexAlias() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {"regex": "[A-Z]{2}"}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .regex(pattern: "[A-Z]{2}"))
    }

    func testStructuredOutputsParsesOMLXGrammarAlias() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {"grammar": "root ::= \\"OK\\""}
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .grammar("root ::= \"OK\""))
    }

    func testStructuredOutputsParsesOMLXJsonAlias() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "structured_outputs": {
                    "json": {
                      "type": "object",
                      "properties": {"answer": {"type": "string"}},
                      "required": ["answer"]
                    }
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(
            request.structuredOutput,
            .jsonSchema(
                name: nil,
                schema: [
                    "type": .string("object"),
                    "properties": .object(["answer": .object(["type": .string("string")])]),
                    "required": .array([.string("answer")]),
                ]
            )
        )
    }

    func testCompletionRequestParsesStructuredOutputsChoice() throws {
        let request = try OpenAICompletionRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "prompt": "hello",
                  "structured_outputs": {
                    "type": "choice",
                    "choices": ["A", "B"]
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.structuredOutput, .choice(["A", "B"]))
    }

    func testAllowedNextTokenIDsForPrefixCases() {
        let allowed = TokenSampler.allowedNextTokenIDs(
            allowedSequences: [
                [10, 20],
                [10, 30],
                [40],
            ],
            generatedTokens: [10]
        )

        XCTAssertEqual(allowed, [20, 30])
    }

    func testAllowedNextTokenIDsExcludesCompletedSequences() {
        let allowed = TokenSampler.allowedNextTokenIDs(
            allowedSequences: [
                [10],
                [10, 20],
            ],
            generatedTokens: [10]
        )

        XCTAssertEqual(allowed, [20])
    }

    func testAllowedNextTokenIDsReturnsNilWhenOnlyViableSequenceIsComplete() {
        let allowed = TokenSampler.allowedNextTokenIDs(
            allowedSequences: [[10, 20]],
            generatedTokens: [10, 20]
        )

        XCTAssertNil(allowed)
    }

    func testAllowedNextTokenIDsForcesEOSAfterChoiceTextWhenSequencesAreEOSTerminated() {
        let eosTokenID = 999
        let afterChoiceText = TokenSampler.allowedNextTokenIDs(
            allowedSequences: [
                [10, 20, eosTokenID],
                [30, 40, eosTokenID],
            ],
            generatedTokens: [10, 20]
        )
        let afterChoiceAndEOS = TokenSampler.allowedNextTokenIDs(
            allowedSequences: [
                [10, 20, eosTokenID],
                [30, 40, eosTokenID],
            ],
            generatedTokens: [10, 20, eosTokenID]
        )

        XCTAssertEqual(afterChoiceText, [eosTokenID])
        XCTAssertNil(afterChoiceAndEOS)
    }

    func testAllowedNextTokenIDsReturnsNilForEmptyOrNoMatchingSequences() {
        XCTAssertNil(
            TokenSampler.allowedNextTokenIDs(
                allowedSequences: [],
                generatedTokens: []
            )
        )
        XCTAssertNil(
            TokenSampler.allowedNextTokenIDs(
                allowedSequences: [[10, 20]],
                generatedTokens: [99]
            )
        )
    }

    func testChoiceLogitsMaskConstrainsGreedyArgmax() throws {
        try MLXMetalRuntime.requireAvailable()

        let logits = MLXArray([0.1, 99.0, 0.2, 0.3, 0.4, 0.5].map(Float.init))
        let sampled = TokenSampler.sample(
            logits: logits,
            parameters: SamplingParameters(
                temperature: 0,
                allowedSequences: [
                    [10, 2],
                    [10, 4],
                ]
            ),
            generatedTokens: [10]
        )

        XCTAssertTrue([2, 4].contains(sampled.item(Int.self)))
    }

    func testThinkingBudgetForcesCloseTokenBeforeSixthReasoningToken() throws {
        try MLXMetalRuntime.requireAvailable()

        let logits = MLXArray([0.0, 10.0, -5.0].map(Float.init))
        let parameters = SamplingParameters(temperature: 0)
        var state: ThinkingBudgetState? = ThinkingBudgetState(
            configuration: ThinkingBudgetConfiguration(
                budget: 5,
                closeTokenIDs: [2],
                startsInThinking: true
            )
        )
        var generatedTokens: [Int] = []

        for _ in 0 ..< 5 {
            let sampled = TokenSampler.sample(
                logits: logits,
                parameters: parameters,
                generatedTokens: generatedTokens,
                thinkingBudgetState: &state
            )
            let tokenID = sampled.item(Int.self)
            XCTAssertEqual(tokenID, 1)
            generatedTokens.append(tokenID)
            state?.advance(tokenID: tokenID)
        }

        let closeToken = TokenSampler.sample(
            logits: logits,
            parameters: parameters,
            generatedTokens: generatedTokens,
            thinkingBudgetState: &state
        ).item(Int.self)
        XCTAssertEqual(closeToken, 2)
        state?.advance(tokenID: closeToken)

        XCTAssertEqual(state?.countedThinkingTokens, 5)
        XCTAssertEqual(state?.isInThinking, false)
    }
}
