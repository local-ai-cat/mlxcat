import Foundation
@testable import MLXCat
@testable import MLXCatHTTP
import XCTest

extension OpenAIServerTests {
    func testChatRequestParsePopulatesSamplingAndOpenAIFields() throws {
        let request = try OpenAIChatRequest.parse(samplingRequestBody)

        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages[0].reasoningContent, "prior reasoning")
        XCTAssertEqual(request.maxTokens, 32)
        XCTAssertEqual(request.temperature, 0.7, accuracy: 0.0001)
        XCTAssertEqual(request.topP, 0.8, accuracy: 0.0001)
        XCTAssertEqual(request.topK, 42)
        XCTAssertEqual(request.repetitionPenalty, 1.2, accuracy: 0.0001)
        XCTAssertEqual(request.minP, 0.05, accuracy: 0.0001)
        XCTAssertEqual(request.xtcProbability, 0.3, accuracy: 0.0001)
        XCTAssertEqual(request.xtcThreshold, 0.4, accuracy: 0.0001)
        XCTAssertEqual(request.presencePenalty, 0.5, accuracy: 0.0001)
        XCTAssertEqual(request.frequencyPenalty, 0.6, accuracy: 0.0001)
        XCTAssertEqual(request.minTokens, 7)
        XCTAssertEqual(try XCTUnwrap(request.logitBias[42]), 3.5, accuracy: 0.0001)
        XCTAssertEqual(try XCTUnwrap(request.logitBias[99]), -2, accuracy: 0.0001)
        XCTAssertEqual(request.stop, ["END", "STOP"])
        XCTAssertEqual(request.seed, 1234)
        XCTAssertTrue(request.stream)
        XCTAssertTrue(request.includeUsage)
        XCTAssertEqual(request.enableThinking, false)
        XCTAssertFalse(request.startInThinking)
        XCTAssertEqual(request.thinkingBudget, 24)
        XCTAssertEqual(request.cacheSession, "body-session")
        XCTAssertEqual(request.chatTemplateKwargs?["reasoning_effort"], .string("low"))
        XCTAssertEqual(request.chatTemplateKwargs?["budget"], .number(16))
        XCTAssertEqual(request.chatTemplateKwargs?["nested"], .object(["enabled": .bool(true)]))
    }

    func testChatRequestParseCapturesToolsArray() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "weather?"}],
                  "tools": [
                    {
                      "type": "function",
                      "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                          "type": "object",
                          "properties": {
                            "city": {"type": "string"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                          },
                          "required": ["city"]
                        }
                      }
                    }
                  ]
                }
                """.utf8
            )
        )

        XCTAssertEqual(
            request.tools,
            [
                .object([
                    "type": .string("function"),
                    "function": .object([
                        "name": .string("get_weather"),
                        "description": .string("Get weather for a city"),
                        "parameters": .object([
                            "type": .string("object"),
                            "properties": .object([
                                "city": .object(["type": .string("string")]),
                                "units": .object([
                                    "type": .string("string"),
                                    "enum": .array([.string("celsius"), .string("fahrenheit")])
                                ])
                            ]),
                            "required": .array([.string("city")])
                        ])
                    ])
                ])
            ]
        )
    }

    func testChatRequestParseToolChoiceStringForms() throws {
        for (rawChoice, expectedChoice) in [
            ("none", OpenAIToolChoice.none),
            ("auto", OpenAIToolChoice.auto),
            ("required", OpenAIToolChoice.required)
        ] {
            let request = try OpenAIChatRequest.parse(
                Data(
                    """
                    {
                      "model": "test-model",
                      "messages": [{"role": "user", "content": "hello"}],
                      "tool_choice": "\(rawChoice)"
                    }
                    """.utf8
                )
            )

            XCTAssertEqual(request.toolChoice, expectedChoice)
        }
    }

    func testChatRequestParseToolChoiceFunctionForm() throws {
        let request = try OpenAIChatRequest.parse(
            Data(
                """
                {
                  "model": "test-model",
                  "messages": [{"role": "user", "content": "hello"}],
                  "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"}
                  }
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.toolChoice, .function("get_weather"))
    }

}

private var samplingRequestBody: Data {
    Data(
        """
        {
          "model": "test-model",
          "messages": [
            {
              "role": "assistant",
              "content": "hello",
              "reasoning_content": "prior reasoning"
            }
          ],
          "max_completion_tokens": 32,
          "temperature": 0.7,
          "top_p": 0.8,
          "top_k": 42,
          "repetition_penalty": 1.2,
          "min_p": 0.05,
          "xtc_probability": 0.3,
          "xtc_threshold": 0.4,
          "presence_penalty": 0.5,
          "frequency_penalty": 0.6,
          "min_tokens": 7,
          "logit_bias": {"42": 3.5, "99": -2},
          "stop": ["END", "STOP"],
          "seed": 1234,
          "stream": true,
          "stream_options": { "include_usage": true },
          "enable_thinking": false,
          "thinking_budget": 24,
          "cache_session": "body-session",
          "chat_template_kwargs": {
            "reasoning_effort": "low",
            "budget": 16,
            "nested": { "enabled": true }
          }
        }
        """.utf8
    )
}
