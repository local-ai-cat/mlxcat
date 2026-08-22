import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import MLXVLM
import Tokenizers
import XCTest

private struct EngineTraceStep {
    let tokenId: Int
    let logits: MLXArray
}

private struct EngineGateResult: Sendable {
    let batchCheckedTokens: Int
    let batchMismatches: Int
    let cancelCheckedTokens: Int
    let cancelMismatches: Int
    let cancelledResponses: Int
    let queueFullRejected: Bool
    let streamCheckedTokens: Int
    let streamMismatches: Int
    let streamMisroutedResponses: Int
}

final class SchedulerEngineTests: XCTestCase {
    private static let prompts = [
        "The capital of France is",
        "Write a haiku about GPU kernels.",
        "In Swift, an actor protects",
        "List three colors:",
        "What is 2 + 2?",
        "Translate hello to Spanish:",
        "The largest planet is",
        "Complete: once upon a",
    ]

    func testEngineConcurrentGenerationCancellationAndBackpressure() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run M2 engine gate.")
        }
        guard resolution.url.lastPathComponent == "Qwen3-0.6B-4bit" else {
            throw XCTSkip("M2 engine fixture is pinned to Qwen3-0.6B-4bit.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )
        let prompts = Self.prompts
        let result = try await container.perform { context in
            try await Self.evaluateEngineGate(context: context, prompts: prompts)
        }

        if ProcessInfo.processInfo.environment["MLXSERVE_DEBUG_ENGINE_GATE"] == "1" {
            print(
                "M2 engine: batchChecked=\(result.batchCheckedTokens), batchMismatches=\(result.batchMismatches), cancelChecked=\(result.cancelCheckedTokens), cancelMismatches=\(result.cancelMismatches), cancelledResponses=\(result.cancelledResponses), streamChecked=\(result.streamCheckedTokens), streamMismatches=\(result.streamMismatches), streamMisroutes=\(result.streamMisroutedResponses), queueFullRejected=\(result.queueFullRejected)"
            )
        }

        XCTAssertGreaterThan(result.batchCheckedTokens, 0)
        XCTAssertEqual(result.batchMismatches, 0)
        XCTAssertGreaterThan(result.cancelCheckedTokens, 0)
        XCTAssertEqual(result.cancelMismatches, 0)
        XCTAssertEqual(result.cancelledResponses, 1)
        XCTAssertGreaterThan(result.streamCheckedTokens, 0)
        XCTAssertEqual(result.streamMismatches, 0)
        XCTAssertEqual(result.streamMisroutedResponses, 0)
        XCTAssertTrue(result.queueFullRejected)
    }

    func testPoisonPillAdmissionDoesNotWedgeFollowingGoodStream() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run poison-pill recovery gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let engine = MLXCatEngine(
                model: context.model,
                parameters: GenerateParameters(maxTokens: 2, temperature: 0),
                maxConcurrentRequests: 1
            )
            let badInput = LMInput(tokens: MLXArray([Int32]()))
            let goodInput = try await context.processor.prepare(
                input: UserInput(prompt: "The capital of France is")
            )

            async let badResponses = Self.collectResponses(
                from: engine.stream(
                    Request(
                        uid: "bad",
                        input: badInput,
                        maxTokens: 2,
                        sampling: SamplingParameters(temperature: 0)
                    )
                )
            )
            async let goodResponses = Self.collectResponses(
                from: engine.stream(
                    Request(
                        uid: "good",
                        input: goodInput,
                        maxTokens: 2,
                        sampling: SamplingParameters(temperature: 0)
                    )
                )
            )

            let (bad, good) = try await (badResponses, goodResponses)
            XCTAssertEqual(bad.last?.uid, "bad")
            if case .failed? = bad.last?.finishReason {
            } else {
                XCTFail("bad request should finish with a failed terminal response")
            }
            XCTAssertEqual(good.filter { $0.token >= 0 }.count, 2)
            XCTAssertEqual(good.last?.finishReason, .length)
        }
    }

    func testSingleTokenAdmissionProducesTokens() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run single-token admission gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let engine = MLXCatEngine(
                model: context.model,
                parameters: GenerateParameters(maxTokens: 2, temperature: 0),
                maxConcurrentRequests: 1
            )
            let responses = try await Self.collectResponses(
                from: engine.stream(
                    Request(
                        uid: "single-token",
                        input: LMInput(tokens: MLXArray([Int32(0)])),
                        maxTokens: 2,
                        sampling: SamplingParameters(temperature: 0)
                    )
                )
            )

            XCTAssertEqual(responses.filter { $0.token >= 0 }.count, 2)
            XCTAssertEqual(responses.last?.finishReason, .length)
        }
    }

    func testEOSFinishReasonStopsStream() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run EOS finish-reason gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
            let input = try await context.processor.prepare(
                input: UserInput(prompt: "The capital of France is")
            )
            let firstToken = try Self.serialTrace(
                model: context.model,
                input: input,
                parameters: parameters,
                steps: 1
            )[0].tokenId
            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 1
            )

            let responses = try await Self.collectResponses(
                from: engine.stream(
                    Request(
                        uid: "eos",
                        input: input,
                        maxTokens: 4,
                        sampling: SamplingParameters(temperature: 0),
                        eosTokenIds: [firstToken]
                    )
                )
            )

            XCTAssertEqual(responses.count, 1)
            XCTAssertEqual(responses.first?.token, firstToken)
            XCTAssertEqual(responses.first?.finishReason, .stop)
        }
    }

    func testConcurrentEngineStreamsStayDemuxed() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip("Set MLXSERVE_TEST_MODEL to run concurrent stream gate.")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 2, temperature: 0)
            let inputs = try await Array(Self.prompts.prefix(4)).mapAsync { prompt in
                try await context.processor.prepare(input: UserInput(prompt: prompt))
            }
            let engine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 2
            )
            let requests = inputs.enumerated().map { index, input in
                Request(
                    uid: "s\(index)",
                    input: input,
                    maxTokens: 2,
                    sampling: SamplingParameters(temperature: 0)
                )
            }

            async let responses0 = Self.collectResponses(from: engine.stream(requests[0]))
            async let responses1 = Self.collectResponses(from: engine.stream(requests[1]))
            async let responses2 = Self.collectResponses(from: engine.stream(requests[2]))
            async let responses3 = Self.collectResponses(from: engine.stream(requests[3]))
            let allResponses = try await [responses0, responses1, responses2, responses3]

            for (index, responses) in allResponses.enumerated() {
                let uid = "s\(index)"
                XCTAssertFalse(responses.isEmpty)
                XCTAssertTrue(responses.allSatisfy { $0.uid == uid })
                XCTAssertEqual(responses.last?.finishReason, .length)
            }
        }
    }

    func testVLMImageDescribeProducesTokens() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolveVLM() else {
            throw XCTSkip("Set MLXSERVE_VLM_TEST_MODEL to run VLM engine gate.")
        }

        let container = try await VLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let eosTokenIds = Self.eosTokenIds(context: context)
            let input = try await context.processor.prepare(
                input: UserInput(
                    prompt: "Describe the image briefly.",
                    images: [Self.testImage(width: 96, height: 96)]
                )
            )
            let engine = MLXCatEngine(
                model: context.model,
                parameters: GenerateParameters(maxTokens: 24, temperature: 0),
                maxConcurrentRequests: 1,
                serializedDecode: true
            )

            let responses = try await Self.collectResponses(
                from: engine.stream(
                    Request(
                        uid: "vlm-single",
                        input: input,
                        maxTokens: 24,
                        sampling: SamplingParameters(temperature: 0),
                        eosTokenIds: eosTokenIds
                    )
                )
            )
            let tokens = responses.filter { $0.token >= 0 && !eosTokenIds.contains($0.token) }

            XCTAssertGreaterThan(tokens.count, 1)
            XCTAssertNotNil(responses.last?.finishReason)
        }
    }

    func testVLMConcurrentImageRequestsBothComplete() async throws {
        try MLXMetalRuntime.requireAvailable()

        guard let resolution = TestModelResolver.resolveVLM() else {
            throw XCTSkip("Set MLXSERVE_VLM_TEST_MODEL to run VLM concurrent engine gate.")
        }
        guard try Self.modelType(in: resolution.url) == "qwen2_vl" else {
            throw XCTSkip("VLM batch-equality gate is pinned to Qwen2-VL.")
        }

        let container = try await VLMModelFactory.shared.loadContainer(
            from: resolution.url,
            using: #huggingFaceTokenizerLoader()
        )

        try await container.perform { context in
            let eosTokenIds = Self.eosTokenIds(context: context)
            let inputs = try await [
                Self.testImage(width: 96, height: 96),
                Self.testImage(width: 160, height: 112),
                Self.testImage(width: 112, height: 160),
            ].mapAsync { image in
                try await context.processor.prepare(
                    input: UserInput(
                        prompt: "Describe the image briefly.",
                        images: [image]
                    )
                )
            }
            let parameters = GenerateParameters(maxTokens: 24, temperature: 0)
            let requests = inputs.enumerated().map { index, input in
                Request(
                    uid: "vlm-\(index)",
                    input: input,
                    maxTokens: parameters.maxTokens ?? 24,
                    sampling: SamplingParameters(temperature: 0),
                    eosTokenIds: eosTokenIds
                )
            }
            var serial: [[Int]] = []
            for request in requests {
                let serialEngine = MLXCatEngine(
                    model: context.model,
                    parameters: parameters,
                    maxConcurrentRequests: 1
                )
                let output = try await serialEngine.generate([request])
                serial.append(output[request.uid, default: []])
            }

            let batchedEngine = MLXCatEngine(
                model: context.model,
                parameters: parameters,
                maxConcurrentRequests: 3,
                serializedDecode: false
            )
            let batched = try await batchedEngine.generate(requests)

            for index in requests.indices {
                let uid = "vlm-\(index)"
                let tokens = batched[uid, default: []]
                let nonEOSTokens = tokens.filter { !eosTokenIds.contains($0) }
                XCTAssertEqual(tokens, serial[index], uid)
                XCTAssertGreaterThan(nonEOSTokens.count, 1, uid)
            }
        }
    }

    /// Batched-vs-serial token equality **with images**, across VLM families.
    ///
    /// `testVLMConcurrentImageRequestsBothComplete` already does exactly this
    /// check — and is pinned to `qwen2_vl`. That pin is why the gemma-4 batching
    /// unlock is stuck: lifting `gemma4` off `scalarOffsetVLMModelTypes` is worth
    /// 50x TTFT and 2.17x aggregate throughput at c4 (`docs/COMPETITIVE.md`), the
    /// text evidence is unambiguous, and the exclusion's stated cause is MRoPE
    /// position ids derived from a scalar `cache.offset` — an **image** concern
    /// that no text measurement can speak to.
    ///
    /// So the pin is lifted here as an opt-in, leaving the pinned gate's meaning
    /// untouched. This is the gate that decides whether the unlock ships.
    ///
    ///     MLXCAT_VLM_BATCH_MODELS=$R/gemma-4-E2B-it-qat-4bit \
    ///       swift test --filter testVLMBatchEqualityAcrossModels
    func testVLMBatchEqualityAcrossModels() async throws {
        try MLXMetalRuntime.requireAvailable()

        let raw = ProcessInfo.processInfo.environment["MLXCAT_VLM_BATCH_MODELS"] ?? ""
        let paths = raw.split(separator: ",").map(String.init).filter { !$0.isEmpty }
        guard !paths.isEmpty else {
            throw XCTSkip("Set MLXCAT_VLM_BATCH_MODELS to a comma-separated list of VLM directories.")
        }

        var failures: [String] = []
        for path in paths {
            let url = URL(fileURLWithPath: path)
            let name = url.lastPathComponent
            let container = try await VLMModelFactory.shared.loadContainer(
                from: url, using: #huggingFaceTokenizerLoader())

            let modelFailures: [String] = try await container.perform { context in
                var found: [String] = []
                let eosTokenIds = Self.eosTokenIds(context: context)
                // Different image sizes on purpose: equal-sized inputs would make
                // every per-row quantity uniform and hide exactly the ragged-batch
                // failure this gate exists to catch.
                let inputs = try await [
                    Self.testImage(width: 96, height: 96),
                    Self.testImage(width: 160, height: 112),
                    Self.testImage(width: 112, height: 160),
                ].mapAsync { image in
                    try await context.processor.prepare(
                        input: UserInput(prompt: "Describe the image briefly.", images: [image]))
                }
                let parameters = GenerateParameters(maxTokens: 24, temperature: 0)
                let requests = inputs.enumerated().map { index, input in
                    Request(
                        uid: "vlm-\(index)", input: input,
                        maxTokens: parameters.maxTokens ?? 24,
                        sampling: SamplingParameters(temperature: 0),
                        eosTokenIds: eosTokenIds)
                }

                var serial: [[Int]] = []
                for request in requests {
                    let serialEngine = MLXCatEngine(
                        model: context.model, parameters: parameters, maxConcurrentRequests: 1)
                    serial.append(try await serialEngine.generate([request])[request.uid, default: []])
                }

                // `.never` is the capability question — CAN these rows share a
                // batch? `.multimodalOnly` is the shipping question — does the
                // policy that unlocks gemma-4's text batching still keep the
                // image path exact? Only the second one is allowed to fail the
                // gate; the first is recorded so the reason for the policy stays
                // visible in the output rather than living only in a comment.
                for policy in [SerializationPolicy.never, .multimodalOnly] {
                    let batchedEngine = MLXCatEngine(
                        model: context.model, parameters: parameters,
                        maxConcurrentRequests: requests.count, serializationPolicy: policy)
                    let batched = try await batchedEngine.generate(requests)

                    for index in requests.indices {
                        let uid = "vlm-\(index)"
                        let got = batched[uid, default: []]
                        let want = serial[index]
                        let nonEOS = got.filter { !eosTokenIds.contains($0) }
                        let ok = got == want && nonEOS.count > 1
                        print(
                            "VLMBATCH \(ok ? "OK  " : "FAIL") \(name) policy=\(policy) \(uid) "
                                + "batched=\(got.count) serial=\(want.count) nonEOS=\(nonEOS.count)")
                        if !ok, policy == .multimodalOnly {
                            found.append(
                                "\(name) \(uid) under \(policy): batched \(got.prefix(8))... "
                                    + "vs serial \(want.prefix(8))...")
                        }
                    }
                }
                return found
            }
            failures.append(contentsOf: modelFailures)
        }
        XCTAssertTrue(
            failures.isEmpty,
            "VLM families failing batched-vs-serial equality with images:\n"
                + failures.joined(separator: "\n"))
    }

    private static func evaluateEngineGate(
        context: ModelContext,
        prompts: [String]
    ) async throws -> EngineGateResult {
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        var inputs: [LMInput] = []
        for prompt in prompts {
            inputs.append(try await context.processor.prepare(input: UserInput(prompt: prompt)))
        }
        let serial = try inputs.map {
            try Self.serialTrace(
                model: context.model,
                input: $0,
                parameters: parameters,
                steps: parameters.maxTokens ?? 4
            )
        }

        let engine = MLXCatEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: 8
        )

        var batchCheckedTokens = 0
        var batchMismatches = 0
        for batchSize in [1, 4, 8] {
            let requests = (0 ..< batchSize).map { row in
                Request(
                    uid: "batch-\(batchSize)-\(row)",
                    input: inputs[row],
                    maxTokens: parameters.maxTokens ?? 4,
                    sampling: SamplingParameters(temperature: 0)
                )
            }
            let outputs = try await engine.generate(requests)
            for row in 0 ..< batchSize {
                let uid = "batch-\(batchSize)-\(row)"
                let comparison = compareWideMarginTokens(
                    generated: outputs[uid, default: []],
                    serial: serial[row]
                )
                batchCheckedTokens += comparison.checked
                batchMismatches += comparison.mismatches
            }
        }

        let cancelRows = [0, 1, 2]
        for row in cancelRows {
            try await engine.submit(
                Request(
                    uid: "cancel-\(row)",
                    input: inputs[row],
                    maxTokens: parameters.maxTokens ?? 4,
                    sampling: SamplingParameters(temperature: 0)
                )
            )
        }

        _ = try await engine.step()
        await engine.cancel(uid: "cancel-1")
        while await !engine.isIdle {
            _ = try await engine.step()
        }

        var cancelCheckedTokens = 0
        var cancelMismatches = 0
        for row in [0, 2] {
            let comparison = await compareWideMarginTokens(
                generated: engine.tokens(for: "cancel-\(row)"),
                serial: serial[row]
            )
            cancelCheckedTokens += comparison.checked
            cancelMismatches += comparison.mismatches
        }
        let cancelledResponses = await engine.responses(for: "cancel-1")
            .filter { $0.finishReason == .cancelled }
            .count

        let streamEngine = MLXCatEngine(
            model: context.model,
            parameters: parameters,
            maxConcurrentRequests: 2
        )
        let stream0 = streamEngine.stream(
            Request(
                uid: "stream-0",
                input: inputs[0],
                maxTokens: parameters.maxTokens ?? 4,
                sampling: SamplingParameters(temperature: 0)
            )
        )
        let stream1 = streamEngine.stream(
            Request(
                uid: "stream-1",
                input: inputs[1],
                maxTokens: parameters.maxTokens ?? 4,
                sampling: SamplingParameters(temperature: 0)
            )
        )
        let streamResponses0 = try await collectResponses(from: stream0)
        let streamResponses1 = try await collectResponses(from: stream1)
        let streamMisroutedResponses = streamResponses0.filter { $0.uid != "stream-0" }.count
            + streamResponses1.filter { $0.uid != "stream-1" }.count
        let streamComparison0 = compareWideMarginTokens(
            generated: streamResponses0.map(\.token),
            serial: serial[0]
        )
        let streamComparison1 = compareWideMarginTokens(
            generated: streamResponses1.map(\.token),
            serial: serial[1]
        )
        let streamCheckedTokens = streamComparison0.checked + streamComparison1.checked
        let streamMismatches = streamComparison0.mismatches + streamComparison1.mismatches

        var queueFullRejected = false
        for index in 0 ..< 32 {
            try await engine.submit(
                Request(
                    uid: "queue-\(index)",
                    input: inputs[0],
                    maxTokens: 1,
                    sampling: SamplingParameters(temperature: 0)
                )
            )
        }
        do {
            try await engine.submit(
                Request(
                    uid: "queue-overflow",
                    input: inputs[0],
                    maxTokens: 1,
                    sampling: SamplingParameters(temperature: 0)
                )
            )
        } catch SchedulerError.queueFull {
            queueFullRejected = true
        }

        return EngineGateResult(
            batchCheckedTokens: batchCheckedTokens,
            batchMismatches: batchMismatches,
            cancelCheckedTokens: cancelCheckedTokens,
            cancelMismatches: cancelMismatches,
            cancelledResponses: cancelledResponses,
            queueFullRejected: queueFullRejected,
            streamCheckedTokens: streamCheckedTokens,
            streamMismatches: streamMismatches,
            streamMisroutedResponses: streamMisroutedResponses
        )
    }

    private static func collectResponses(
        from stream: AsyncThrowingStream<Response, Error>
    ) async throws -> [Response] {
        var responses: [Response] = []
        for try await response in stream {
            responses.append(response)
        }
        return responses
    }

    private static func compareWideMarginTokens(
        generated: [Int],
        serial: [EngineTraceStep]
    ) -> (checked: Int, mismatches: Int) {
        var checked = 0
        var mismatches = 0
        for (step, token) in generated.enumerated() where step < serial.count {
            let serialStep = serial[step]
            let margin = topOneTopTwoMargin(serialStep.logits)
            if margin > 1.25 * 4 + 1e-3 {
                checked += 1
                if token != serialStep.tokenId {
                    mismatches += 1
                }
            }
        }
        return (checked, mismatches)
    }

    private static func serialTrace(
        model: any LanguageModel,
        input: LMInput,
        parameters: GenerateParameters,
        steps: Int
    ) throws -> [EngineTraceStep] {
        let cache = try model.newCache(parameters: parameters)
        var state: LMOutput.State?

        var currentLogits: MLXArray
        var currentToken: MLXArray

        switch try model.prepare(input, cache: cache, state: nil, prefill: parameters.prefill) {
        case .tokens(let tokens):
            let output = model(tokens[text: .newAxis], cache: cache, state: state)
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
        case .logits(let output):
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
        }
        eval(currentToken, currentLogits, cache)

        var trace: [EngineTraceStep] = []
        for _ in 0 ..< steps {
            eval(currentToken, currentLogits)
            trace.append(EngineTraceStep(tokenId: currentToken.item(Int.self), logits: currentLogits))

            let output = model(
                LMInput.Text(tokens: currentToken[.newAxis, 0...]),
                cache: cache,
                state: state
            )
            state = output.state
            currentLogits = output.logits[0..., -1, 0...]
            currentToken = argMax(currentLogits, axis: -1)
            asyncEval(currentToken, currentLogits)
        }

        eval(currentToken, currentLogits)
        return trace
    }

    private static func testImage(width: Int, height: Int) -> UserInput.Image {
        let extent = CGRect(x: 0, y: 0, width: width, height: height)
        let image = CIImage(color: CIColor(red: 0.2, green: 0.6, blue: 0.9))
            .cropped(to: extent)
        return .ciImage(image)
    }

    private static func eosTokenIds(context: ModelContext) -> Set<Int> {
        var eosTokenIds = context.configuration.eosTokenIds
        if let tokenizerEosTokenId = context.tokenizer.eosTokenId {
            eosTokenIds.insert(tokenizerEosTokenId)
        }
        return eosTokenIds
    }

    private static func modelType(in modelURL: URL) throws -> String {
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder.json5().decode(ModelKindConfiguration.self, from: configData)
        return config.modelType.lowercased()
    }

    private static func topOneTopTwoMargin(_ logits: MLXArray) -> Float {
        let topValues = top(logits.asType(.float32), k: 2, axis: -1).asArray(Float.self)
        let sorted = topValues.sorted(by: >)
        return sorted[0] - sorted[1]
    }
}

private struct ModelKindConfiguration: Decodable {
    let modelType: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
    }
}

private extension Array {
    func mapAsync<T>(_ transform: (Element) async throws -> T) async throws -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self {
            result.append(try await transform(element))
        }
        return result
    }
}
