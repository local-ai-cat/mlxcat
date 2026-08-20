import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import Tokenizers

private let prefixSentence =
    "The capital of France is Paris. Swift concurrency protects shared state. GPU kernels execute matrix operations quickly. "
private let prefixRepeatCount = 26

@main
struct MLXCatBench {
    static func main() async throws {
        let config = try BenchConfig.parse(CommandLine.arguments)
        let modelURL = URL(fileURLWithPath: config.modelPath)

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )

        let result = try await container.perform { context in
            try await NativeBenchmark(config: config, modelName: modelURL.lastPathComponent)
                .run(context: context)
        }

        let markdown = result.markdown()
        print(markdown)

        if let outputPath = config.outputPath {
            try markdown.write(
                to: URL(fileURLWithPath: outputPath),
                atomically: true,
                encoding: .utf8
            )
        }
    }
}

private struct BenchConfig {
    let modelPath: String
    let runs: Int
    let warmup: Int
    let decodeTokens: Int
    let outputPath: String?

    static func parse(_ arguments: [String]) throws -> BenchConfig {
        var modelPath = ProcessInfo.processInfo.environment["MLXSERVE_MODEL_DIR"]
            ?? ProcessInfo.processInfo.environment["MLXSERVE_TEST_MODEL"]
        var runs = 5
        var warmup = 2
        var decodeTokens = 16
        var outputPath: String?

        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--model-dir":
                index += 1
                modelPath = try value(arguments, at: index, for: argument)
            case "--runs":
                index += 1
                runs = try intValue(arguments, at: index, for: argument)
            case "--warmup":
                index += 1
                warmup = try intValue(arguments, at: index, for: argument)
            case "--decode-tokens":
                index += 1
                decodeTokens = try intValue(arguments, at: index, for: argument)
            case "--output":
                index += 1
                outputPath = try value(arguments, at: index, for: argument)
            case "--help", "-h":
                printHelp()
                Foundation.exit(0)
            default:
                throw BenchError.invalidArgument(argument)
            }
            index += 1
        }

        guard let modelPath else {
            throw BenchError.invalidArgument("missing --model-dir or MLXSERVE_MODEL_DIR")
        }

        guard runs > 0 else { throw BenchError.invalidArgument("--runs must be > 0") }
        guard warmup >= 0 else { throw BenchError.invalidArgument("--warmup must be >= 0") }
        guard decodeTokens > 0 else { throw BenchError.invalidArgument("--decode-tokens must be > 0") }

        return BenchConfig(
            modelPath: modelPath,
            runs: runs,
            warmup: warmup,
            decodeTokens: decodeTokens,
            outputPath: outputPath
        )
    }

    private static func value(_ arguments: [String], at index: Int, for option: String) throws -> String {
        guard arguments.indices.contains(index) else {
            throw BenchError.invalidArgument("missing value for \(option)")
        }
        return arguments[index]
    }

    private static func intValue(_ arguments: [String], at index: Int, for option: String) throws -> Int {
        guard let value = Int(try value(arguments, at: index, for: option)) else {
            throw BenchError.invalidArgument("invalid integer for \(option)")
        }
        return value
    }

    private static func printHelp() {
        print(
            """
            Usage: swift run mlxcat-bench [options]

            Options:
              --model-dir PATH       MLX model directory. Required unless MLXSERVE_MODEL_DIR is set.
              --runs N               Timed runs per metric. Default: 5
              --warmup N             Warmup runs per metric. Default: 2
              --decode-tokens N      Generated tokens per throughput run. Default: 16
              --output PATH          Optional Markdown output path.
            """
        )
    }
}

private final class NativeBenchmark {
    private let config: BenchConfig
    private let modelName: String
    private let blockSize = 256

    init(config: BenchConfig, modelName: String) {
        self.config = config
        self.modelName = modelName
    }

    func run(context: ModelContext) async throws -> NativeBenchmarkResult {
        let model = context.model
        let parameters = GenerateParameters(maxTokens: config.decodeTokens, temperature: 0)
        let prefixText = String(repeating: prefixSentence, count: prefixRepeatCount)
        let prefixTokens = tokenIDs(for: prefixText, context: context)
        let shortPrompt = tokenIDs(
            for: "The capital of France is",
            context: context
        )
        let suffixes = [
            " The answer is",
            " Therefore",
            " In summary",
            " A careful implementation",
            " The benchmark result",
            " Swift concurrency",
            " The next step",
            " Prefix caching",
        ].map { text in
            tokenIDs(
                for: text,
                context: context,
                addSpecialTokens: false
            )
        }

        let prefillPrompt = prefixTokens
        let decodePrompt = prefixTokens + suffixes[0]
        let ttftPrompt = shortPrompt

        let prefillSeconds = try measureMany {
            try prefill(model: model, tokens: prefillPrompt, parameters: parameters)
        }
        let prefillTPS = Double(prefillPrompt.count) / median(prefillSeconds)

        let decodeSeconds = try measureDecodeMany(
            prepare: {
                try prepareDecodeState(model: model, tokens: decodePrompt, parameters: parameters)
            },
            run: { state in
                try decode(model: model, state: &state, decodeTokens: config.decodeTokens)
            }
        )
        let decodeTPS = Double(config.decodeTokens) / median(decodeSeconds)

        let ttftSeconds = try await measureManyAsync {
            try await ttft(model: model, tokens: ttftPrompt, parameters: parameters, prefixStore: nil)
        }

        let cache = try await cacheSpeedup(
            model: model,
            prefixTokens: prefixTokens,
            warmSuffix: suffixes[0],
            measuredSuffix: suffixes[1],
            parameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )

        var concurrencyResults: [ConcurrencyBenchmarkResult] = []
        for concurrency in [1, 2, 4, 8] {
            let prompts = (0 ..< concurrency).map { index in
                prefixTokens + suffixes[index % suffixes.count]
            }
            let seconds = try await measureManyAsync {
                try await throughput(
                    model: model,
                    prompts: prompts,
                    parameters: parameters,
                    concurrency: concurrency
                )
            }
            let totalTokens = concurrency * config.decodeTokens
            concurrencyResults.append(
                ConcurrencyBenchmarkResult(
                    concurrency: concurrency,
                    generatedTokens: totalTokens,
                    medianSeconds: median(seconds),
                    throughputTokensPerSecond: Double(totalTokens) / median(seconds)
                )
            )
        }

        return NativeBenchmarkResult(
            modelName: modelName,
            runs: config.runs,
            warmup: config.warmup,
            promptDescription: "prefix sentence repeated \(prefixRepeatCount)x; suffix prompts are fixed literals in MLXCatBench",
            promptTokens: prefillPrompt.count,
            decodeTokens: config.decodeTokens,
            prefillTokensPerSecond: prefillTPS,
            decodeTokensPerSecond: decodeTPS,
            ttftMilliseconds: median(ttftSeconds) * 1_000,
            coldTTFTMilliseconds: cache.coldMedianSeconds * 1_000,
            warmTTFTMilliseconds: cache.warmMedianSeconds * 1_000,
            cacheSpeedup: cache.speedup,
            cacheFetchHits: cache.fetchHits,
            concurrency: concurrencyResults
        )
    }

    private func measureMany(_ body: () throws -> Void) throws -> [Double] {
        for _ in 0 ..< config.warmup {
            try body()
        }

        return try (0 ..< config.runs).map { _ in
            let start = DispatchTime.now().uptimeNanoseconds
            try body()
            let end = DispatchTime.now().uptimeNanoseconds
            return Double(end - start) / 1_000_000_000
        }
    }

    private func measureDecodeMany<State>(
        prepare: () throws -> State,
        run: (inout State) throws -> Void
    ) throws -> [Double] {
        for _ in 0 ..< config.warmup {
            var state = try prepare()
            try run(&state)
        }

        return try (0 ..< config.runs).map { _ in
            var state = try prepare()
            let start = DispatchTime.now().uptimeNanoseconds
            try run(&state)
            let end = DispatchTime.now().uptimeNanoseconds
            return Double(end - start) / 1_000_000_000
        }
    }

    private func measureManyAsync(_ body: () async throws -> Void) async throws -> [Double] {
        for _ in 0 ..< config.warmup {
            try await body()
        }

        var timings: [Double] = []
        timings.reserveCapacity(config.runs)
        for _ in 0 ..< config.runs {
            let start = DispatchTime.now().uptimeNanoseconds
            try await body()
            let end = DispatchTime.now().uptimeNanoseconds
            timings.append(Double(end - start) / 1_000_000_000)
        }
        return timings
    }

    private func prefill(
        model: any LanguageModel,
        tokens: [Int],
        parameters: GenerateParameters
    ) throws {
        let cache = try model.newCache(parameters: parameters)
        let input = LMInput.Text(tokens: MLXArray(tokens.map { Int32($0) }))
        let output = model(input[text: .newAxis], cache: cache, state: nil)
        eval(output.logits, cache)
        Stream.gpu.synchronize()
        _ = output.logits[0, -1, 0].item(Float.self)
    }

    private func prepareDecodeState(
        model: any LanguageModel,
        tokens: [Int],
        parameters: GenerateParameters
    ) throws -> DecodeState {
        let cache = try model.newCache(parameters: parameters)
        var state: LMOutput.State?
        let input = LMInput.Text(tokens: MLXArray(tokens.map { Int32($0) }))
        let output = model(input[text: .newAxis], cache: cache, state: state)
        state = output.state
        let currentLogits = output.logits[0..., -1, 0...]
        let currentToken = argMax(currentLogits, axis: -1)
        eval(currentToken, currentLogits, cache)
        Stream.gpu.synchronize()
        _ = currentToken.item(Int.self)
        return DecodeState(
            cache: cache,
            state: state,
            currentToken: currentToken,
            currentLogits: currentLogits
        )
    }

    private func decode(
        model: any LanguageModel,
        state: inout DecodeState,
        decodeTokens: Int
    ) throws {
        for _ in 0 ..< decodeTokens {
            let stepOutput = model(
                LMInput.Text(tokens: state.currentToken[.newAxis, 0...]),
                cache: state.cache,
                state: state.state
            )
            state.state = stepOutput.state
            state.currentLogits = stepOutput.logits[0..., -1, 0...]
            state.currentToken = argMax(state.currentLogits, axis: -1)
            eval(state.currentToken, state.currentLogits, state.cache)
        }
        Stream.gpu.synchronize()
        _ = state.currentToken.item(Int.self)
    }

    private func ttft(
        model: any LanguageModel,
        tokens: [Int],
        parameters: GenerateParameters,
        prefixStore: (any PrefixKVStore)?
    ) async throws {
        let engine = MLXCatEngine(
            model: model,
            parameters: GenerateParameters(maxTokens: 1, temperature: parameters.temperature),
            maxConcurrentRequests: 1,
            prefixStore: prefixStore
        )
        let uid = "ttft-\(UUID().uuidString)"
        try await engine.submit(
            Request(
                uid: uid,
                input: input(tokens),
                maxTokens: 1,
                sampling: SamplingParameters(temperature: 0)
            )
        )

        var sawFirstToken = false
        while !sawFirstToken {
            let responses = try await engine.step()
            sawFirstToken = responses.contains { $0.uid == uid && $0.token >= 0 }
        }
    }

    private func cacheSpeedup(
        model: any LanguageModel,
        prefixTokens: [Int],
        warmSuffix: [Int],
        measuredSuffix: [Int],
        parameters: GenerateParameters
    ) async throws -> CacheBenchmarkResult {
        let coldPrompt = prefixTokens + measuredSuffix
        let coldSeconds = try await measureManyAsync {
            try await ttft(model: model, tokens: coldPrompt, parameters: parameters, prefixStore: nil)
        }

        let manager = PagedCacheManager(blockSize: blockSize)
        let prefixCache = BlockAwarePrefixCache(
            modelName: modelName,
            blockSize: blockSize,
            manager: manager
        )
        let prefixStore = BlockAwarePrefixKVStore(prefixCache: prefixCache)
        let warmEngine = MLXCatEngine(
            model: model,
            parameters: parameters,
            maxConcurrentRequests: 1,
            prefixStore: prefixStore
        )

        _ = try await warmEngine.generate([
            Request(
                uid: "cache-warm-\(UUID().uuidString)",
                input: input(prefixTokens + warmSuffix),
                maxTokens: 1,
                sampling: SamplingParameters(temperature: 0)
            )
        ])

        let warmSeconds = try await measureManyAsync {
            try await ttft(model: model, tokens: prefixTokens + measuredSuffix, parameters: parameters, prefixStore: prefixStore)
        }

        let coldMedian = median(coldSeconds)
        let warmMedian = median(warmSeconds)
        return CacheBenchmarkResult(
            coldMedianSeconds: coldMedian,
            warmMedianSeconds: warmMedian,
            speedup: coldMedian / max(warmMedian, 1e-9),
            fetchHits: prefixStore.fetchHitCount
        )
    }

    private func throughput(
        model: any LanguageModel,
        prompts: [[Int]],
        parameters: GenerateParameters,
        concurrency: Int
    ) async throws {
        let engine = MLXCatEngine(
            model: model,
            parameters: parameters,
            maxConcurrentRequests: concurrency
        )
        let requests = prompts.enumerated().map { index, tokens in
            Request(
                uid: "throughput-\(concurrency)-\(index)-\(UUID().uuidString)",
                input: input(tokens),
                maxTokens: config.decodeTokens,
                sampling: SamplingParameters(temperature: 0)
            )
        }
        _ = try await engine.generate(requests)
    }

    private func tokenIDs(
        for text: String,
        context: ModelContext,
        addSpecialTokens: Bool = true
    ) -> [Int] {
        context.tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    private func input(_ tokens: [Int]) -> LMInput {
        LMInput(text: LMInput.Text(tokens: MLXArray(tokens.map { Int32($0) })))
    }
}

private struct NativeBenchmarkResult {
    let modelName: String
    let runs: Int
    let warmup: Int
    let promptDescription: String
    let promptTokens: Int
    let decodeTokens: Int
    let prefillTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let ttftMilliseconds: Double
    let coldTTFTMilliseconds: Double
    let warmTTFTMilliseconds: Double
    let cacheSpeedup: Double
    let cacheFetchHits: Int
    let concurrency: [ConcurrencyBenchmarkResult]

    func markdown() -> String {
        var lines: [String] = []
        lines.append("# MLXCat Native Benchmark")
        lines.append("")
        lines.append("- Model: `\(modelName)`")
        lines.append("- Runs: \(runs) timed, \(warmup) warmup")
        lines.append("- Prompts: \(promptDescription)")
        lines.append("- Prompt tokens for PP/cache: \(promptTokens)")
        lines.append("- Decode tokens per request: \(decodeTokens)")
        lines.append("")
        lines.append("| Metric | Median |")
        lines.append("| --- | ---: |")
        lines.append("| Prefill PP | \(format(prefillTokensPerSecond)) tok/s |")
        lines.append("| Decode TG | \(format(decodeTokensPerSecond)) tok/s |")
        lines.append("| TTFT | \(format(ttftMilliseconds)) ms |")
        lines.append("| Cold prefill TTFT | \(format(coldTTFTMilliseconds)) ms |")
        lines.append("| Warm prefix restore TTFT | \(format(warmTTFTMilliseconds)) ms |")
        lines.append("| Prefix cache speedup | \(format(cacheSpeedup))x |")
        lines.append("| Prefix fetch hits during cache bench | \(cacheFetchHits) |")
        lines.append("")
        lines.append("| Concurrency | Generated Tokens | Median Time | Throughput |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for result in concurrency {
            lines.append(
                "| \(result.concurrency) | \(result.generatedTokens) | \(format(result.medianSeconds)) s | \(format(result.throughputTokensPerSecond)) tok/s |"
            )
        }
        return lines.joined(separator: "\n")
    }

    private func format(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

private struct ConcurrencyBenchmarkResult {
    let concurrency: Int
    let generatedTokens: Int
    let medianSeconds: Double
    let throughputTokensPerSecond: Double
}

private struct CacheBenchmarkResult {
    let coldMedianSeconds: Double
    let warmMedianSeconds: Double
    let speedup: Double
    let fetchHits: Int
}

private struct DecodeState {
    var cache: [any KVCache]
    var state: LMOutput.State?
    var currentToken: MLXArray
    var currentLogits: MLXArray
}

private enum BenchError: Error, CustomStringConvertible {
    case invalidArgument(String)
    case unexpectedPreparedLogits

    var description: String {
        switch self {
        case .invalidArgument(let message):
            return message
        case .unexpectedPreparedLogits:
            return "model.prepare returned logits; token prompts are required for this benchmark"
        }
    }
}

private func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    guard !sorted.isEmpty else { return 0 }
    let middle = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[middle - 1] + sorted[middle]) / 2
    }
    return sorted[middle]
}

private extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async throws -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self {
            result.append(try await transform(element))
        }
        return result
    }
}
