import CryptoKit
import Foundation
import MLX
import MLXCatNative
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

struct InklingReferenceProbe {
    private struct Fixture: Decodable {
        struct TopLogit: Decodable {
            let id: Int
            let logit: Float
        }

        let inputIDs: [Int]
        let generatedIDs: [Int]
        let prefillChunk: Int
        let topLogits: [[TopLogit]]
        let imageSHA256: String?
        let messages: MLXLMCommon.JSONValue
        let settings: [String: MLXLMCommon.JSONValue]

        enum CodingKeys: String, CodingKey {
            case inputIDs = "input_ids"
            case generatedIDs = "generated_ids"
            case prefillChunk = "prefill_chunk_tokens"
            case topLogits = "top_logits"
            case imageSHA256 = "image_sha256"
            case messages
            case settings
        }
    }

    static func run() async throws -> Bool {
        let environment = ProcessInfo.processInfo.environment
        guard let checkpoint = environment["MLXCAT_INKLING_VERIFIED_MODEL"],
            let reference = environment["MLXCAT_INKLING_REFERENCE"]
        else { throw InputError.missingPaths }
        let directory = URL(fileURLWithPath: checkpoint, isDirectory: true)
        let fixtures = URL(fileURLWithPath: reference, isDirectory: true)
        let selected = environment["MLXCAT_INKLING_CASE"]
        let prefix = directory.lastPathComponent + "-"
        let selectedFile = selected.map { prefix + $0 }
        let paths = try FileManager.default.contentsOfDirectory(at: fixtures, includingPropertiesForKeys: nil)
            .filter {
                $0.pathExtension == "json" && $0.lastPathComponent.hasPrefix(prefix)
                    && (selectedFile == nil || $0.deletingPathExtension().lastPathComponent == selectedFile)
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard paths.count == (selected == nil ? 12 : 1) else { throw InputError.wrongCaseCount(paths.count) }

        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: directory)
        let runtime = try InklingReferenceRuntime(directory: directory, tokenizer: tokenizer)
        var passed = true
        let tolerance: Float = 0.25
        for path in paths {
            let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: path))
            let input = try runtime.prepare(try userInput(fixture, directory: fixtures))
            let name = path.lastPathComponent
            guard input.text.tokens.asArray(Int.self) == fixture.inputIDs else {
                print("\(name): formatted input token IDs differ")
                passed = false
                continue
            }
            if let rawSteps = environment["MLXCAT_INKLING_PROFILE_STEPS"],
                let steps = Int(rawSteps), steps > 0
            {
                let profile = try runtime.profileDecode(in: input, chunk: fixture.prefillChunk, steps: steps)
                print("\(name) decode over \(steps) tokens: graph \(profile.buildMs) ms, submit "
                    + "\(profile.submitMs) ms, readback \(profile.readMs) ms")
                print("Token IDs: \(profile.tokens)")
                continue
            }
            let expected = Array(fixture.generatedIDs.prefix(64))
            guard expected.count == 64 || expected.last == runtime.endToken else {
                print("\(name): the reference terminated before 64 tokens without EOS")
                passed = false
                continue
            }
            if environment["MLXCAT_INKLING_PROBE_VERIFY_PANEL"] == "1" {
                let panel = Array(expected.prefix(8))
                let proof = try runtime.compareVerificationPanel(
                    panel, in: input, chunk: fixture.prefillChunk)
                print("\(name): panel vs serial over \(panel.count) rows, max error "
                    + "\(proof.maximumLogitError), top-token differences \(proof.differentTopTokens)")
            }
            let actual = try runtime.compare(
                expected, in: input, chunk: fixture.prefillChunk,
                topTokenIDs: fixture.topLogits.map { $0.map(\.id) })
            let matching = zip(actual.tokens, expected).prefix(while: { $0.0 == $0.1 }).count
            let errors = zip(actual.scores, fixture.topLogits).flatMap { actual, expected in
                zip(actual, expected).map { abs($0 - $1.logit) }
            }.sorted()
            guard let maximum = errors.last, errors.allSatisfy(\.isFinite) else {
                print("\(name): missing or non-finite logit scores")
                passed = false
                continue
            }
            let median = errors[errors.count / 2]
            print("\(name): \(matching)/\(expected.count) greedy tokens, max error \(maximum), median \(median)")
            if matching != expected.count || maximum > tolerance {
                passed = false
            }
        }
        return passed
    }

    private static func userInput(_ fixture: Fixture, directory: URL) throws -> UserInput {
        let images: [UserInput.Image]
        if let hash = fixture.imageSHA256 {
            let image = directory.appending(component: "shapes.png")
            let digest = SHA256.hash(data: try Data(contentsOf: image))
                .map { String(format: "%02x", $0) }.joined()
            guard digest == hash else { throw InputError.imageHash }
            images = [.url(image)]
        } else {
            images = []
        }
        let settings = fixture.settings.mapValues(sendableValue)
        let tools = settings["tools"] as? [[String: any Sendable]]
        switch fixture.messages {
        case .string(let prompt):
            return UserInput(chat: [.user(prompt, images: images)], tools: tools, additionalContext: settings)
        case .array(let rows):
            let messages = rows.compactMap { sendableValue($0) as? [String: any Sendable] }
            guard messages.count == rows.count else { throw InputError.messageFormat }
            return UserInput(messages: messages, images: images, tools: tools, additionalContext: settings)
        default:
            throw InputError.messageFormat
        }
    }

    private enum InputError: Error {
        case missingPaths
        case wrongCaseCount(Int)
        case imageHash
        case messageFormat
    }

    private static func sendableValue(_ value: MLXLMCommon.JSONValue) -> any Sendable {
        switch value {
        case .null: NSNull()
        case .bool(let value): value
        case .int(let value): value
        case .double(let value): value
        case .string(let value): value
        case .array(let values): values.map(sendableValue)
        case .object(let values): values.mapValues(sendableValue)
        }
    }

}

do {
    let passed = try await InklingReferenceProbe.run()
    if !passed { exit(1) }
} catch {
    print("Inkling reference verification failed: \(error)")
    exit(1)
}
