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
            if let rawDepth = environment["MLXCAT_INKLING_PROBE_DRAFT"], let depth = Int(rawDepth), depth >= 0 {
                let steps = environment["MLXCAT_INKLING_DRAFT_STEPS"].flatMap(Int.init).flatMap { $0 > 0 ? $0 : nil }
                let promptPath = environment["MLXCAT_INKLING_PROFILE_PROMPT_FILE"]
                let target: LMInput
                if let promptPath {
                    let prompt = try String(contentsOfFile: promptPath, encoding: .utf8)
                    target = try runtime.prepare(UserInput(
                        chat: [.user(prompt)], additionalContext: ["reasoning_effort": "none"]))
                } else {
                    target = input
                }
                let expected = Array(fixture.generatedIDs.prefix(64))
                let result = try runtime.greedyDecode(
                    in: target, chunk: fixture.prefillChunk, steps: steps ?? expected.count, draftDepth: depth)
                print("\(name) decode: \(target.text.tokens.size) prompt, \(result.tokens.count) output, "
                    + "\(Double(result.tokens.count) / result.seconds) tok/s, "
                    + "\(result.accepted)/\(result.proposed) accepted")
                print("Token IDs: \(result.tokens)")
                if steps == nil && promptPath == nil {
                    let matching = zip(result.tokens, expected).prefix(while: { $0.0 == $0.1 }).count
                    print("\(name): \(matching)/\(expected.count) speculative greedy tokens")
                    passed = passed && matching == expected.count && result.tokens.count == expected.count
                }
                continue
            }
            if let rawSteps = environment["MLXCAT_INKLING_PROFILE_STEPS"],
                let steps = Int(rawSteps), steps > 0
            {
                let profileInput: LMInput
                if let promptPath = environment["MLXCAT_INKLING_PROFILE_PROMPT_FILE"] {
                    let prompt = try String(contentsOfFile: promptPath, encoding: .utf8)
                    profileInput = try runtime.prepare(UserInput(
                        chat: [.user(prompt)], additionalContext: ["reasoning_effort": "none"]))
                } else {
                    profileInput = input
                }
                print("Inkling decode probe input tokens \(profileInput.text.tokens.size)")
                let profile = try runtime.profileDecode(
                    in: profileInput, chunk: fixture.prefillChunk, steps: steps)
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
                let prefix = environment["MLXCAT_INKLING_VERIFY_PREFIX"].flatMap(Int.init)
                let panelInput: LMInput
                if let prefix, prefix > 0, prefix < input.text.tokens.size, input.image == nil {
                    panelInput = LMInput(tokens: input.text.tokens.reshaped(-1)[..<prefix])
                } else {
                    panelInput = input
                }
                let proof = try runtime.compareVerificationPanel(
                    panel, in: panelInput, chunk: fixture.prefillChunk)
                print("\(name): panel vs serial over \(panel.count) rows, max error "
                    + "\(proof.maximumLogitError), top-token differences \(proof.differentTopTokens), "
                    + "nonidentical logit rows \(proof.differentLogitRows), cache exact \(proof.cacheExact)")
                if prefix != nil { continue }
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
