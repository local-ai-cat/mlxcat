import Hub
import MLXCat
import MLXCatHTTP
import MLXCatNative
import os
import XCTest

/// Tool-call CORRECTNESS on a real iPhone — the axis the perf rows never
/// measured. Boots the real `OpenAIServer` (NWListener, loopback, ephemeral
/// port) over a single-model `EnginePool` inside the host app, then replays
/// the same scenario suite the Mac campaigns use (`toolcall-scenarios.json`,
/// dumped from `bench/toolcalling/run.py --dump-scenarios` — one source of
/// truth) and validates the emitted calls with the same rules.
///
/// Rows land as TOOLROW console lines (crash-safe) and as an XCTAttachment
/// named `mlxcat-toolcall-ios.jsonl` (schema `mlxcat-toolcall/1`, plus
/// platform/device stamps).
///
/// Config via the xcodebuild ENVIRONMENT (TEST_RUNNER_ prefix):
///   BENCHHOST_MODEL_ID   HF repo id or comma-separated list
///   BENCHHOST_TRIALS     trials per (scenario x temperature), default 3
///   BENCHHOST_TEMPS      comma list, default "0,0.7"
final class DeviceToolCallTests: XCTestCase {

    private struct Scenario: Decodable {
        let tools: [JSONValue]
        let messages: [JSONValue]
        let toolChoice: JSONValue?
        let callExpected: Bool
        let maxTokens: Int?
        let minCalls: Int?
        let minArgChars: Int?
        let expectAnswerContains: String?

        enum CodingKeys: String, CodingKey {
            case tools, messages
            case toolChoice = "tool_choice"
            case callExpected = "call_expected"
            case maxTokens = "max_tokens"
            case minCalls = "min_calls"
            case minArgChars = "min_arg_chars"
            case expectAnswerContains = "expect_answer_contains"
        }
    }

    /// Minimal JSON tree that round-trips scenario fragments untouched.
    private enum JSONValue: Codable {
        case null
        case bool(Bool)
        case number(Double)
        case string(String)
        case array([JSONValue])
        case object([String: JSONValue])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if c.decodeNil() { self = .null }
            else if let v = try? c.decode(Bool.self) { self = .bool(v) }
            else if let v = try? c.decode(Double.self) { self = .number(v) }
            else if let v = try? c.decode(String.self) { self = .string(v) }
            else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
            else { self = .object(try c.decode([String: JSONValue].self)) }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .null: try c.encodeNil()
            case .bool(let v): try c.encode(v)
            case .number(let v):
                if v == v.rounded() && abs(v) < 1e15 { try c.encode(Int64(v)) }
                else { try c.encode(v) }
            case .string(let v): try c.encode(v)
            case .array(let v): try c.encode(v)
            case .object(let v): try c.encode(v)
            }
        }

        var any: Any {
            switch self {
            case .null: return NSNull()
            case .bool(let v): return v
            case .number(let v):
                if v == v.rounded() && abs(v) < 1e15 { return Int(v) }
                return v
            case .string(let v): return v
            case .array(let v): return v.map(\.any)
            case .object(let v): return v.mapValues(\.any)
            }
        }
    }

    // MARK: - the minimal schema validator (mirror of run.py's schema_valid)

    private static func schemaValid(_ value: Any, schema: [String: Any]) -> Bool {
        switch schema["type"] as? String {
        case "object":
            guard let dict = value as? [String: Any] else { return false }
            for key in schema["required"] as? [String] ?? [] where dict[key] == nil {
                return false
            }
            let props = schema["properties"] as? [String: [String: Any]] ?? [:]
            if schema["additionalProperties"] as? Bool == false {
                if dict.keys.contains(where: { props[$0] == nil }) { return false }
            }
            return dict.allSatisfy { key, v in
                props[key].map { schemaValid(v, schema: $0) } ?? true
            }
        case "array":
            guard let arr = value as? [Any] else { return false }
            let items = schema["items"] as? [String: Any] ?? [:]
            return arr.allSatisfy { schemaValid($0, schema: items) }
        case "string":
            guard let s = value as? String else { return false }
            if let allowed = schema["enum"] as? [String] { return allowed.contains(s) }
            return true
        case "integer":
            return value is Int || (value as? NSNumber).map {
                CFNumberIsFloatType($0) == false && !($0 === kCFBooleanTrue || $0 === kCFBooleanFalse)
            } ?? false
        case "number":
            return value is Int || value is Double
        case "boolean":
            return value is Bool
        default:
            return true
        }
    }

    func testToolCallCorrectness() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("MLX needs a real Apple Silicon iPhone.")
        #else
        let environment = ProcessInfo.processInfo.environment
        let modelIDs = (environment["BENCHHOST_MODEL_ID"] ?? "mlx-community/Qwen3-1.7B-4bit")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let trials = Int(environment["BENCHHOST_TRIALS"] ?? "") ?? 3
        let temps = (environment["BENCHHOST_TEMPS"] ?? "0,0.7")
            .split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }

        guard let scenariosURL = Bundle(for: Self.self)
            .url(forResource: "toolcall-scenarios", withExtension: "json") else {
            XCTFail("toolcall-scenarios.json missing from the test bundle")
            return
        }
        let scenarios = try JSONDecoder().decode(
            [String: Scenario].self, from: Data(contentsOf: scenariosURL))

        let deviceModel = Self.hardwareModel()
        var lines: [String] = []

        for modelID in modelIDs {
            print("TOOLHOST model \(modelID) starting")
            let hub = HubApi()
            let modelDirectory = try await hub.snapshot(
                from: modelID,
                matching: ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]
            )
            let shortID = URL(fileURLWithPath: modelID).lastPathComponent

            let availableBytes = Int64(os_proc_available_memory())
            let ceiling = availableBytes > 0 ? Int64(Double(availableBytes) * 0.8) : 4 << 30
            let weightBytes = (try? FileManager.default
                .contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: [.fileSizeKey])
                .reduce(Int64(0)) { sum, url in
                    sum + Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
                }) ?? 0
            let pool = EnginePool(
                models: [shortID: DiscoveredModel(
                    id: shortID, modelURL: modelDirectory, estimatedSize: weightBytes)],
                loader: NativeModelLoader(maxConcurrentRequests: 1),
                finalCeiling: ceiling,
                idleTimeout: nil
            )
            let backend = PoolBackedChatBackend(
                pool: pool,
                modelIDs: [shortID],
                embeddingsBackend: nil,
                rerankBackend: nil,
                speechBackend: nil,
                memoryWatchdog: nil
            )
            let server = try OpenAIServer(host: "127.0.0.1", port: 0, backend: backend)
            try await server.start()
            guard let port = server.boundPort else {
                XCTFail("server bound no port")
                return
            }
            print("TOOLHOST serving \(shortID) on 127.0.0.1:\(port)")
            defer { server.stop() }

            for (name, scenario) in scenarios.sorted(by: { $0.key < $1.key }) {
                for temp in temps {
                    for trial in 0..<trials {
                        let row = try await Self.runTrial(
                            port: port, model: shortID, scenarioName: name,
                            scenario: scenario, temperature: temp)
                        var stamped = row
                        stamped["schema"] = "mlxcat-toolcall/1"
                        stamped["platform"] = "ios"
                        stamped["engine"] = "mlxcat-openaiserver-ondevice"
                        stamped["device"] = ["model": deviceModel]
                        stamped["trial"] = trial
                        stamped["timestamp"] = ISO8601DateFormatter().string(from: Date())
                        guard let data = try? JSONSerialization.data(
                            withJSONObject: stamped, options: [.sortedKeys]),
                            let line = String(data: data, encoding: .utf8) else { continue }
                        lines.append(line)
                        print("TOOLROW \(line)")
                        let okText = (stamped["ok"] as? Bool) == true ? "ok " : "FAIL"
                        print("TOOLHOST \(okText) \(shortID) t=\(temp) \(name)")
                    }
                }
            }
        }

        XCTAssertFalse(lines.isEmpty, "no tool-call rows produced")
        let attachment = XCTAttachment(
            data: Data((lines.joined(separator: "\n") + "\n").utf8),
            uniformTypeIdentifier: "public.plain-text")
        attachment.name = "mlxcat-toolcall-ios.jsonl"
        attachment.lifetime = .keepAlways
        add(attachment)
        #endif
    }

    private static func runTrial(
        port: UInt16, model: String, scenarioName: String,
        scenario: Scenario, temperature: Double
    ) async throws -> [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "messages": scenario.messages.map(\.any),
            "tools": scenario.tools.map(\.any),
            "max_tokens": scenario.maxTokens ?? 1024,
            "temperature": temperature,
        ]
        if let choice = scenario.toolChoice { body["tool_choice"] = choice.any }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 600

        var row: [String: Any] = ["scenario": scenarioName, "temperature": temperature]
        let started = Date()
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            row["error"] = String(describing: error)
            row["ok"] = false
            return row
        }
        row["wall_s"] = Date().timeIntervalSince(started)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            row["error"] = "HTTP \(http.statusCode): " + (String(data: data, encoding: .utf8) ?? "")
                .prefix(200).description
            row["ok"] = false
            return row
        }

        guard let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choice = (payload["choices"] as? [[String: Any]])?.first else {
            row["error"] = "unparseable response"
            row["ok"] = false
            return row
        }
        let message = choice["message"] as? [String: Any] ?? [:]
        let content = message["content"] as? String ?? ""
        let calls = message["tool_calls"] as? [[String: Any]] ?? []
        row["finish_reason"] = choice["finish_reason"] as? String ?? NSNull()
        row["raw_content"] = String(content.prefix(1500))
        row["raw_tool_calls"] = calls
        row["called"] = !calls.isEmpty
        row["n_calls"] = calls.count

        guard let expectedTool = (scenario.tools.first?.any as? [String: Any])?["function"]
            as? [String: Any],
            let expectedName = expectedTool["name"] as? String else {
            row["error"] = "scenario has no tool"
            row["ok"] = false
            return row
        }
        let parameters = expectedTool["parameters"] as? [String: Any] ?? [:]

        var jsonOK = !calls.isEmpty
        var schemaOK = !calls.isEmpty
        var namesOK = !calls.isEmpty
        var parsedArgs: [[String: Any]] = []
        for call in calls {
            let fn = call["function"] as? [String: Any] ?? [:]
            if fn["name"] as? String != expectedName { namesOK = false }
            guard let argsData = (fn["arguments"] as? String)?.data(using: .utf8),
                  let args = try? JSONSerialization.jsonObject(with: argsData)
                    as? [String: Any] else {
                jsonOK = false
                schemaOK = false
                continue
            }
            parsedArgs.append(args)
            if !schemaValid(args, schema: parameters) { schemaOK = false }
        }

        var ok: Bool
        if scenario.callExpected {
            ok = !calls.isEmpty && jsonOK && schemaOK && namesOK
            if let minCalls = scenario.minCalls { ok = ok && calls.count >= minCalls }
            if let minChars = scenario.minArgChars, !parsedArgs.isEmpty {
                let longest = parsedArgs.flatMap { $0.values.map { "\($0)".count } }.max() ?? 0
                row["longest_arg_chars"] = longest
                ok = ok && longest >= minChars
            }
        } else {
            ok = calls.isEmpty
            if let needle = scenario.expectAnswerContains {
                ok = ok && content.contains(needle)
            }
        }
        row["json_valid"] = jsonOK
        row["schema_valid"] = schemaOK
        row["name_correct"] = namesOK
        row["ok"] = ok
        return row
    }

    private static func hardwareModel() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafeBytes(of: &systemInfo.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}
