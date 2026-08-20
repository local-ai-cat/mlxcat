import Foundation

struct TestModelResolution {
    let url: URL
    let source: String
}

enum TestModelResolver {
    static func resolve() -> TestModelResolution? {
        let environment = ProcessInfo.processInfo.environment
        if let override = environment["MLXSERVE_TEST_MODEL"], !override.isEmpty {
            let url = URL(fileURLWithPath: override)
            if isMLXModelDirectory(url) {
                return TestModelResolution(url: url, source: "MLXSERVE_TEST_MODEL")
            }
        }

        for path in candidatePaths {
            let url = URL(fileURLWithPath: path)
            if isMLXModelDirectory(url) {
                return TestModelResolution(url: url, source: "local candidate")
            }
        }

        return nil
    }

    static func resolveVLM() -> TestModelResolution? {
        let environment = ProcessInfo.processInfo.environment
        guard let override = environment["MLXSERVE_VLM_TEST_MODEL"], !override.isEmpty else {
            return nil
        }

        let url = URL(fileURLWithPath: override)
        guard isMLXModelDirectory(url) else {
            return nil
        }
        return TestModelResolution(url: url, source: "MLXSERVE_VLM_TEST_MODEL")
    }

    private static let candidatePaths = [
        "/Users/timapple/Documents/Guest/Local AI Chat/BundledModels/mlc-chat-SmolLM-135M-4bit",
        "/Users/timapple/Documents/Guest/BundledModels/mlc-chat-SmolLM-135M-4bit",
        "/Users/timapple/Documents/BundledModels/mlc-chat-SmolLM-135M-4bit",
        "/Users/timapple/models/mlc-chat-SmolLM-135M-4bit",
    ]

    private static func isMLXModelDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            return false
        }

        let config = url.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: config.path) else {
            return false
        }

        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        for case let fileURL as URL in enumerator where fileURL.pathExtension == "safetensors" {
            return true
        }

        return false
    }
}
