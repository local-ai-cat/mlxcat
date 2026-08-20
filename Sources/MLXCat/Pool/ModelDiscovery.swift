import Foundation

public struct DiscoveredModel: Equatable, Sendable {
    public let id: String
    public let modelURL: URL
    public let estimatedSize: Int64

    public init(id: String, modelURL: URL, estimatedSize: Int64) {
        self.id = id
        self.modelURL = modelURL
        self.estimatedSize = estimatedSize
    }
}

public enum ModelDiscoveryError: Error, CustomStringConvertible, Equatable {
    case directoryNotFound(String)
    case noModelWeights(String)

    public var description: String {
        switch self {
        case .directoryNotFound(let path):
            return "model directory not found: \(path)"
        case .noModelWeights(let path):
            return "No model weights found in \(path)"
        }
    }
}

public enum ModelDiscovery {
    private static let overheadFactor = 1.05

    public static func discoverModels(in root: URL) throws -> [String: DiscoveredModel] {
        guard isDirectory(root) else {
            throw ModelDiscoveryError.directoryNotFound(root.path)
        }

        if isModelDirectory(root), let model = try discoveredModel(at: root, id: root.lastPathComponent) {
            return [model.id: model]
        }

        var models: [String: DiscoveredModel] = [:]
        let children = directoryChildren(root)
        for child in children where isDirectory(child) && !child.lastPathComponent.hasPrefix(".") {
            if isModelDirectory(child) {
                if let model = try discoveredModel(at: child, id: child.lastPathComponent) {
                    models[model.id] = model
                }
                continue
            }

            let grandchildren = directoryChildren(child)
            for grandchild in grandchildren where isDirectory(grandchild) && !grandchild.lastPathComponent.hasPrefix(".") {
                guard isModelDirectory(grandchild) else { continue }
                if let model = try discoveredModel(at: grandchild, id: grandchild.lastPathComponent) {
                    models[model.id] = model
                }
            }
        }

        return models
    }

    public static func estimateModelSize(at modelURL: URL) throws -> Int64 {
        var totalSize = sizeOfFiles(matching: "safetensors", in: modelURL, recursive: false)

        if totalSize == 0 {
            totalSize = directoryChildren(modelURL)
                .filter { $0.pathExtension == "bin" }
                .filter { url in
                    let name = url.lastPathComponent.lowercased()
                    return !name.contains("optimizer") && !name.contains("training")
                }
                .reduce(Int64(0)) { partial, url in
                    partial + fileSize(url)
                }
        }

        if totalSize == 0 {
            totalSize = sizeOfFiles(matching: "safetensors", in: modelURL, recursive: true)
        }

        guard totalSize > 0 else {
            throw ModelDiscoveryError.noModelWeights(modelURL.path)
        }

        return Int64((Double(totalSize) * overheadFactor).rounded(.down))
    }

    public static func formatSize(_ bytes: Int64) -> String {
        var value = Double(bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"] {
            if abs(value) < 1024.0 {
                return String(format: "%.2f%@", value, unit)
            }
            value /= 1024.0
        }
        return String(format: "%.2fPB", value)
    }

    private static func discoveredModel(at url: URL, id: String) throws -> DiscoveredModel? {
        do {
            return DiscoveredModel(
                id: id,
                modelURL: url,
                estimatedSize: try estimateModelSize(at: url)
            )
        } catch ModelDiscoveryError.noModelWeights {
            return nil
        }
    }

    private static func isModelDirectory(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path)
    }

    private static func isDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return false
        }
        return isDirectory.boolValue
    }

    private static func directoryChildren(_ url: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )) ?? []
    }

    private static func sizeOfFiles(matching pathExtension: String, in root: URL, recursive: Bool) -> Int64 {
        if !recursive {
            return directoryChildren(root)
                .filter { $0.pathExtension == pathExtension }
                .reduce(Int64(0)) { partial, url in
                    partial + fileSize(url)
                }
        }

        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var total: Int64 = 0
        for case let url as URL in enumerator where url.pathExtension == pathExtension {
            total += fileSize(url)
        }
        return total
    }

    private static func fileSize(_ url: URL) -> Int64 {
        guard
            let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
            let size = values.fileSize
        else {
            return 0
        }
        return Int64(size)
    }
}
