import CryptoKit
import Foundation
import MLX

/// Optional persistence for named, text-only session snapshots. A checkpoint
/// file keeps all layer types and recurrent boundaries in one atomic payload.
package final class SessionPrefixDiskStore: @unchecked Sendable {
    struct Snapshot {
        let tokens: [Int]
        let layers: [SerializedKVLayer]
        let checkpoints: [PrefixRecurrentCheckpoint]
    }

    private struct Manifest: Codable {
        let version: Int
        let tokens: [Int]
        let layers: [Layer]
        let checkpoints: [Checkpoint]
    }

    private struct Layer: Codable {
        let names: [String]
        let metadata: [String]
        let type: String
    }

    private struct Checkpoint: Codable {
        let position: Int
        let layers: [Layer?]
    }

    private let directory: URL
    private let maxBytes: Int64
    private let writer = DispatchQueue(label: "mlxcat.session-prefix-disk")
    private let writeSlot = DispatchSemaphore(value: 1)

    package init?(root: URL, modelDirectory: URL, maxBytes: Int64 = 4 * 1024 * 1024 * 1024) {
        let model = modelDirectory.resolvingSymlinksInPath()
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey, .fileResourceIdentifierKey]
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: model, includingPropertiesForKeys: Array(keys))
        else { return nil }
        let files = contents.filter { ["json", "safetensors", "bin", "jinja"].contains($0.pathExtension) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard files.contains(where: { ["safetensors", "bin"].contains($0.pathExtension) }) else { return nil }
        var identity = model.path
        for file in files {
            let resolved = file.resolvingSymlinksInPath()
            guard let metadata = try? resolved.resourceValues(forKeys: keys),
                let size = metadata.fileSize, let modified = metadata.contentModificationDate,
                let identifier = metadata.fileResourceIdentifier
            else { return nil }
            identity += "\n\(file.lastPathComponent):\(size):\(modified.timeIntervalSince1970):\(identifier)"
        }
        let directory = root.appendingPathComponent(Self.digest(identity), isDirectory: true)
        guard (try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])) != nil
        else { return nil }
        self.directory = directory
        self.maxBytes = maxBytes
    }

    func read(session: String) -> Snapshot? {
        let url = file(session)
        guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize,
            Int64(size) <= maxBytes,
            let block = try? SafetensorsBlockIO.read(from: url),
            let text = block.metadata["manifest"],
            let manifest = try? JSONDecoder().decode(Manifest.self, from: Data(text.utf8)),
            manifest.version == 1
        else { return nil }

        func layer(_ descriptor: Layer) -> SerializedKVLayer? {
            let state = descriptor.names.compactMap { block.arrays[$0] }
            guard state.count == descriptor.names.count else { return nil }
            return SerializedKVLayer(state: state, metaState: descriptor.metadata, className: descriptor.type)
        }

        let layers = manifest.layers.compactMap(layer)
        guard layers.count == manifest.layers.count else { return nil }
        let checkpoints = manifest.checkpoints.map { checkpoint in
            PrefixRecurrentCheckpoint(
                position: checkpoint.position,
                layers: checkpoint.layers.map { $0.flatMap(layer) })
        }
        return Snapshot(tokens: manifest.tokens, layers: layers, checkpoints: checkpoints)
    }

    func store(session: String, snapshot: Snapshot, byteCount: Int64) {
        guard snapshot.tokens.count >= 256, byteCount < maxBytes,
            writeSlot.wait(timeout: .now()) == .success
        else { return }
        var tensors: [SafetensorsTensorSnapshot] = []
        func layer(_ value: SerializedKVLayer) -> Layer {
            let names = value.state.map { array in
                let name = "tensor.\(tensors.count)"
                tensors.append(SafetensorsBlockIO.snapshotTensor(name: name, array: array))
                return name
            }
            return Layer(names: names, metadata: value.metaState, type: value.className)
        }
        let manifest = Manifest(
            version: 1, tokens: snapshot.tokens, layers: snapshot.layers.map(layer),
            checkpoints: snapshot.checkpoints.map { checkpoint in
                Checkpoint(position: checkpoint.position, layers: checkpoint.layers.map { $0.map(layer) })
            })
        guard let encoded = try? JSONEncoder().encode(manifest) else {
            writeSlot.signal()
            return
        }
        let block = SafetensorsBlockFile(
            tensors: tensors, metadata: ["manifest": String(decoding: encoded, as: UTF8.self)])
        let destination = file(session)
        writer.async { [self] in
            defer { writeSlot.signal() }
            guard (try? SafetensorsBlockIO.write(block, to: destination)) != nil else { return }
            trim()
        }
    }

    private func file(_ session: String) -> URL {
        directory.appendingPathComponent(Self.digest(session)).appendingPathExtension("safetensors")
    }

    package func flushWrites() {
        writer.sync {}
    }

    func invalidate(session: String) {
        let url = file(session)
        writer.async {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private func trim() {
        let manager = FileManager.default
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]
        guard let urls = try? manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: Array(keys))
        else { return }
        let entries = urls.filter { $0.pathExtension == "safetensors" }.compactMap { url -> (URL, Int64, Date)? in
            guard let values = try? url.resourceValues(forKeys: keys),
                let size = values.fileSize, let date = values.contentModificationDate
            else { return nil }
            return (url, Int64(size), date)
        }.sorted { $0.2 < $1.2 }
        var stored = entries.reduce(Int64(0)) { $0 + $1.1 }
        for (url, size, _) in entries where stored > maxBytes {
            if (try? manager.removeItem(at: url)) != nil { stored -= size }
        }
    }

    private static func digest(_ string: String) -> String {
        SHA256.hash(data: Data(string.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
