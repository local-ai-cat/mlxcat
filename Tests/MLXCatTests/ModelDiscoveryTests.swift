import Foundation
@testable import MLXCat
import XCTest

final class ModelDiscoveryTests: XCTestCase {
    func testDiscoversFarmDirectory() throws {
        let root = try temporaryDirectory()
        try makeModel(at: root.appendingPathComponent("alpha"), weightBytes: 100)
        try makeModel(at: root.appendingPathComponent("beta"), weightBytes: 200)

        let models = try ModelDiscovery.discoverModels(in: root)

        XCTAssertEqual(models.keys.sorted(), ["alpha", "beta"])
        XCTAssertEqual(models["alpha"]?.estimatedSize, 105)
        XCTAssertEqual(models["beta"]?.estimatedSize, 210)
    }

    func testSingleModelDirectoryBackCompatUsesDirectoryNameAsID() throws {
        let root = try temporaryDirectory().appendingPathComponent("single-model")
        try makeModel(at: root, weightBytes: 100)

        let models = try ModelDiscovery.discoverModels(in: root)

        XCTAssertEqual(models.keys.sorted(), ["single-model"])
        XCTAssertEqual(models["single-model"]?.modelURL.standardizedFileURL, root.standardizedFileURL)
    }

    func testEstimateModelSizeFallsBackToBinAndSkipsTrainingFiles() throws {
        let root = try temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        try Data(count: 200).write(to: root.appendingPathComponent("model.bin"))
        try Data(count: 1_000).write(to: root.appendingPathComponent("optimizer.bin"))
        try Data(count: 1_000).write(to: root.appendingPathComponent("training_state.bin"))

        let size = try ModelDiscovery.estimateModelSize(at: root)

        XCTAssertEqual(size, 210)
    }

    func testEstimateModelSizeFallsBackToRecursiveSafetensors() throws {
        let root = try temporaryDirectory()
        let shardDirectory = root.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: shardDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        try Data(count: 300).write(to: shardDirectory.appendingPathComponent("model.safetensors"))

        let size = try ModelDiscovery.estimateModelSize(at: root)

        XCTAssertEqual(size, 315)
    }

    func testFormatSizeMatchesReferenceStyle() {
        XCTAssertEqual(ModelDiscovery.formatSize(512), "512.00B")
        XCTAssertEqual(ModelDiscovery.formatSize(1_536), "1.50KB")
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxcat-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeModel(at url: URL, weightBytes: Int) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: url.appendingPathComponent("config.json"))
        try Data(count: weightBytes).write(to: url.appendingPathComponent("model.safetensors"))
    }
}
