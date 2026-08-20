import Foundation
@testable import MLXCatHTTP
@testable import MLXCatHTTPServer
@testable import MLXCatSpeech
@testable import MLXCatSpeechWhisperKit
import XCTest

/// Live WhisperKit tests — real CoreML models, real audio. Gated: they need the
/// argmaxinc/whisperkit-coreml models directory (present on dev machines that
/// have run the app's Whisper stack) and ANE/GPU access, so they skip cleanly
/// elsewhere. Set MLXSERVE_WHISPERKIT_LIVE=0 to force-skip.
final class WhisperKitAdapterLiveTests: XCTestCase {
    static let modelsRoot = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Documents/huggingface/models/argmaxinc/whisperkit-coreml")
    static let liveModel = "openai_whisper-tiny"

    private func requireLiveEnvironment() throws -> URL {
        if ProcessInfo.processInfo.environment["MLXSERVE_WHISPERKIT_LIVE"] == "0" {
            throw XCTSkip("MLXSERVE_WHISPERKIT_LIVE=0 — live WhisperKit tests disabled.")
        }
        let modelFolder = Self.modelsRoot.appendingPathComponent(Self.liveModel)
        guard FileManager.default.fileExists(atPath: modelFolder.path) else {
            throw XCTSkip("WhisperKit model \(Self.liveModel) not present at \(Self.modelsRoot.path).")
        }
        return Self.modelsRoot
    }

    private func fixtureData() throws -> Data {
        guard let url = Bundle.module.url(forResource: "test_speech", withExtension: "wav") else {
            throw XCTSkip("test_speech.wav fixture missing from bundle.")
        }
        return try Data(contentsOf: url)
    }

    func testCatalogListsLocalModels() async throws {
        let root = try requireLiveEnvironment()
        let adapter = WhisperKitSpeechAdapter(modelsRoot: root)
        let models = await adapter.availableModels()
        XCTAssertTrue(models.contains { $0.id == Self.liveModel })
        XCTAssertTrue(models.allSatisfy { $0.engineID == "whisperkit" })
    }

    func testFileTranscriptionProducesTextAndWordTimings() async throws {
        let root = try requireLiveEnvironment()
        let adapter = WhisperKitSpeechAdapter(modelsRoot: root)

        let result = try await adapter.transcribeFile(
            SpeechFileTranscriptionRequest(
                model: Self.liveModel,
                fileName: "test_speech.wav",
                fileData: try fixtureData(),
                language: "en"
            )
        )

        XCTAssertFalse(result.text.isEmpty, "expected a non-empty transcript")
        let words = try XCTUnwrap(result.words)
        XCTAssertFalse(words.isEmpty)
        for word in words {
            XCTAssertGreaterThanOrEqual(word.end, word.start)
        }
        XCTAssertEqual(
            words.map(\.start),
            words.map(\.start).sorted(),
            "word start times must be monotone"
        )
    }

    func testUnknownModelThrowsWithoutDownload() async throws {
        let root = try requireLiveEnvironment()
        let adapter = WhisperKitSpeechAdapter(modelsRoot: root, allowDownload: false)
        do {
            _ = try await adapter.transcribeFile(
                SpeechFileTranscriptionRequest(model: "ghost-model", fileName: "x.wav", fileData: Data())
            )
            XCTFail("expected unknownModel")
        } catch let error as SpeechEngineError {
            XCTAssertEqual(error, .unknownModel("ghost-model"))
        }
    }

    func testStreamSessionEmitsPartialsThenFinal() async throws {
        let root = try requireLiveEnvironment()
        let adapter = WhisperKitSpeechAdapter(modelsRoot: root)
        let session = try await adapter.makeStreamSession(modelID: Self.liveModel, language: "en")

        let pcm = try Self.decodePCM(from: try fixtureData())
        // Push in ~0.5s chunks so at least one partial fires (interval 2s) before finish.
        let chunk = 8_000
        var index = 0
        while index < pcm.count {
            let slice = Array(pcm[index..<min(index + chunk, pcm.count)])
            try await session.push(SpeechPCMBuffer(samples: slice, sampleRate: 16_000))
            index += chunk
        }
        try await session.finish()

        var collected: [SpeechStreamSegment] = []
        for try await segment in session.segments {
            collected.append(segment)
        }

        XCTAssertEqual(collected.last?.kind, .final)
        XCTAssertFalse(collected.last?.text.isEmpty ?? true)
        XCTAssertEqual(collected.filter { $0.kind == .final }.count, 1)
        let stats = await session.latencyStats()
        XCTAssertGreaterThan(stats.sampleCount, 0)
        XCTAssertGreaterThan(stats.p50Seconds, 0)
    }

    func testRegistryBridgeServesWhisperKitEndToEnd() async throws {
        let root = try requireLiveEnvironment()
        let registry = SpeechEngineRegistry()
        await registry.register(WhisperKitSpeechAdapter(modelsRoot: root))
        let backend = await RegistrySpeechBackend(registry: registry)

        XCTAssertTrue(backend.transcriptionModels.contains { $0.id == Self.liveModel })

        let result = try await backend.transcribe(
            AudioTranscriptionRequest(
                model: Self.liveModel,
                fileName: "test_speech.wav",
                fileData: try fixtureData(),
                language: "en"
            )
        )
        XCTAssertFalse(result.text.isEmpty)
        XCTAssertNotNil(result.segments)
    }

    /// Minimal 16-bit mono WAV decode for the fixture — avoids an AVFoundation
    /// dependency in the test target.
    static func decodePCM(from wav: Data) throws -> [Float] {
        guard wav.count > 44 else { throw XCTSkip("fixture too small to be a WAV") }
        guard let dataRange = findChunk(id: "data", in: wav) else {
            throw XCTSkip("fixture WAV has no data chunk")
        }
        let payload = wav.subdata(in: dataRange)
        var samples = [Float]()
        samples.reserveCapacity(payload.count / 2)
        payload.withUnsafeBytes { raw in
            let int16Buffer = raw.bindMemory(to: Int16.self)
            for sample in int16Buffer {
                samples.append(Float(Int16(littleEndian: sample)) / Float(Int16.max))
            }
        }
        return samples
    }

    private static func findChunk(id: String, in data: Data) -> Range<Data.Index>? {
        let idBytes = Array(id.utf8)
        var index = 12  // past RIFF header
        while index + 8 <= data.count {
            let chunkID = Array(data[index..<index + 4])
            let size = data.subdata(in: index + 4..<index + 8).withUnsafeBytes {
                Int($0.loadUnaligned(as: UInt32.self).littleEndian)
            }
            let body = index + 8
            if chunkID == idBytes {
                let end = min(body + size, data.count)
                return body..<end
            }
            index = body + size + (size % 2)
        }
        return nil
    }
}
