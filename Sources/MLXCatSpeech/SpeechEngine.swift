import Foundation

// MARK: - Capabilities

public enum SpeechEngineSilicon: String, Sendable, Hashable {
    case ane
    case gpu
    case cpu
    case multiple
}

public struct SpeechEngineCapabilities: Sendable, Equatable {
    public let supportsFileTranscription: Bool
    public let supportsStreaming: Bool
    public let silicon: SpeechEngineSilicon
    public let wordTimestamps: Bool
    public let confidence: Bool
    /// BCP-47 codes the engine supports; nil = unrestricted / auto-detect.
    public let languages: Set<String>?

    public init(
        supportsFileTranscription: Bool,
        supportsStreaming: Bool,
        silicon: SpeechEngineSilicon,
        wordTimestamps: Bool = false,
        confidence: Bool = false,
        languages: Set<String>? = nil
    ) {
        self.supportsFileTranscription = supportsFileTranscription
        self.supportsStreaming = supportsStreaming
        self.silicon = silicon
        self.wordTimestamps = wordTimestamps
        self.confidence = confidence
        self.languages = languages
    }
}

// MARK: - Models

public struct SpeechModelInfo: Sendable, Equatable {
    public let id: String
    public let engineID: String
    public let displayName: String
    /// Bytes on disk/memory when loaded; nil = unknown (pool treats as unaccounted).
    public let footprint: Int64?

    public init(id: String, engineID: String, displayName: String, footprint: Int64? = nil) {
        self.id = id
        self.engineID = engineID
        self.displayName = displayName
        self.footprint = footprint
    }
}

// MARK: - Requests / results (file lane)

public struct SpeechFileTranscriptionRequest: Sendable {
    public let model: String
    public let fileName: String
    public let fileData: Data
    public let language: String?
    public let temperature: Float

    public init(
        model: String,
        fileName: String,
        fileData: Data,
        language: String? = nil,
        temperature: Float = 0
    ) {
        self.model = model
        self.fileName = fileName
        self.fileData = fileData
        self.language = language
        self.temperature = temperature
    }
}

public struct SpeechWord: Sendable, Equatable {
    public let text: String
    /// Engine-relative seconds from the start of the fed audio.
    public let start: Double
    public let end: Double
    public let confidence: Float?

    public init(text: String, start: Double, end: Double, confidence: Float? = nil) {
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
    }
}

public struct SpeechTranscriptionResult: Sendable, Equatable {
    public let text: String
    public let language: String?
    public let duration: Double?
    public let words: [SpeechWord]?

    public init(
        text: String,
        language: String? = nil,
        duration: Double? = nil,
        words: [SpeechWord]? = nil
    ) {
        self.text = text
        self.language = language
        self.duration = duration
        self.words = words
    }
}

// MARK: - Streaming lane

/// Mono float32 PCM. `captureTime` is the caller's clock (e.g. a studio session
/// clock) for the FIRST sample in the buffer; engines never interpret it — it is
/// echoed back so consumers can map engine-relative word times onto their clock.
public struct SpeechPCMBuffer: Sendable {
    public let samples: [Float]
    public let sampleRate: Double
    public let captureTime: Double?

    public init(samples: [Float], sampleRate: Double, captureTime: Double? = nil) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.captureTime = captureTime
    }

    public var duration: Double {
        Double(samples.count) / sampleRate
    }
}

public enum SpeechSegmentKind: String, Sendable {
    /// Provisional hypothesis; may be revised by a later segment with the same id.
    case partial
    /// Committed text; the id will not be revised again.
    case final
}

public struct SpeechStreamSegment: Sendable, Equatable {
    /// Stable identity: a `final` supersedes every `partial` with the same id.
    public let id: Int
    public let kind: SpeechSegmentKind
    public let text: String
    /// Engine-relative seconds over the audio fed so far.
    public let start: Double
    public let end: Double
    public let words: [SpeechWord]?

    public init(
        id: Int,
        kind: SpeechSegmentKind,
        text: String,
        start: Double,
        end: Double,
        words: [SpeechWord]? = nil
    ) {
        self.id = id
        self.kind = kind
        self.text = text
        self.start = start
        self.end = end
        self.words = words
    }
}

/// Word-emission latency relative to the audio time the word ends at — the
/// number a live-caption delay budget consumes.
public struct SpeechSessionLatencyStats: Sendable, Equatable {
    public let sampleCount: Int
    public let p50Seconds: Double
    public let p95Seconds: Double

    public init(sampleCount: Int, p50Seconds: Double, p95Seconds: Double) {
        self.sampleCount = sampleCount
        self.p50Seconds = p50Seconds
        self.p95Seconds = p95Seconds
    }

    public static let empty = SpeechSessionLatencyStats(sampleCount: 0, p50Seconds: 0, p95Seconds: 0)
}

/// One live transcription session: push timestamped PCM in, consume
/// partial/final segments out. Sessions are single-use.
public protocol SpeechStreamSession: Sendable {
    var segments: AsyncThrowingStream<SpeechStreamSegment, Error> { get }
    func push(_ buffer: SpeechPCMBuffer) async throws
    /// Flush: emits remaining finals, then ends the segment stream.
    func finish() async throws
    func cancel() async
    func latencyStats() async -> SpeechSessionLatencyStats
}

// MARK: - Adapter

public enum SpeechEngineError: Error, Equatable {
    case unknownModel(String)
    case fileTranscriptionUnsupported(engineID: String)
    case streamingUnsupported(engineID: String)
    case modelNotLoaded(String)
    case engineFailure(String)
}

/// One speech engine (WhisperKit, Parakeet, whisper.cpp, MLX-ASR, …) exposed as
/// a peer adapter. Adapters WRAP existing engine implementations; lane
/// mismatches (stream-only, file-only) are bridged inside the adapter, never by
/// consumers.
public protocol SpeechEngineAdapter: Sendable {
    var engineID: String { get }
    var displayName: String { get }
    var capabilities: SpeechEngineCapabilities { get }

    /// Models this adapter can serve right now (downloaded/installed).
    func availableModels() async -> [SpeechModelInfo]

    // Lifecycle hooks — phase 1 adapters may manage load internally and treat
    // these as advisory; the engine pool drives them in phase 2 (audio_stt).
    func loadModel(_ modelID: String) async throws
    func unloadModel(_ modelID: String) async
    func loadedFootprint() async -> Int64

    func transcribeFile(_ request: SpeechFileTranscriptionRequest) async throws -> SpeechTranscriptionResult
    func makeStreamSession(modelID: String, language: String?) async throws -> any SpeechStreamSession
}

public extension SpeechEngineAdapter {
    func loadModel(_ modelID: String) async throws {}
    func unloadModel(_ modelID: String) async {}
    func loadedFootprint() async -> Int64 { 0 }
}
