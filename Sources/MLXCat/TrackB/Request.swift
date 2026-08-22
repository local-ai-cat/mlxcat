import MLXLMCommon

public struct Request: @unchecked Sendable {
    public let uid: String
    public let input: LMInput
    public let maxTokens: Int
    public let sampling: SamplingParameters
    public let eosTokenIds: Set<Int>
    public let cacheSession: String?

    public init(
        uid: String,
        input: LMInput,
        maxTokens: Int,
        sampling: SamplingParameters = SamplingParameters(),
        eosTokenIds: Set<Int> = [],
        cacheSession: String? = nil
    ) {
        self.uid = uid
        self.input = input
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.eosTokenIds = eosTokenIds
        self.cacheSession = cacheSession
    }
}

extension Request {
    /// Whether this request carries anything but text.
    ///
    /// This is the property that decides batchability for the families whose
    /// RoPE position ids come from a scalar `cache.offset`: an image, video or
    /// audio payload makes a row's token count differ from its neighbours', and
    /// a single scalar offset cannot describe both. Text rows of the same
    /// families batch correctly.
    public var isMultimodal: Bool {
        input.image != nil || input.video != nil || input.audio != nil
    }
}

/// When rows of a model family may share a decode batch.
///
/// `NativeModelLoader.usesSerializedDecode` used to answer this with a Bool, and
/// `true` meant "never batch anything for this model". For a VLM that is far
/// broader than the defect requires — it disables batching for text chat, which
/// is most traffic, to protect the image path. Measured on gemma-4-E2B
/// (2026-08-23): serialising everything costs 50x TTFT and 2.17x aggregate
/// throughput at c4, while ragged *image* rows genuinely do break (all three
/// rows stop at the same early token) and text rows are exact against serial.
public enum SerializationPolicy: Sendable, Equatable {
    /// Batch anything. The default for families with no scalar-offset defect.
    case never
    /// Never batch. The old `serializedDecode: true`.
    case always
    /// Batch text rows; give any row carrying image, video or audio the batch to
    /// itself. The protection the defect actually calls for.
    case multimodalOnly
    /// Batch, but never wider than this many rows.
    ///
    /// For a family whose batched numerics are sound at a narrow width and break
    /// at a wide one, `.always` throws away the width that works. `qwen3_moe` is
    /// the measured case: max logit error 0.94 at width 2 — inside the 1.25
    /// tolerance every other family is held to — and 2.69 at widths 4 and 8,
    /// which is not (`NativeModelLoader.batchDecodeRegressedModelTypes`). This
    /// case takes the width that is earned and refuses the width that is not,
    /// instead of refusing both.
    case maxWidth(Int)

    func requiresSolitude(_ request: Request) -> Bool {
        switch self {
        case .never: return false
        case .always: return true
        case .multimodalOnly: return request.isMultimodal
        // A capped-width row needs no protection of its own; the cap is a
        // property of the batch it would join, and `refusesToJoin` applies it.
        case .maxWidth: return false
        }
    }

    /// True when a waiting row may not be admitted alongside what is already
    /// running — either because it must have the batch to itself, or because the
    /// batch is already at its permitted width.
    func refusesToJoin(running requests: some Collection<Request>) -> Bool {
        switch self {
        case .never: return false
        case .always: return true
        case .multimodalOnly: return requests.contains { $0.isMultimodal }
        case .maxWidth(let width): return requests.count >= width
        }
    }
}

final class RunningRequest {
    let request: Request
    let promptTokens: [Int]
    let prefixHit: PrefixKVStoreHit?
    var generatedTokens: [Int]
    let generatedTokensIncludedInPrompt: Int
    var cachedTokenCount: Int

    init(
        request: Request,
        promptTokens: [Int],
        prefixHit: PrefixKVStoreHit?,
        generatedTokens: [Int],
        generatedTokensIncludedInPrompt: Int,
        cachedTokenCount: Int
    ) {
        self.request = request
        self.promptTokens = promptTokens
        self.prefixHit = prefixHit
        self.generatedTokens = generatedTokens
        self.generatedTokensIncludedInPrompt = generatedTokensIncludedInPrompt
        self.cachedTokenCount = cachedTokenCount
    }

    var generatedTokenCount: Int {
        generatedTokens.count
    }
}

public enum SchedulerError: Error, Equatable {
    case queueFull(retryAfterSteps: Int)
    case duplicateRequest(String)
    case invalidPrefixHit
}
