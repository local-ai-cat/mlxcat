import Foundation
import MLXLMCommon

/// Whether, and how, the KV cache is quantized.
///
/// The KV cache — not the weights — is what decides whether long context fits.
/// At 128k tokens Qwen3-Coder-30B holds **12.0 GiB** of fp16 KV next to ~17 GiB
/// of weights; at 4 bits it holds 3.4 GiB, which is the difference between
/// "does not fit on a 48 GB Mac" and "fits with room for a second stream". On
/// iOS the same arithmetic is a jetsam kill rather than a swap.
///
/// The real reduction is **1.88x at 8 bits and 3.56x at 4 bits**, not 2x and 4x:
/// affine quantization stores an fp16 scale and bias per group of 64, so a group
/// costs `64*bits/8 + 4` bytes. Quoting 4x here would overstate every number
/// downstream of it.
///
/// ## What this does NOT cover yet
///
/// Stage 1 is **single-stream only**, and that is enforced rather than assumed —
/// see ``Scheduler``. Upstream's converter rewrites only `.simple` leaves
/// (`KVCache.swift:2512`), so a `BatchKVCache` is invisible to it and a batch
/// that quantizes AFTER merging is a clean no-op. The dangerous order is the
/// other one: quantizing a row and THEN merging it. `BatchLayerCache`
/// classifies a layer by `state.count == 2`, and a quantized layer's state is 4
/// or 6 arrays, so it is misrouted — which throws on rows of different lengths
/// and, for two rows of the SAME length, passes shape validation and then reads
/// `[step, offset, groupSize, bits]` as slot metadata. That second case is
/// silent wrong output, which is why this is gated and not merely discouraged.
///
/// Rotating (sliding-window) and recurrent (Mamba/GDN) layers are skipped by
/// upstream too — `RotatingKVCache.toQuantized` throws by design
/// (`KVCache.swift:894`), matching mlx-lm (`guest/mlx-lm/mlx_lm/models/cache.py:551`).
/// Hybrid models therefore degrade gracefully: their full-attention layers
/// convert and the rest stay fp16 at a bounded size.
///
/// ## Defaults
///
/// Off. mlx-lm ships `--kv-bits` defaulting to None with group size 64 and
/// `--quantized-kv-start 5000` (`guest/mlx-lm/mlx_lm/generate.py:56,192-209`),
/// and does not expose it from its server at all. omlx defaults its own scheme
/// off; mlx-serve has no KV quantization.
///
/// The price is a decode slowdown — quantized attention is an unfused
/// `quantizedMatmul` + softmax rather than a fused SDPA kernel, and `qwen3_5`
/// additionally loses its compiled decode route (`Qwen35.swift:707-712` gates
/// that on a plain attention path). Measured rather than feared, Qwen3.5-4B at a
/// 16,384-token prompt on an M5 Max:
///
///     fp16   31.5 tok/s     561 MiB resident
///     kv4    27.6 tok/s     193 MiB resident
///
/// **12% of decode for 65% of the memory** — and on this model only 8 of 32
/// layers are even quantizable. That is a good trade for anyone near a memory
/// ceiling and a bad one for anyone not, which is exactly why it is a switch
/// rather than a default.
public struct KVQuantizationPolicy: Sendable, Equatable {
    /// Bits per KV element. `nil` — the default — leaves the cache in fp16.
    public var bits: Int?
    /// Elements per affine group. 64 is mlx-lm's default and divides every head
    /// dimension our models use (64, 128, 256).
    public var groupSize: Int
    /// Convert only once the cache is longer than this. mlx-lm uses 5000: below
    /// it the KV is small enough that quantization buys nothing and costs
    /// dequantization on every attention.
    ///
    /// Conversion itself allocates the quantized copy while the fp16 original is
    /// still live, so the moment of crossing is the peak. On a device where the
    /// peak is the constraint, a threshold of 0 is safer than a large one.
    public var startTokens: Int

    public init(bits: Int? = nil, groupSize: Int = 64, startTokens: Int = 5000) {
        self.bits = bits
        self.groupSize = groupSize
        self.startTokens = startTokens
    }

    public static let off = KVQuantizationPolicy()

    public var isEnabled: Bool { bits != nil }

    /// Families whose attention does something the quantized route drops.
    ///
    /// `gpt_oss` uses attention sinks. `GPTOSS.swift:255-271` takes a quantized
    /// path that does not pass `sinks:` — and `quantizedScaledDotProductAttention`
    /// has no such parameter to pass them to — while the fp16 path at :283-287
    /// does. Quantizing it would silently drop the sinks, which is a correctness
    /// change wearing quantization's clothes. omlx excludes attention-sink models
    /// for the same reason (`guest/omlx/omlx/scheduler.py:2725-2741`).
    public static let unsupportedModelTypes: Set<String> = ["gpt_oss"]

    /// The policy actually applied to a model — `.off` for a family we exclude,
    /// whatever was asked for otherwise.
    public static func resolved(_ requested: KVQuantizationPolicy, modelType: String?)
        -> KVQuantizationPolicy
    {
        guard requested.isEnabled, let modelType else { return requested }
        return unsupportedModelTypes.contains(modelType.lowercased()) ? .off : requested
    }

    /// `MLXCAT_KV_BITS` (4 or 8; anything else leaves it off), `MLXCAT_KV_GROUP_SIZE`,
    /// `MLXCAT_QUANTIZED_KV_START`.
    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> KVQuantizationPolicy {
        var policy = KVQuantizationPolicy.off
        if let raw = environment["MLXCAT_KV_BITS"]?.trimmingCharacters(in: .whitespaces),
            let bits = Int(raw), bits == 4 || bits == 8
        {
            policy.bits = bits
        }
        if let raw = environment["MLXCAT_KV_GROUP_SIZE"]?.trimmingCharacters(in: .whitespaces),
            let group = Int(raw), group > 0
        {
            policy.groupSize = group
        }
        if let raw = environment["MLXCAT_QUANTIZED_KV_START"]?.trimmingCharacters(in: .whitespaces),
            let start = Int(raw), start >= 0
        {
            policy.startTokens = start
        }
        return policy
    }

    /// Convert in place, if this policy is on. A no-op for every cache type
    /// upstream does not recognize, which is what makes it safe to call on a
    /// hybrid model's mixed stack.
    func apply(to cache: inout [any KVCache]) {
        guard let bits else { return }
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: bits,
            kvGroupSize: groupSize,
            quantizedKVStart: startTokens
        )
    }
}
