import MLX
import MLXLMCommon

/// Per-row RoPE anchors for model families that carry their positional anchor in
/// `LMOutput.State` instead of reading it back off the KV cache.
///
/// Most families place a decode token at `cache.ropeOffset`, which
/// `BatchKVCache` already reports per row (`idx - leftPadding`, pinned by
/// `BatchRoPEOffsetTests`). Qwen3.5 does not. It computes
///
///     position(row, j) = cache[faIdx].offset + ropeDeltas[row] + j
///
/// from a SCALAR cache offset plus a delta it hands back in `LMOutput.State`
/// under `"qwen35.ropeDeltas"` — the M-RoPE anchor the prompt's images pushed
/// the text past. Upstream reference, `mlx-swift-lm`
/// `Libraries/MLXVLM/Models/Qwen35.swift`: the prefill branch stores the delta
/// at :894, the decode branch consumes it at :897-918, and the entry point
/// refuses a warm cache without it at :1298
///
///     precondition(faCacheOffset(cache ?? []) == 0 || state?[ropeDeltasKey] != nil,
///                  "Qwen35 cannot continue a warm prompt cache without qwen35.ropeDeltas")
///
/// That precondition is what killed the benchmark's batched `qwen3_5` server —
/// `ContinuousBatchGenerator` dropped the state the moment a second row joined,
/// so the next decode step trapped (`docs/KNOWN-FAILURES.md` §1d). The fix is
/// not to restore the scalar: a scalar cannot describe two rows. It is to feed
/// the vector the decode branch already accepts — :908-914 broadcasts a
/// 0-d delta, truncates a long one, and indexes `delta[row]` otherwise.
///
/// Two things fold into the one vector:
///
///   * the row's own prefill delta, captured when it was inserted, and
///   * `-leftPadding[row]`, because the scalar `cache.offset` the model adds is
///     the PADDED batch length, while the row's true position is
///     `offset - leftPadding[row]`. This is the same correction
///     `BatchKVCache.ropeOffset` applies for every other family.
///
/// so `delta[row] = prefillDelta[row] - leftPadding[row]`, and at width 1 with
/// no padding it degenerates to exactly the scalar that shipped before.
///
/// Only `"qwen35.ropeDeltas"` is recognized. Sibling families
/// (`qwen25vl.ropeDeltas`, `qwen2vl.ropeDeltas`, `glmocr.ropeDeltas`) carry a
/// key of the same shape but consume it differently, and none has been read
/// line by line the way this one has — they stay serialized rather than get a
/// guess.
struct BatchPositionalState {
    /// Verified against the decode branch cited above. Adding a key here is a
    /// claim that its model indexes the delta per row; verify before extending.
    static let recognizedKeys: [LMOutput.Key<MLXArray>] = [
        LMOutput.Key<MLXArray>("qwen35.ropeDeltas")
    ]

    /// The anchor a freshly prefilled row handed back, or nil if this model
    /// carries no positional state at all (the common case).
    static func delta(in state: LMOutput.State?) -> (key: LMOutput.Key<MLXArray>, delta: Int32)? {
        guard let state else { return nil }
        for key in recognizedKeys {
            guard let value = state[key] else { continue }
            return (key, scalar(value))
        }
        return nil
    }

    /// The prefill delta is `[1]`-shaped for a single row and 0-d when a model
    /// collapses it; both mean one row, so read element zero either way.
    private static func scalar(_ value: MLXArray) -> Int32 {
        let flattened = value.asType(.int32).reshaped([-1])
        guard flattened.dim(0) > 0 else { return 0 }
        return flattened[0].item(Int32.self)
    }

    /// The anchor a prefix-cache resume needs.
    ///
    /// A prefix hit hands the scheduler a WARM cache and no state, which is the
    /// same shape that trapped the batch path — `Qwen35.callAsFunction` refuses
    /// it. Zero is not a placeholder: `isPrefixCacheEligible` admits text-only
    /// inputs only, and `getRopeIndex`'s text-only branch returns exactly the
    /// position offset as its delta (Qwen3VL.swift:1467-1470), which is zero for
    /// a prefix that starts at position zero.
    ///
    /// Unrecognized keys are inert — a model reads the ones it defined and
    /// ignores the rest — so this is seeded unconditionally rather than behind a
    /// model-type test the scheduler would have to learn.
    static func textOnlyResumeState() -> LMOutput.State {
        var state = LMOutput.State()
        for key in recognizedKeys {
            state[key] = MLXArray([Int32(0)])
        }
        return state
    }

    /// The state one decode step feeds the model: one delta per live row, in
    /// row order, already corrected for left padding.
    static func stepState(
        key: LMOutput.Key<MLXArray>,
        prefillDeltas: [Int32],
        leftPadding: [Int32]
    ) -> LMOutput.State {
        var state = LMOutput.State()
        let deltas = prefillDeltas.enumerated().map { row, delta in
            delta - (row < leftPadding.count ? leftPadding[row] : 0)
        }
        state[key] = MLXArray(deltas)
        return state
    }
}
