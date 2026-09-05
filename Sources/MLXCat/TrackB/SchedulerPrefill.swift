import Foundation
import MLX
import MLXLMCommon

/// Synchronous prefill mechanics. Scheduler owns the admission value and calls
/// these functions on its actor; no model, cache, or state crosses an await.
enum SchedulerPrefill {
    struct Configuration {
        let stepSize: Int?
        let chunkIdlePrefill: Bool
        let prefillsLastTokenAlone: Bool
        let kvQuantization: KVQuantizationPolicy
    }

    static func advance(
        _ admission: inout Admission,
        model: any LanguageModel,
        configuration: Configuration,
        allowPartialPrefill: Bool,
        priorGeneratedTokens: [Int]
    ) -> Row? {

        // Two different questions wear one number here, and they want opposite
        // answers, so they get two.
        //
        // IDLE prefill (nothing decoding) is a pure memory question: no stream
        // is waiting on a tick, so a narrow chunk costs only per-chunk overhead
        // and saves attention scratch. Measured at 16k on an M5 Max, widening it
        // from 512 to 2048 cost gemma-4-12B 13.42 -> 15.56 GiB and Qwen3.8-27B
        // 18.90 -> 22.91 GiB. So idle stays narrow.
        //
        // BUSY prefill (something is decoding) is a scheduling question: every
        // chunk boundary costs a decode tick and an actor round trip, so at 512
        // a 16k prompt pays that 32 times. mlx-lm and omlx use 2048
        // (`guest/mlx-lm/mlx_lm/generate.py:1509`,
        // `guest/omlx/omlx/scheduler.py:1304`); mlx-serve uses 8192 with
        // per-model caps (`guest/mlx-serve/src/generate.zig:34`, `:111`). 2048 is
        // the conservative end — what the two engines closest to our stack use.
        //
        // An explicit `prefill.stepSize` overrides both: a caller that names a
        // width means it.
        let idlePrefillStep = 512
        let busyPrefillStep = 2048
        let prefillStep: Int = max(
            1,
            configuration.stepSize ?? (allowPartialPrefill ? busyPrefillStep : idlePrefillStep)
        )
        let stepBudget: Int = allowPartialPrefill ? prefillStep : Int.max
        var remainingBudget = stepBudget
        // `allowPartialPrefill` decides whether admission YIELDS between chunks
        // so other rows can decode. Whether the forward pass itself spans the
        // whole prompt in one `model()` call is a separate question, and
        // `chunkIdlePrefill` answers it.
        //
        // They used to be the same flag, so an idle admission — every
        // single-stream request, so every c1 cell — ran one call over L
        // positions and materialized a [1, L, vocab] logits tensor plus
        // full-length attention scratch. Chunking that away is worth GiBs on
        // most models and costs GiBs on at least one, which is why the caller
        // decides. Measured at 16k on an M5 Max, after-load -> lifetime max,
        // both arms back to back:
        //
        //     model                  vocab    single-pass   chunked
        //     gemma-4-12B           262,144    34.51         20.74   -13.8
        //     Qwen3.8-27B           248,320    53.02         39.02   -14.0
        //     gpt-oss-20b           201,088    35.17         25.37    -9.8
        //     Qwen3-Coder-30B-A3B   151,936    24.75         40.34   +15.6
        //
        // See `NativeModelLoader.singlePassPrefillModelTypes` for why the last
        // row is an exemption rather than a reason to abandon the change.
        let chunked = allowPartialPrefill || configuration.chunkIdlePrefill
        // The LAST prompt token is prefilled alone, so the one forward whose
        // logits we actually read produces `[1, 1, vocab]` instead of
        // `[1, chunk, vocab]`.
        //
        // mlx-lm structures prefill exactly this way — it loops `while y.size > 1`
        // and hands the single remaining token to the step that samples
        // (`guest/mlx-lm/mlx_lm/generate.py:580-587`, and the same shape at
        // `:430-452`). Its prefill chunks evaluate only the cache
        // (`mx.eval([c.state for c in cache])`), never the logits, so MLX's
        // laziness means a prefill chunk's logits are never computed at all.
        //
        // We read `output.logits` on whichever chunk ends the range, which at a
        // 262k vocab is ~268 MB of fp16 for a 512-token chunk and gigabytes on
        // the single-pass path — all of it discarded except one row.
        let logitsBoundary =
            configuration.prefillsLastTokenAlone
            ? admission.prefillRange.upperBound - 1
            : admission.prefillRange.upperBound
        while admission.nextPrefillIndex < admission.prefillRange.upperBound, remainingBudget > 0 {
            let end: Int
            if admission.nextPrefillIndex == logitsBoundary {
                // The final token, alone. This is the forward we sample from.
                end = admission.prefillRange.upperBound
            } else if chunked {
                end = min(
                    admission.nextPrefillIndex + min(prefillStep, remainingBudget),
                    logitsBoundary
                )
            } else {
                end = logitsBoundary
            }
            prefillChunk(&admission, endingAt: end, model: model,
                         priorGeneratedTokens: priorGeneratedTokens, kvQuantization: configuration.kvQuantization)
            remainingBudget -= end - admission.nextPrefillIndex
            admission.nextPrefillIndex = end
            // Copied from mlx-lm's `_prefill` (guest/mlx-lm/mlx_lm/generate.py:579),
            // which calls `mx.clear_cache()` at the end of every chunk. Each
            // chunk attends over a longer key range than the last, so no two
            // chunks allocate the same attention-scratch shape and MLX's buffer
            // cache can never reuse one — it just accumulates every shape for
            // the whole prefill. Freeing between chunks is what makes chunked
            // prefill cheap for them and was the difference from us.
            //
            // Only between chunks, and only when chunking: the final chunk's
            // buffers are about to be reused by decode, and the single-pass path
            // allocates nothing worth freeing mid-loop.
            if chunked, admission.nextPrefillIndex < admission.prefillRange.upperBound {
                Memory.clearCache()
            }
        }

        if admission.nextPrefillIndex < admission.prefillRange.upperBound {
            return nil
        }

        eval(admission.cache)
        guard let initialGeneratedToken = admission.initialGeneratedToken else {
            preconditionFailure("completed prompt prefill without an initial generated token")
        }
        return Row(
            cache: admission.cache,
            lastToken: initialGeneratedToken.token,
            promptTokens: admission.storedPromptTokens,
            prefixHit: admission.prefixHit,
            initialGeneratedToken: initialGeneratedToken,
            modelState: admission.state
        )
    }

    private static func prefillChunk(
        _ admission: inout Admission, endingAt end: Int, model: any LanguageModel,
        priorGeneratedTokens: [Int], kvQuantization: KVQuantizationPolicy
    ) {
        let input = LMInput.Text(
            tokens: admission.promptTokensArray[admission.nextPrefillIndex ..< end]
        )
        let output = model(input[text: .newAxis], cache: admission.cache, state: admission.state)
        admission.state = output.state
        if end == admission.prefillRange.upperBound {
            admission.initialGeneratedToken = sampledToken(
                from: output.logits,
                sampling: admission.sampling,
                priorGeneratedTokens: priorGeneratedTokens
            )
        }
        asyncEval(admission.cache)
        // Convert inside the chunk loop, as mlx-lm does
        // (`guest/mlx-lm/mlx_lm/generate.py:418,441`), not once at the end:
        // conversion allocates the quantized copy while the fp16 original is
        // still live, so converting per chunk caps that doubling at one
        // chunk's worth of KV instead of the whole prompt's. A no-op until
        // the cache passes `startTokens`, and a no-op forever for rotating
        // and recurrent layers, which upstream declines to quantize.
        kvQuantization.apply(to: &admission.cache)
    }

    static func sampledToken(
        from logits: MLXArray,
        sampling: SamplingParameters,
        priorGeneratedTokens: [Int] = []
    ) -> GeneratedToken {
        let nextTokenLogits = logits[0..., -1, 0...]
        let matcher = sampling.jsonGrammar?.makeMatcher()
        let regexMatcher = sampling.regexGrammar?.makeMatcher()
        let gbnfMatcher = sampling.gbnfGrammar?.makeMatcher()
        var toolGrammarState = sampling.toolGrammar.map { QwenXMLToolGrammarState(configuration: $0) }
        for tokenID in priorGeneratedTokens {
            toolGrammarState?.observeGeneratedToken(tokenID)
        }
        var thinkingBudgetState = sampling.thinkingBudget.map(ThinkingBudgetState.init(configuration:))
        let token = TokenSampler.sample(
            logits: nextTokenLogits[0, 0...],
            parameters: sampling,
            generatedTokens: priorGeneratedTokens,
            jsonGrammarMatcher: matcher,
            regexGrammarMatcher: regexMatcher,
            gbnfGrammarMatcher: gbnfMatcher,
            toolGrammarMatcher: toolGrammarState?.activeMatcher,
            postToolAllowedTokenIDs: toolGrammarState?.activeMatcher == nil
                ? toolGrammarState?.activeAllowedTokenIDs
                : nil,
            thinkingBudgetState: &thinkingBudgetState
        )
        let tokenID = token.item(Int.self)
        if matcher?.accepts(tokenID: tokenID) == true {
            matcher?.advance(tokenID: tokenID)
        }
        if regexMatcher?.accepts(tokenID: tokenID) == true {
            regexMatcher?.advance(tokenID: tokenID)
        }
        if gbnfMatcher?.accepts(tokenID: tokenID) == true {
            gbnfMatcher?.advance(tokenID: tokenID)
        }
        toolGrammarState?.observeGeneratedToken(tokenID)
        thinkingBudgetState?.advance(tokenID: tokenID)
        return GeneratedToken(
            token: token,
            tokenID: tokenID,
            toolGrammarState: toolGrammarState,
            thinkingBudgetState: thinkingBudgetState
        )
    }

    static func finishReason(
        token: Int,
        generatedTokenCount: Int,
        request: Request
    ) -> FinishReason? {
        if request.eosTokenIds.contains(token) {
            return .stop
        }
        if generatedTokenCount >= request.maxTokens {
            return .length
        }
        return nil
    }
    enum Preparation {
        case ready(Row)
        case pending(Admission)
    }

    struct Admission {
        let request: Request
        let sampling: SamplingParameters
        /// `var` because KV quantization REPLACES layers in place partway through
        /// prefill: `KVCacheSimple` becomes `QuantizedKVCache` once the cache passes
        /// the threshold, so the array cannot be a constant.
        var cache: [any KVCache]
        let promptTokens: [Int]
        let promptTokensArray: MLXArray
        let storedPromptTokens: [Int]
        let prefixHit: PrefixKVStoreHit?
        let prefillRange: Range<Int>
        var nextPrefillIndex: Int
        var state: LMOutput.State?
        var initialGeneratedToken: GeneratedToken?
    }

    struct Row {
        let cache: [any KVCache]
        let lastToken: MLXArray
        let promptTokens: [Int]
        let prefixHit: PrefixKVStoreHit?
        let initialGeneratedToken: GeneratedToken?
        /// Model state produced by the prefill (e.g. Qwen3.5/Qwen-VL M-RoPE
        /// ropeDeltas). Stateful models refuse to decode a warm cache without it.
        let modelState: LMOutput.State?
    }

    struct GeneratedToken {
        let token: MLXArray
        let tokenID: Int
        let toolGrammarState: QwenXMLToolGrammarState?
        let thinkingBudgetState: ThinkingBudgetState?
    }
}
