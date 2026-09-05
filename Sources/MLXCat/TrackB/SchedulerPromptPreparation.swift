import MLX
import MLXLMCommon

public final class LanguageModelBox: @unchecked Sendable {
    let model: any LanguageModel

    public init(_ model: any LanguageModel) {
        self.model = model
    }
}

/// Model preparation and prefix reconstruction execute synchronously on the
/// owning Scheduler actor. Mutable admission state is returned to that owner.
struct SchedulerPromptPreparation {
    private let model: any LanguageModel
    private let parameters: GenerateParameters
    private let prefixStore: (any PrefixKVStore)?
    private let prefixCacheEnabled: Bool
    private let schedulerManagedTextPrefill: Bool

    init(model: any LanguageModel, parameters: GenerateParameters,
         prefixStore: (any PrefixKVStore)?, prefixCacheEnabled: Bool, schedulerManagedTextPrefill: Bool) {
        self.model = model
        self.parameters = parameters
        self.prefixStore = prefixStore
        self.prefixCacheEnabled = prefixCacheEnabled
        self.schedulerManagedTextPrefill = schedulerManagedTextPrefill
    }

    func prepare(_ request: Request, sampling: SamplingParameters) throws
    -> SchedulerPrefill.Preparation {
        let rowCache = try model.newCache(parameters: parameters)
        let prefixCacheEligible = Self.isPrefixCacheEligible(request.input)
        let promptText: LMInput.Text
        switch try preparePromptInput(request, sampling: sampling, cache: rowCache) {
        case .tokens(let tokens): promptText = tokens
        case .row(let row): return .ready(row)
        }

        let promptTokensArray = promptText.tokens
        let promptTokenCount = promptTokensArray.dim(0)
        guard promptTokenCount > 0 else {
            throw BatchGeneratorError.promptTooShortForExternalPrefill
        }
        let prompt = PrefillPrompt(tokens: promptTokensArray.asArray(Int.self), array: promptTokensArray)

        if promptTokenCount == 1 {
            let output = model(promptText[text: .newAxis], cache: rowCache, state: nil)
            eval(rowCache)
            let firstToken = SchedulerPrefill.sampledToken(from: output.logits, sampling: sampling)
            return .ready(SchedulerPrefill.Row(
                cache: rowCache,
                lastToken: firstToken.token,
                promptTokens: [],
                prefixHit: nil,
                initialGeneratedToken: firstToken,
                modelState: output.state
            ))
        }

        if prefixCacheEligible, let hit = prepareCachedRow(request, sampling: sampling, prompt: prompt) {
            return hit
        }

        return prefillMissRow(
            request: request,
            sampling: sampling,
            prompt: prompt,
            storedPromptTokens: prefixCacheEligible ? prompt.tokens : [],
            rowCache: rowCache
        )
    }

    private func prepareHitRow(
        request: Request,
        sampling: SamplingParameters,
        prompt: PrefillPrompt,
        hit: PrefixKVStoreHit,
        reconstructedCache: [any KVCache]
    ) throws -> SchedulerPrefill.Preparation {
        let matched = hit.matchedTokenCount
        guard matched <= prompt.tokens.count else {
            throw SchedulerError.invalidPrefixHit
        }

        return .pending(SchedulerPrefill.Admission(
            request: request,
            sampling: sampling,
            cache: reconstructedCache,
            promptTokens: prompt.tokens,
            promptTokensArray: prompt.array,
            storedPromptTokens: prompt.tokens,
            prefixHit: hit,
            prefillRange: matched ..< prompt.tokens.count,
            nextPrefillIndex: matched,
            // A hit resumes on a WARM cache, which is the shape that trapped
            // batched qwen3_5 inside the model. Seeded rather than left nil —
            // see ``BatchPositionalState/textOnlyResumeState()``.
            state: BatchPositionalState.textOnlyResumeState(),
            initialGeneratedToken: nil
        ))
    }

    private func prefillMissRow(
        request: Request,
        sampling: SamplingParameters,
        prompt: PrefillPrompt,
        storedPromptTokens: [Int],
        rowCache: [any KVCache]
    ) -> SchedulerPrefill.Preparation {
        .pending(SchedulerPrefill.Admission(
            request: request,
            sampling: sampling,
            cache: rowCache,
            promptTokens: prompt.tokens,
            promptTokensArray: prompt.array,
            storedPromptTokens: storedPromptTokens,
            prefixHit: nil,
            prefillRange: 0 ..< prompt.tokens.count,
            nextPrefillIndex: 0,
            state: nil,
            initialGeneratedToken: nil
        ))
    }

    private struct PrefillPrompt {
        let tokens: [Int]
        let array: MLXArray
    }

    private enum PreparedPromptInput {
        case tokens(LMInput.Text)
        case row(SchedulerPrefill.Row)
    }

    private func preparePromptInput(
        _ request: Request, sampling: SamplingParameters, cache: [any KVCache]
    ) throws -> PreparedPromptInput {
        let canUseRawTextTokens = schedulerManagedTextPrefill
            && Self.isPrefixCacheEligible(request.input)
            && request.input.text.tokens.ndim == 1
            && request.input.text.mask == nil
        if canUseRawTextTokens {
            return .tokens(request.input.text)
        } else {
            // state: nil — `cache` was just created above, so there is no
            // prior model state to carry into the prefill.
            switch try model.prepare(
                request.input, cache: cache, state: nil,
                prefill: parameters.prefill) {
            case .tokens(let tokens):
                return .tokens(tokens)
            case .logits(let output):
                let firstToken = SchedulerPrefill.sampledToken(from: output.logits, sampling: sampling)
                return .row(SchedulerPrefill.Row(
                    cache: cache,
                    lastToken: firstToken.token,
                    promptTokens: [],
                    prefixHit: nil,
                    initialGeneratedToken: firstToken,
                    modelState: output.state
                ))
            }
        }

    }

    private func prepareCachedRow(
        _ request: Request, sampling: SamplingParameters, prompt: PrefillPrompt
    ) -> SchedulerPrefill.Preparation? {
        guard prefixCacheEnabled, let prefixStore,
              let hit = prefixStore.fetch(tokens: prompt.tokens, sessionKey: request.cacheSession)
        else { return nil }
        if hit.matchedTokenCount == prompt.tokens.count {
            prefixStore.release(hit)
            return nil
        }
        do {
            let serialized = try prefixStore.reconstructCache(from: hit)
            let reconstructedCache = try serialized.map {
                try BlockAwarePrefixKVStore.cache(from: $0)
            }
            return try prepareHitRow(
                request: request,
                sampling: sampling,
                prompt: prompt,
                hit: hit,
                reconstructedCache: reconstructedCache
            )
        } catch {
            prefixStore.release(hit)
            SchedulerPrefixCache.logFailure("prefix hit reconstruction failed; falling back to cache miss", error)
        }
        return nil
    }
    private static func isPrefixCacheEligible(_ input: LMInput) -> Bool {
        input.image == nil && input.video == nil && input.audio == nil
    }
    static func resumedRequest(for runningRequest: RunningRequest) -> Request {
        let resumedTokens = currentContextTokens(for: runningRequest)
        let resumedInput = LMInput(tokens: MLXArray(resumedTokens.map(Int32.init)))
        return Request(
            uid: runningRequest.request.uid,
            input: resumedInput,
            maxTokens: runningRequest.request.maxTokens,
            sampling: runningRequest.request.sampling,
            eosTokenIds: runningRequest.request.eosTokenIds,
            cacheSession: runningRequest.request.cacheSession,
            retainedAllocation: runningRequest.request.retainedAllocation
        )

    }

    static func canResumeFromTokenPrompt(_ runningRequest: RunningRequest) -> Bool {
        !runningRequest.promptTokens.isEmpty
            && runningRequest.request.input.image == nil
            && runningRequest.request.input.video == nil
            && runningRequest.request.input.audio == nil
            && runningRequest.request.input.text.tokens.ndim == 1
            && runningRequest.request.input.text.mask == nil
    }

    private static func currentContextTokens(for runningRequest: RunningRequest) -> [Int] {
        runningRequest.promptTokens
            + runningRequest.generatedTokens.dropFirst(runningRequest.generatedTokensIncludedInPrompt)
    }

}
