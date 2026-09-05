import Foundation
import MLX
import MLXLMCommon

/// Cache effects are synchronous and invoked only by the owning Scheduler.
enum SchedulerPrefixCache {
    private static func cacheSnapshot(uid: String, generator: ContinuousBatchGenerator) -> [SerializedKVLayer]? {
        guard let cache = generator.extractCache(uid: uid) else {
            return nil
        }
        return cache.map {
            SerializedKVLayer(
                state: $0.state,
                metaState: $0.metaState,
                className: String(describing: type(of: $0))
            )
        }
    }

    static func storeCompletedAdmissionPrefix(
        _ row: SchedulerPrefill.Row, request: Request, store: (any PrefixKVStore)?
    ) {
        guard let prefixStore = store,
              !row.promptTokens.isEmpty
        else {
            return
        }

        let snapshot = row.cache.map {
            SerializedKVLayer(
                state: $0.state,
                metaState: $0.metaState,
                className: String(describing: type(of: $0))
            )
        }
        do {
            try prefixStore.store(
                tokens: row.promptTokens,
                sessionKey: request.cacheSession,
                cache: snapshot,
                allocation: request.retainedAllocation
            )
        } catch {
            logFailure("completed admission prefix cache store failed", error)
        }
    }

    static func publishAvailablePrefixBlocks(
        uid: String,
        runningRequest: RunningRequest,
        generator: ContinuousBatchGenerator,
        store: (any PrefixKVStore)?,
        minimumNewTokens: Int
    ) {
        guard let prefixStore = store,
              !runningRequest.promptTokens.isEmpty
        else {
            return
        }

        // The live KV cache has consumed the token from the previous step. The
        // token sampled by the current step is now `currentTokens`, so it is not
        // part of the cache snapshot until the next decode call.
        let generatedTokensAfterPrompt = runningRequest.generatedTokens
            .dropFirst(runningRequest.generatedTokensIncludedInPrompt)
        let cachedGeneratedTokens = generatedTokensAfterPrompt.dropLast()
        let availableTokens = runningRequest.promptTokens + cachedGeneratedTokens
        let requiredNewTokens = runningRequest.cachedTokenCount == 0 ? 1 : minimumNewTokens
        guard availableTokens.count >= runningRequest.cachedTokenCount + requiredNewTokens,
              let snapshot = cacheSnapshot(uid: uid, generator: generator)
        else {
            return
        }

        do {
            try prefixStore.store(
                tokens: availableTokens,
                sessionKey: runningRequest.request.cacheSession,
                cache: snapshot,
                allocation: runningRequest.request.retainedAllocation
            )
            runningRequest.cachedTokenCount = availableTokens.count
        } catch {
            logFailure("prefix cache store failed", error)
        }
    }

    static func logFailure(_ message: String, _ error: Error) {
        let line = "MLXCat cache warning: \(message): \(error)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

}
