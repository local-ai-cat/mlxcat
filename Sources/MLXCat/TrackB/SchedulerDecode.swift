import MLX

/// Decode response processing is synchronous. The Scheduler actor lends its
/// running rows only for this call, retaining queue and cancellation ownership.
struct SchedulerDecode {
    private let generator: ContinuousBatchGenerator
    private let collector: OutputCollector
    private let prefixStore: (any PrefixKVStore)?
    private let prefixCacheEnabled: Bool
    private let cacheReleasePolicy: Scheduler.CacheReleasePolicy
    private(set) var droppedStaleResponseCount = 0
    private var decodeStepsSinceCacheClear = 0
    /// Drain-to-idle clears only once for each idle stretch.
    private var hasReleasedCacheForThisIdleStretch = true

    init(generator: ContinuousBatchGenerator, collector: OutputCollector, prefixStore: (any PrefixKVStore)?,
         prefixCacheEnabled: Bool, cacheReleasePolicy: Scheduler.CacheReleasePolicy) {
        self.generator = generator
        self.collector = collector
        self.prefixStore = prefixStore
        self.prefixCacheEnabled = prefixCacheEnabled
        self.cacheReleasePolicy = cacheReleasePolicy
    }

    mutating func processGeneratedResponses(
        _ rawResponses: [Response], running: inout [String: RunningRequest], resumedTokens: inout [String: [Int]]
    ) -> [Response] {
        var processed: [Response] = []
        decodeStepsSinceCacheClear += 1
        if cacheReleasePolicy.decodeStepInterval > 0,
           decodeStepsSinceCacheClear >= cacheReleasePolicy.decodeStepInterval {
            Memory.clearCache()
            decodeStepsSinceCacheClear = 0
        }
        var finishedUIDs: [String] = []
        var touchedUIDs: Set<String> = []

        for response in rawResponses {
            guard let runningRequest = running[response.uid] else {
                droppedStaleResponseCount += 1
                continue
            }
            runningRequest.generatedTokens.append(response.token)
            touchedUIDs.insert(response.uid)

            let finishReason: FinishReason?
            if runningRequest.request.eosTokenIds.contains(response.token) {
                finishReason = .stop
            } else if runningRequest.generatedTokenCount >= runningRequest.request.maxTokens {
                finishReason = .length
            } else {
                finishReason = nil
            }

            let processedResponse = Response(
                uid: response.uid,
                token: response.token,
                finishReason: finishReason,
                logprobs: response.logprobs
            )
            processed.append(processedResponse)
            collector.record(processedResponse)

            if finishReason != nil {
                finishedUIDs.append(response.uid)
            }
        }

        for uid in touchedUIDs where !finishedUIDs.contains(uid) {
            if let runningRequest = running[uid] {
                publishAvailablePrefixBlocks(
                    uid: uid,
                    runningRequest: runningRequest,
                    minimumNewTokens: Scheduler.midGenerationPublishInterval
                )
            }
        }

        finishGeneratedRequests(finishedUIDs, running: &running, resumedTokens: &resumedTokens)
        return processed
    }

    private func finishGeneratedRequests(
        _ finishedUIDs: [String], running: inout [String: RunningRequest], resumedTokens: inout [String: [Int]]
    ) {
        guard !finishedUIDs.isEmpty else { return }
        Stream.gpu.synchronize()
        for uid in finishedUIDs {
            if let finishedRequest = running[uid] {
                publishAvailablePrefixBlocks(uid: uid, runningRequest: finishedRequest)
            }
            let finishedRequest = running[uid]
            generator.remove(uid: uid)
            if let hit = finishedRequest?.prefixHit {
                prefixStore?.release(hit)
            }
            running.removeValue(forKey: uid)
            resumedTokens.removeValue(forKey: uid)
        }
    }

    /// Hand the buffer cache back once per idle stretch, not once per poll.
    mutating func releaseCacheIfDrained(isIdle: Bool) {
        guard isIdle else {
            hasReleasedCacheForThisIdleStretch = false
            return
        }
        guard cacheReleasePolicy.releasesWhenIdle, !hasReleasedCacheForThisIdleStretch else {
            return
        }
        hasReleasedCacheForThisIdleStretch = true
        decodeStepsSinceCacheClear = 0
        Memory.clearCache()
    }

    private func publishAvailablePrefixBlocks(
        uid: String, runningRequest: RunningRequest, minimumNewTokens: Int = 1
    ) {
        guard prefixCacheEnabled else { return }
        SchedulerPrefixCache.publishAvailablePrefixBlocks(
            uid: uid, runningRequest: runningRequest, generator: generator,
            store: prefixStore, minimumNewTokens: minimumNewTokens
        )
    }

}
