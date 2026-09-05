import Foundation
import MLXCat
import MLXCatHTTP

/// Preparation tokenizes only. The factory is invoked once, after admission.
struct PreparedNativeChat: Sendable {
    let start: @Sendable (RetainedAllocation?) async throws -> OpenAIChatStream
    let promptTokens: Int

    func admitAndStart<Loader: EnginePoolModelLoader>(pool: EnginePool<Loader>, additionalBytes: Int64?,
                                                      checkPressure: @Sendable (Int64) async throws
                                                      -> Void) async throws
                                                      -> (stream: OpenAIChatStream, reservation: UUID) {
        let reservation = try await pool.reserveRequest(additionalBytes: additionalBytes)
        do {
            try await checkPressure(await pool.reservedRequestBytes)
            try Task.checkCancellation()
            let allocation = await pool.retainedAllocation(for: reservation)
            return (try await start(allocation), reservation)
        } catch {
            await pool.releaseRequest(reservation)
            throw error
        }
    }
}

enum PoolLeasedChatStream {
    static func make<Loader: EnginePoolModelLoader>(_ stream: OpenAIChatStream, pool: EnginePool<Loader>,
                                                    lease: EnginePoolLease<Loader.Engine>,
                                                    ticket: EnginePoolQueueTicket,
                                                    reservation: UUID?) -> OpenAIChatStream {
        let release = PoolChatReleaseState(pool: pool, lease: lease, ticket: ticket,
                                           reservation: reservation, teardown: stream.cancelAndWait)
        let chunks = AsyncThrowingStream<OpenAIChatChunk, Error> { continuation in
            let task = Task {
                do {
                    for try await chunk in stream.chunks { continuation.yield(chunk) }
                    await release.finish()
                    continuation.finish()
                } catch {
                    await release.finish()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
                Task { await release.finish() }
            }
        }
        return OpenAIChatStream(promptTokens: stream.promptTokens, chunks: chunks,
                                cancelAndWait: { await release.finish() })
    }
}

private actor PoolChatReleaseState<Loader: EnginePoolModelLoader> {
    let pool: EnginePool<Loader>
    private var lease: EnginePoolLease<Loader.Engine>?
    let ticket: EnginePoolQueueTicket
    let reservation: UUID?
    private var teardown: (@Sendable () async -> Void)?
    private var completion: Task<Void, Never>?

    init(pool: EnginePool<Loader>, lease: EnginePoolLease<Loader.Engine>, ticket: EnginePoolQueueTicket,
         reservation: UUID?, teardown: @escaping @Sendable () async -> Void) {
        self.pool = pool
        self.lease = lease
        self.ticket = ticket
        self.reservation = reservation
        self.teardown = teardown
    }

    func finish() async {
        if let completion { await completion.value; return }
        let task = Task {
            await teardown?()
            teardown = nil
            if let reservation { await pool.releaseRequest(reservation) }
            if let lease { await pool.release(lease) }
            lease = nil
            await pool.finishWaitingRequest(ticket)
        }
        completion = task
        await task.value
    }
}
