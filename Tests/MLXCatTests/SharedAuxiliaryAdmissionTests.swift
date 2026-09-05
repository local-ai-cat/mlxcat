import Foundation
import MLXCat
import MLXCatHTTP
import MLXCatSpeech
@testable import MLXCatNative
@testable import MLXCatHTTPServer
import Testing

extension SharedModelAccountingTests {
    @Test func residentAuxiliaryWorkChecksPressureAndRecoversWithoutReloading() async throws {
        let pool = emptyPool(100)
        let pressure = AuxiliaryPressureControl()
        await pool.setPressureAdmission { try await pressure.check($0) }
        let rankCache = AccountingRerankCache()
        let rank = await NativeRerankBackend(models: ["rank": model("rank", 20)], pool: pool, cache: rankCache)
        let speechGate = AccountingBarrier()
        await speechGate.open()
        let adapter = AccountingSpeechAdapter(bytes: 20, work: speechGate)
        let speech = await PoolBackedSpeechAdapter(adapter: adapter, pool: pool)
        let embedCache = AccountingEmbeddingCache()
        let embed = NativeEmbeddingsBackend(modelURL: URL(fileURLWithPath: "/fixture/embed"),
                                            footprint: EnginePoolFootprint(bytes: 20, source: .configuredEstimate),
                                            pool: pool, cache: embedCache)
        let operations: [@Sendable () async throws -> Void] = [
            { _ = try await rank.rerank(rerankRequest()) },
            { _ = try await speech.transcribeFile(.init(model: "speech", fileName: "fixture.wav", fileData: Data())) },
            { _ = try await embed.embed(.init(input: .string("hello"), model: "embed")) }
        ]
        for operation in operations { try await operation() }
        #expect(await pool.status().currentModelMemory == 60)
        let growth = try await pool.reserveRequest(additionalBytes: 10)
        await pressure.setDenied(true)
        for operation in operations {
            do {
                try await operation()
                Issue.record("Resident work bypassed pressure admission")
            } catch let error as OpenAIHTTPError { #expect(error.status == 507)
            } catch let error as EnginePoolError { #expect(error.httpStatus == 507) }
        }
        #expect(await rankCache.scores == 1)
        #expect(await adapter.transcriptions == 1)
        #expect(await embedCache.requests.count == 1)
        #expect(await pool.resourceStatuses().allSatisfy { $0.inUse == 0 && $0.loaded })
        #expect(await pool.status().currentModelMemory == 60)
        #expect(await pressure.arguments.suffix(3) == [10, 10, 10])
        await pressure.setDenied(false)
        for operation in operations { try await operation() }
        #expect(await rankCache.scores == 2)
        #expect(await adapter.transcriptions == 2)
        #expect(await embedCache.requests.count == 2)
        #expect(await rankCache.loads == 1)
        #expect(await adapter.loads == 1)
        #expect(await embedCache.loads == 1)
        await pool.releaseRequest(growth)
    }

    @Test func residentPressureWaitProtectsResourceAndCancellationUnwindsWaiter() async throws {
        let pool = emptyPool(100)
        let state = AccountingResourceState()
        await pool.registerResource(EnginePoolResource(id: "rerank://rank", modelID: "rank", modelType: "rerank",
                                                       footprint: EnginePoolFootprint(bytes: 40,
                                                                                      source: .configuredEstimate),
                                                       usesMLXAllocator: true,
                                                       load: { try await state.load() },
                                                       unload: { await state.unload() }))
        let initial = try await pool.acquireResource("rerank://rank")
        await pool.releaseResource(initial)
        let pressure = AccountingBarrier()
        await pool.setPressureAdmission { _ in await pressure.wait() }
        let admission = Task { try await pool.acquireResource("rerank://rank") }
        await pressure.waitUntilEntered()
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 0)
        admission.cancel()
        #expect(await pool.status().currentModelMemory == 40)
        await pressure.open()
        do {
            _ = try await admission.value
            Issue.record("Cancelled pressure waiter acquired a lease")
        } catch is CancellationError {}
        #expect(await pool.resourceStatuses().first?.inUse == 0)
        #expect(await pool.reclaimIdleModels(targetBytes: 100) == 40)
        #expect(await state.unloads == 1)
    }

    @Test func configuredAuxiliaryDescriptorsSupportCanonicalLifecycleBeforeInference() async throws {
        let pool = emptyPool(100)
        let rankCache = AccountingRerankCache()
        _ = await NativeRerankBackend(models: ["rank": model("rank", 20)], pool: pool, cache: rankCache)
        let adapter = AccountingSpeechAdapter(bytes: 20, work: AccountingBarrier())
        let speech = await PoolBackedSpeechAdapter(adapter: adapter, pool: pool)
        let registry = SpeechEngineRegistry()
        await registry.register(speech)
        let backend = await RegistrySpeechBackend(registry: registry)
        let statuses = await backend.speechModelStatuses()
        #expect(statuses.first?.loaded == false)
        #expect(await pool.resourceStatuses().count == 2)
        #expect(await rankCache.loads == 0)
        #expect(await adapter.loads == 0)
        let router = PoolResourceModelLifecycle(pool: pool)
        for canonical in ["rerank://rank", "speech://fixture/speech"] {
            #expect(try await router.load(canonical, allowBareModelID: false)?.modelID == canonical)
            #expect(try await router.unload(canonical, allowBareModelID: false)?.modelID == canonical)
        }
        #expect(await rankCache.loads == 1)
        #expect(await adapter.loads == 1)
        #expect(await rankCache.unloads == 1)
        #expect(await adapter.unloads == 1)
        #expect(await rankCache.scores == 0)
        #expect(await adapter.transcriptions == 0)
        #expect(try await router.load("rank", allowBareModelID: false) == nil)
        #expect(try await router.load("speech", allowBareModelID: false) == nil)
    }
}

private actor AuxiliaryPressureControl {
    private var denied = false
    private(set) var arguments: [Int64] = []
    func setDenied(_ denied: Bool) { self.denied = denied }
    func check(_ bytes: Int64) throws {
        arguments.append(bytes)
        if denied {
            throw MemoryWatchdogError.admissionDenied(requiredBytes: bytes, currentBytes: 99, ceilingBytes: 92)
        }
    }
}
