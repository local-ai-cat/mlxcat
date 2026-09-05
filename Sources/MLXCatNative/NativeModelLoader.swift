import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXVLM
import Tokenizers

// NOTE: Cross-model concurrent MLX eval is not process-serialized here. Each
// model has its own scheduler; a process-wide MLX eval gate is a known follow-up
// if native serving needs to mirror omlx's single executor more tightly.
public struct NativeModelLoader: EnginePoolModelLoader {
    let maxConcurrentRequests: Int

    public init(maxConcurrentRequests: Int) {
        self.maxConcurrentRequests = maxConcurrentRequests
    }

    /// Brackets the load with an MLX active-memory snapshot so the pool can
    /// account for the model by its real resident footprint rather than the
    /// discovery-time estimate. `Memory.activeMemory` counts live `MLXArray`
    /// allocations (the model weights), distinct from the reclaimable cache pool
    /// (`Memory.cacheMemory`). The pool serializes loads behind its load gate, so
    /// concurrent pool loads cannot overlap this measurement. Ongoing generation
    /// or freeing can still contaminate the process-wide delta; it is not an
    /// isolated per-model working-set measurement. A non-positive delta reports `nil`,
    /// falling the pool back to the estimate.
    public func loadModelMeasuringActualSize(
        id: String,
        modelURL: URL
    ) async throws -> EnginePoolMeasuredLoad<NativeModelEngine> {
        let activeBefore = Memory.activeMemory
        let engine = try await loadModel(id: id, modelURL: modelURL)
        let delta = Memory.activeMemory - activeBefore
        let measured: Int64? = delta > 0 ? Int64(delta) : nil
        return EnginePoolMeasuredLoad(engine: engine, actualSizeBytes: measured)
    }

    public func loadModel(id: String, modelURL: URL) async throws -> NativeModelEngine {
        let modelConfiguration = try modelConfiguration(in: modelURL)
        let modelType = modelConfiguration.modelType.lowercased()
        let isVLM = try isVLMModelDirectory(modelURL, modelType: modelType)
        let container =
            if isVLM {
                try await VLMModelFactory.shared.loadContainer(
                    from: modelURL,
                    using: #huggingFaceTokenizerLoader()
                )
            } else {
                try await LLMModelFactory.shared.loadContainer(
                    from: modelURL,
                    using: #huggingFaceTokenizerLoader()
                )
            }
        return await container.perform { context in
            NativeModelEngine(
                context: context,
                modelID: id,
                maxContextTokens: try? contextWindow(in: modelURL),
                maxConcurrentRequests: maxConcurrentRequests,
                cacheCapabilities: Self.cacheCapabilities(for: modelConfiguration),
                serializationPolicy: Self.serializationPolicy(modelType: modelType, isVLM: isVLM),
                schedulerManagedTextPrefill: !isVLM,
                chunkIdlePrefill: Self.chunksIdlePrefill(modelType: modelType),
                // Resolved per model, not per process: `gpt_oss` is excluded
                // because its quantized attention route drops attention sinks.
                kvQuantization: KVQuantizationPolicy.resolved(
                    .fromEnvironment(), modelType: modelType
                )
            )
        }
    }

    public func unloadModel(_ engine: NativeModelEngine, id: String) async {
        await engine.shutdown()
        Memory.clearCache()
    }

    private func modelConfiguration(in modelURL: URL) throws -> NativeModelKindConfiguration {
        let configURL = modelURL.appending(component: "config.json")
        let configData = try Data(contentsOf: configURL)
        return try JSONDecoder.json5().decode(NativeModelKindConfiguration.self, from: configData)
    }

    func contextWindow(in modelURL: URL) throws -> Int? {
        let configURL = modelURL.appending(component: "config.json")
        let data = try Data(contentsOf: configURL)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let textConfig = object["text_config"] as? [String: Any]
        let candidates = [
            object["max_position_embeddings"],
            object["max_sequence_length"],
            object["model_max_length"],
            textConfig?["max_position_embeddings"],
            textConfig?["max_sequence_length"],
            textConfig?["model_max_length"]
        ]
        return candidates.compactMap { ($0 as? NSNumber)?.intValue }.first { $0 > 0 }
    }

    func cacheCapabilities(in modelURL: URL) throws -> ModelCacheCapabilities {
        try Self.cacheCapabilities(for: modelConfiguration(in: modelURL))
    }

    private func isVLMModelDirectory(_ modelURL: URL, modelType: String) throws -> Bool {
        if let processorClass = processorClass(in: modelURL),
           Self.vlmProcessorClasses.contains(processorClass) {
            return true
        }

        guard Self.vlmModelTypes.contains(modelType) else {
            return false
        }
        return hasProcessorConfiguration(in: modelURL)
    }

}
