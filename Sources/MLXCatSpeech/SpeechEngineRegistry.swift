import Foundation

/// Session hints a consumer can pass when it cares where inference runs — e.g.
/// the studio prefers ANE while the GPU is compositing a live stream.
public struct SpeechSessionPreferences: Sendable {
    public let preferredSilicon: SpeechEngineSilicon?
    public let requireStreaming: Bool

    public init(preferredSilicon: SpeechEngineSilicon? = nil, requireStreaming: Bool = false) {
        self.preferredSilicon = preferredSilicon
        self.requireStreaming = requireStreaming
    }
}

/// One home for every speech engine. Adapters register once; every consumer
/// (HTTP route, app picker, studio tap, consensus dispatcher) resolves models
/// through here. Model ids are namespaced per adapter catalog — the first
/// adapter listing a model id serves it; `engineID:modelID` disambiguates.
public actor SpeechEngineRegistry {
    private var adapters: [String: any SpeechEngineAdapter] = [:]
    private var registrationOrder: [String] = []

    public init() {}

    public func register(_ adapter: any SpeechEngineAdapter) {
        if adapters[adapter.engineID] == nil {
            registrationOrder.append(adapter.engineID)
        }
        adapters[adapter.engineID] = adapter
    }

    public func adapter(engineID: String) -> (any SpeechEngineAdapter)? {
        adapters[engineID]
    }

    public func allAdapters() -> [any SpeechEngineAdapter] {
        registrationOrder.compactMap { adapters[$0] }
    }

    public func allModels() async -> [SpeechModelInfo] {
        var models: [SpeechModelInfo] = []
        for adapter in allAdapters() {
            models.append(contentsOf: await adapter.availableModels())
        }
        return models
    }

    /// Resolve a model id (bare or `engineID:modelID`) to its serving adapter.
    public func resolve(
        model: String,
        preferences: SpeechSessionPreferences = SpeechSessionPreferences()
    ) async throws -> (adapter: any SpeechEngineAdapter, modelID: String) {
        let (candidates, modelID) = try await resolveCandidates(model: model, preferences: preferences)
        return (candidates[0], modelID)
    }

    /// All adapters that claim the model, best-first (silicon preference, then
    /// registration order). Callers that LOAD should try candidates in order —
    /// a claimed model can still fail to load (corrupt folder, missing files).
    public func resolveCandidates(
        model: String,
        preferences: SpeechSessionPreferences = SpeechSessionPreferences()
    ) async throws -> (adapters: [any SpeechEngineAdapter], modelID: String) {
        let (explicitEngine, modelID) = Self.splitModelReference(model)

        var candidates: [any SpeechEngineAdapter] = []
        for adapter in allAdapters() {
            if let explicitEngine, adapter.engineID != explicitEngine { continue }
            if preferences.requireStreaming && !adapter.capabilities.supportsStreaming { continue }
            let models = await adapter.availableModels()
            if models.contains(where: { $0.id == modelID }) {
                candidates.append(adapter)
            }
        }

        guard !candidates.isEmpty else {
            throw SpeechEngineError.unknownModel(model)
        }
        if let preferred = preferences.preferredSilicon {
            // Swift's sort is not stability-guaranteed; sort by (match, original index)
            // so registration order is preserved within each preference class.
            candidates = candidates.enumerated()
                .sorted { lhs, rhs in
                    let lhsKey = (lhs.element.capabilities.silicon == preferred ? 0 : 1, lhs.offset)
                    let rhsKey = (rhs.element.capabilities.silicon == preferred ? 0 : 1, rhs.offset)
                    return lhsKey < rhsKey
                }
                .map(\.element)
        }
        return (candidates, modelID)
    }

    static func splitModelReference(_ model: String) -> (engineID: String?, modelID: String) {
        guard let separator = model.firstIndex(of: ":") else {
            return (nil, model)
        }
        let engine = String(model[..<separator])
        let rest = String(model[model.index(after: separator)...])
        guard !engine.isEmpty, !rest.isEmpty else {
            return (nil, model)
        }
        return (engine, rest)
    }
}
