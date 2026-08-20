import Foundation

/// Speech-model lifecycle routing that `PoolBackedChatBackend` can consult
/// without a compile-time dependency on the concrete WhisperKit-backed speech
/// backend. This lets the chat backend live in a library (`MLXCatNative`) that
/// never links WhisperKit; only the server executable supplies a conforming
/// speech backend, and in-process embedders (e.g. the iOS app) simply pass `nil`.
public protocol SpeechModelLifecycleRouting: AnyObject, Sendable {
    func speechModelStatuses() async -> [OpenAIModelRuntimeStatus]
    func isNamespacedSpeechModelReference(_ id: String) -> Bool
    func canResolveModelReference(_ id: String) async -> Bool
    func loadModel(_ id: String) async throws -> OpenAIModelLifecycleResult
    func unloadModel(_ id: String) async throws -> OpenAIModelLifecycleResult
}
