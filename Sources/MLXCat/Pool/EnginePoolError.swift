import Foundation

public enum EnginePoolError: Error, CustomStringConvertible, Equatable, Sendable {
    case modelNotFound(id: String, available: [String])
    case modelLoading(id: String)
    case modelBusy(id: String, operation: String)
    case modelTooLarge(id: String, size: Int64, ceiling: Int64)
    case insufficientMemory(id: String, required: Int64, current: Int64, ceiling: Int64)
    case schedulerQueueFull(current: Int, max: Int)
    case modelNotLoaded(id: String)

    public var httpStatus: Int {
        switch self {
        case .modelNotFound:
            return 404
        case .modelLoading, .modelBusy:
            return 409
        case .modelTooLarge, .insufficientMemory:
            return 507
        case .schedulerQueueFull:
            return 503
        case .modelNotLoaded:
            return 400
        }
    }

    public var retryAfterSeconds: Int? {
        switch self {
        case .schedulerQueueFull:
            return 1
        default:
            return nil
        }
    }

    public var message: String {
        switch self {
        case .modelNotFound(let id, let available):
            let availableList = available.isEmpty ? "(none)" : available.sorted().joined(separator: ", ")
            return "Model '\(id)' not found. Available models: \(availableList)"
        case .modelLoading(let id):
            return "Model '\(id)' is currently loading"
        case .modelBusy(let id, let operation):
            return "Model '\(id)' is busy and cannot \(operation)"
        case .modelTooLarge(let id, let size, let ceiling):
            return "Model '\(id)' is too large: \(ModelDiscovery.formatSize(size)) exceeds memory ceiling \(ModelDiscovery.formatSize(ceiling))"
        case .insufficientMemory(let id, let required, let current, let ceiling):
            let projected = current + required
            return "Cannot load \(id): projected memory \(ModelDiscovery.formatSize(projected)) would exceed the memory ceiling \(ModelDiscovery.formatSize(ceiling)) (current: \(ModelDiscovery.formatSize(current)), model: \(ModelDiscovery.formatSize(required))). Free system memory or lower memory_guard_tier."
        case .schedulerQueueFull(let current, let max):
            return "Scheduler waiting queue full (\(current)/\(max)). Try again shortly."
        case .modelNotLoaded(let id):
            return "Model not loaded: \(id)"
        }
    }

    public var description: String {
        message
    }
}
