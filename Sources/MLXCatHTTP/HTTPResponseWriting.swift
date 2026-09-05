import Foundation
import Network

/// The response effects used by dialect handlers; production uses NWConnection.
protocol HTTPResponseWriting: Sendable {
    func send(data: Data) async throws
    func sendFinal(data: Data) async throws
}

extension NWConnection: HTTPResponseWriting {}
