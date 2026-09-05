import Foundation
import Network

private final class ListenerStartState: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?

    init(_ continuation: CheckedContinuation<Void, Error>) {
        self.continuation = continuation
    }

    func resume(_ result: Result<Void, Error>) {
        lock.lock()
        let continuation = continuation
        self.continuation = nil
        lock.unlock()

        guard let continuation else { return }
        switch result {
        case .success:
            continuation.resume(returning: ())
        case .failure(let error):
            continuation.resume(throwing: error)
        }
    }
}

public final class OpenAIServer: @unchecked Sendable {
    private let host: NWEndpoint.Host
    private let port: NWEndpoint.Port
    let backend: any OpenAIChatBackend
    private let listener: NWListener
    let responsesStore: ResponsesStore
    let mcpManager: MCPManager?
    private let connectionsLock = NSLock()
    private var openConnections: [ObjectIdentifier: NWConnection] = [:]
    private var isStopped = false

    public init(
        host: String = "127.0.0.1",
        port: UInt16,
        backend: any OpenAIChatBackend,
        mcpManager: MCPManager? = nil
    ) throws {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw OpenAIServerError.invalidPort(port)
        }
        self.host = NWEndpoint.Host(host)
        self.port = nwPort
        self.backend = backend
        // Loopback hosts must actually bind loopback-only: the `host` parameter
        // used to be stored but never applied, so the listener accepted
        // connections on every interface regardless of it.
        let parameters: NWParameters = .tcp
        if ["127.0.0.1", "localhost", "::1"].contains(host) {
            parameters.requiredInterfaceType = .loopback
        }
        self.listener = try NWListener(using: parameters, on: nwPort)
        self.responsesStore = ResponsesStore()
        self.mcpManager = mcpManager
    }

    /// The port the listener is actually bound to — differs from the requested
    /// port when constructed with port 0 (ephemeral). Nil until `start()` returns.
    public var boundPort: UInt16? {
        listener.port?.rawValue
    }

    public func start() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let startState = ListenerStartState(continuation)
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    startState.resume(.success(()))
                case .failed(let error):
                    startState.resume(.failure(OpenAIServerError.listenerFailed(String(describing: error))))
                case .cancelled:
                    startState.resume(.failure(OpenAIServerError.listenerFailed("listener cancelled")))
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                guard let self, self.track(connection) else {
                    connection.cancel()
                    return
                }
                connection.start(queue: .global(qos: .userInitiated))
                Task {
                    await self.handle(connection)
                    self.untrack(connection)
                }
            }
            listener.start(queue: .global(qos: .userInitiated))
        }
    }

    /// Stops accepting new connections and cancels the in-flight ones, releasing
    /// the port. Idempotent. Embedded hosts (in-app serving) use this; the CLI
    /// never stops. The instance cannot be restarted — create a new server.
    public func stop() {
        connectionsLock.lock()
        let alreadyStopped = isStopped
        isStopped = true
        let connections = Array(openConnections.values)
        openConnections.removeAll()
        connectionsLock.unlock()

        guard !alreadyStopped else { return }
        listener.cancel()
        for connection in connections {
            connection.cancel()
        }
    }

    private func track(_ connection: NWConnection) -> Bool {
        connectionsLock.lock()
        defer { connectionsLock.unlock() }
        guard !isStopped else { return false }
        openConnections[ObjectIdentifier(connection)] = connection
        return true
    }

    private func untrack(_ connection: NWConnection) {
        connectionsLock.lock()
        openConnections.removeValue(forKey: ObjectIdentifier(connection))
        connectionsLock.unlock()
        // NWConnection keeps its file descriptor until cancel() is called, even
        // after the peer closes (lsof shows the socket parked in (CLOSED)). This
        // server is connection-per-request, so every handled connection ends
        // here — without this cancel the process leaks one fd per request and
        // wedges at the 256-fd rlimit (~245 requests served, then EMFILE).
        connection.cancel()
    }

    public func waitForever() {
        DispatchSemaphore(value: 0).wait()
    }

}
