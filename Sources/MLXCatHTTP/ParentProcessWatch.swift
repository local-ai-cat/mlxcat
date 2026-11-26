import Foundation

/// Lets a server follow the process that launched it out of the door.
///
/// macOS has no "die with my parent" switch, so a child whose launcher is
/// SIGKILLed or crashes is reparented to launchd and keeps running. For
/// `mlxcat-http` that means an orphan holding the port, the memory and any
/// loaded model until somebody notices: on 2026-11-26 an app that crashed at
/// launch left one holding 127.0.0.1:11435 for 31 hours, and every later app
/// launch refused to start its own sidecar because the port was taken.
///
/// The launcher opts in by naming itself in `MLXCAT_PARENT_PID` (an environment
/// variable rather than a flag, so a launcher may set it for any build: an older
/// `mlxcat-http` rejects unknown flags but ignores unknown variables). The child
/// watches that pid with a kqueue `NOTE_EXIT` source and exits when it goes.
public enum ParentProcessWatch {
    public static let environmentKey = "MLXCAT_PARENT_PID"

    /// The launcher's pid from the environment, or nil when unset or unusable.
    /// launchd (pid 1) is never a launcher in this sense.
    public static func parentPID(environment: [String: String]) -> pid_t? {
        guard let raw = environment[environmentKey], let pid = pid_t(raw), pid > 1 else {
            return nil
        }
        return pid
    }

    /// Starts watching `parentPID`; `onParentExit` runs once, on a background
    /// queue, when that process exits — or at once if it has already gone,
    /// which is the race between being spawned and getting here.
    ///
    /// Keep the returned source alive for as long as the watch should run.
    public static func start(
        parentPID: pid_t,
        onParentExit: @escaping @Sendable () -> Void
    ) -> DispatchSourceProcess {
        let once = ParentExitOnce(onParentExit)
        let source = DispatchSource.makeProcessSource(
            identifier: parentPID,
            eventMask: .exit,
            queue: .global(qos: .utility)
        )
        source.setEventHandler { once.fire() }
        source.resume()
        // A kqueue registration on a pid that no longer exists never fires, so
        // the "already gone" case is checked after registering, not before:
        // checked first, the parent could exit in between and be missed.
        if kill(parentPID, 0) != 0, errno == ESRCH {
            once.fire()
        }
        return source
    }

    /// Watches the pid named in the environment, if any.
    public static func startFromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment,
        onParentExit: @escaping @Sendable () -> Void
    ) -> DispatchSourceProcess? {
        guard let parentPID = parentPID(environment: environment) else { return nil }
        return start(parentPID: parentPID, onParentExit: onParentExit)
    }
}

/// Runs the exit handler at most once: the kqueue event and the post-registration
/// liveness check can both see the same exit.
private final class ParentExitOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var handler: (@Sendable () -> Void)?

    init(_ handler: @escaping @Sendable () -> Void) {
        self.handler = handler
    }

    func fire() {
        lock.lock()
        let taken = handler
        handler = nil
        lock.unlock()
        taken?()
    }
}
