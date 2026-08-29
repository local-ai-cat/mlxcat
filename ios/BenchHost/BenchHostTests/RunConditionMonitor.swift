import UIKit
import XCTest

/// Measures the run conditions the RUNBOOK previously asked a human to attest.
///
/// The Mac harness has a quiet-machine guard (loadavg, free memory, thermal
/// speed limit) and stamps its evidence into every row's `host` object; the
/// phone had nothing, so every iOS row shipped `valid_for_leaderboard: false`
/// with "operator must confirm run conditions" — and no row was ever promoted,
/// because a human attestation after a multi-hour night is exactly the check
/// that never returns true. This monitor replaces the attestation with
/// measurement for everything the platform can actually observe:
///
///   on power        UIDevice.batteryState (charging/full at every sample)
///   cool            ProcessInfo.thermalState (worst observed; serious+ fails)
///   unlocked        UIApplication.isProtectedDataAvailable at every sample —
///                   iOS SIGKILLs Metal work behind a locked screen, so a lock
///                   mid-cell also explains a dead session
///   foregrounded    UIApplication.applicationState == .active at every sample
///   full speed      ProcessInfo.isLowPowerModeEnabled (LPM caps perf cores)
///
/// Each row receives the WORST state observed since the previous row was
/// emitted, so a mid-night thermal event invalidates exactly the cells it
/// touched, not the whole night. What remains unmeasurable — a finger on the
/// screen that never locks or backgrounds the app — cannot bias a row by more
/// than the compositor cost of rendering the static host view, so the measured
/// set is the promotion gate.
final class RunConditionMonitor: @unchecked Sendable {

    struct Snapshot {
        var worstThermal: ProcessInfo.ThermalState = .nominal
        var everOnBattery = false
        var everBatteryUnknown = false
        var everLocked = false
        var everBackgrounded = false
        var everLowPower = false
        var lastBatteryLevel: Float = -1
        var sampleCount = 0

        var thermalName: String {
            switch worstThermal {
            case .nominal: return "nominal"
            case .fair: return "fair"
            case .serious: return "serious"
            case .critical: return "critical"
            @unknown default: return "unknown"
            }
        }

        /// Empty means every measured condition held for every sample.
        /// `fair` thermal is allowed: a phone charging under load sits at fair
        /// routinely; `serious` is the state where iOS begins throttling the
        /// GPU, which is the point where a perf number stops being the device's.
        var violations: [String] {
            var out: [String] = []
            if sampleCount == 0 { out.append("no condition samples") }
            if everOnBattery { out.append("ran on battery") }
            if everBatteryUnknown { out.append("battery state unreadable") }
            if worstThermal == .serious || worstThermal == .critical {
                out.append("thermal \(thermalName)")
            }
            if everLocked { out.append("screen locked during cell") }
            if everBackgrounded { out.append("app left foreground") }
            if everLowPower { out.append("low power mode") }
            return out
        }

        /// The `host` evidence object for the row — the iOS analog of the Mac
        /// rows' loadavg/memory/thermal block. Promotion decisions must be
        /// reconstructable from the row alone.
        var hostEvidence: [String: Any] {
            let batteryLevel: Any = lastBatteryLevel >= 0 ? lastBatteryLevel as Any : NSNull()
            return [
                "thermal_state_worst": thermalName,
                "on_power_throughout": !everOnBattery && !everBatteryUnknown,
                "battery_level": batteryLevel,
                "screen_locked_ever": everLocked,
                "app_backgrounded_ever": everBackgrounded,
                "low_power_mode_ever": everLowPower,
                "condition_samples": sampleCount,
            ]
        }
    }

    private let lock = NSLock()
    private var current = Snapshot()
    private var task: Task<Void, Never>?

    /// Samples every 2 seconds until `stop()`. Battery monitoring must be
    /// switched on or `batteryState` reads `.unknown` forever — an absent
    /// check reading as a pass is exactly what this class exists to prevent,
    /// so `.unknown` is a violation, never a default-pass.
    func start() {
        task = Task { [weak self] in
            await MainActor.run { UIDevice.current.isBatteryMonitoringEnabled = true }
            while !Task.isCancelled {
                await self?.sampleOnce()
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    /// The worst conditions observed since the last call, then reset — one
    /// call per emitted row gives per-cell evidence.
    func snapshotAndReset() -> Snapshot {
        lock.lock()
        defer {
            current = Snapshot()
            lock.unlock()
        }
        return current
    }

    @MainActor
    private func sampleOnce() {
        let device = UIDevice.current
        let application = UIApplication.shared
        let process = ProcessInfo.processInfo

        let batteryState = device.batteryState
        let thermal = process.thermalState
        let locked = !application.isProtectedDataAvailable
        let backgrounded = application.applicationState != .active
        let lowPower = process.isLowPowerModeEnabled
        let level = device.batteryLevel

        lock.lock()
        if thermal.rawValue > current.worstThermal.rawValue { current.worstThermal = thermal }
        switch batteryState {
        case .charging, .full: break
        case .unplugged: current.everOnBattery = true
        case .unknown: current.everBatteryUnknown = true
        @unknown default: current.everBatteryUnknown = true
        }
        if locked { current.everLocked = true }
        if backgrounded { current.everBackgrounded = true }
        if lowPower { current.everLowPower = true }
        if level >= 0 { current.lastBatteryLevel = level }
        current.sampleCount += 1
        lock.unlock()
    }
}
