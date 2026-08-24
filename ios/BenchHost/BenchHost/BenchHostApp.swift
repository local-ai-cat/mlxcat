import SwiftUI

// The app exists to host BenchHostTests on a device; it renders only enough
// to prove it launched. All measurement lives in the test bundle so rows are
// produced headlessly by `xcodebuild test`, never by tapping.
@main
struct BenchHostApp: App {
    var body: some Scene {
        WindowGroup {
            VStack(spacing: 8) {
                Text("mlxcat BenchHost")
                    .font(.headline)
                Text("Run BenchHostTests via xcodebuild to produce leaderboard rows.")
                    .font(.caption)
                    .multilineTextAlignment(.center)
            }
            .padding()
            // The first device night died twice with "Test crashed with signal
            // kill" at unrelated times (41 min, then 10 min) — the signature of
            // auto-lock: iOS SIGKILLs Metal work the moment the screen locks.
            // Holding the idle timer keeps the phone awake while BenchHost is
            // frontmost; a manual lock press still kills the run, so the phone
            // sits untouched for the night.
            .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        }
    }
}
