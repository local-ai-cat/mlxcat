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
        }
    }
}
