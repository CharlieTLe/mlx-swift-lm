import AppKit
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import SwiftUI

@main
struct MuseGlimmerDemoApp: App {
    init() {
        // An unbundled SwiftPM executable launches as an accessory process, which
        // gets no focused window and no menu bar. Promote it to a regular app.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Muse-Glimmer") {
            ContentView()
                .frame(minWidth: 760, minHeight: 560)
        }
        .windowResizability(.contentSize)
    }
}
