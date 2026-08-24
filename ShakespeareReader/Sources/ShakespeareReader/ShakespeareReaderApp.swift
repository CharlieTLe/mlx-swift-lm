import Combine
import Dispatch
import Foundation
import MLX
import SwiftUI

#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// Command-line options.
///
/// The app normally launches with no arguments. These flags exist for the work
/// that is easier from a terminal than through the UI: inspecting the assembled
/// prompt, pinning sampling for prompt A/B comparisons, and the model-free self
/// test.
struct AppOptions: Sendable {
    /// Run the model-free assertions and exit. No download, no network.
    var runSelfTest = false
    /// Evaluate one tiny array on the GPU and exit. No download, no network.
    var runMetalCheck = false
    /// Print the assembled prompt and its exact token count for a sample of
    /// passages, then exit. Needs the tokenizer, so it does load the model.
    var showPrompt = false
    /// Run the real annotation path over a sample of passages and report the
    /// latency numbers, then exit. This is how the README's table is produced, and
    /// how a prompt change is checked against the token budget.
    var benchmark = false
    /// `temperature: 0, seed: 0` everywhere, so two runs of the same prompt are
    /// byte-identical and a prompt edit is the only variable.
    var greedy = false
    /// Show the model capsule, the load check and the latency numbers for this launch,
    /// without persisting the preference.
    var diagnostics = false
    /// Overrides the model, e.g. `--model mlx-community/Qwen3-8B-4bit`.
    var modelID: String?
    /// Passages for `--show-prompt`, as `play:act.scene:first-last`. Empty means
    /// the built-in sample.
    var passages: [String] = []

    static func parse(_ arguments: [String]) -> AppOptions {
        var options = AppOptions()
        var rest = arguments.makeIterator()
        while let argument = rest.next() {
            switch argument {
            case "--selftest": options.runSelfTest = true
            case "--metal-check": options.runMetalCheck = true
            case "--show-prompt": options.showPrompt = true
            case "--benchmark": options.benchmark = true
            case "--greedy": options.greedy = true
            case "--diagnostics": options.diagnostics = true
            case "--model": options.modelID = rest.next()
            case "--passage": options.passages.append(rest.next() ?? "")
            default: break
            }
        }
        return options
    }
}

/// The one-line GPU check behind `--metal-check`.
///
/// `--selftest` decodes the corpus and renders prompts, and never touches the
/// GPU, so it passes on a build whose Metal kernels were never compiled.
/// mlx-swift's `.metal` sources become a `default.metallib` inside
/// `mlx-swift_Cmlx.bundle`, which is looked up beside the running executable and
/// only on first GPU use, where a miss surfaces from C++ as "Failed to load the
/// default metallib": long after launch, in the middle of annotating a passage.
/// Evaluating one array proves the kernels are there before a user finds out the
/// slow way, which is what makes this worth a flag of its own.
enum MetalCheck {
    static func run() -> Bool {
        let sum = MLXArray([1, 2, 3]).sum(stream: .gpu)
        eval(sum)
        guard sum.item(Int32.self) == 6 else {
            FileHandle.standardError.write(Data("metal: wrong result\n".utf8))
            return false
        }
        print("metal: ok")
        return true
    }
}

/// Separate from the `App` so the headless flags can run before SwiftUI starts.
///
/// `App` supplies its own `static main()`, and there is no way to call that
/// default implementation from an override of it. Owning the entry point and
/// forwarding to `ShakespeareReaderApp.main()` is the way to get a
/// `--selftest` that exits without ever opening a window.
@main
enum EntryPoint {
    /// `@MainActor` so it can hand the parsed options to the `App`, and because
    /// this is the main thread at process start either way.
    @MainActor
    static func main() {
        let options = AppOptions.parse(Array(CommandLine.arguments.dropFirst()))

        // The headless flags are terminal work and macOS-only. `dispatchMain()` and
        // `exit()` have no place in an iOS app, where an app that exits itself on
        // launch reads to the system as a crash, and there is no terminal to print a
        // prompt dump or a benchmark table to. iOS parses the arguments and ignores them,
        // which leaves `--diagnostics` and `--model` working under
        // `xcrun simctl launch`.
        #if os(macOS)
        if options.runSelfTest {
            // Entirely synchronous and off the main actor, so it can run right here.
            exit(SelfTest.run() ? 0 : 1)
        }

        if options.runMetalCheck {
            exit(MetalCheck.run() ? 0 : 1)
        }

        if options.showPrompt || options.benchmark {
            // The prompt dump needs the main actor (it drives `AnnotationService`),
            // so the main thread has to keep servicing it rather than block on a
            // semaphore — that would deadlock against the main-actor executor.
            // `dispatchMain()` parks the main thread on the main queue and never
            // returns; the task exits the process itself.
            Task {
                let ok =
                    options.benchmark
                    ? await Benchmark.run(options: options)
                    : await PromptDump.run(options: options)
                exit(ok ? 0 : 1)
            }
            dispatchMain()
        }
        #endif

        ShakespeareReaderApp.options = options
        ShakespeareReaderApp.main()
    }
}

struct ShakespeareReaderApp: App {
    /// Set by `EntryPoint` before SwiftUI starts. `static` because `App` is
    /// initialized by the framework, so there is no initializer to pass through.
    @MainActor static var options = AppOptions()

    init() {
        // An unbundled SwiftPM executable launches as an accessory process, which
        // gets no focused window and no menu bar. Promote it to a regular app.
        //
        // Nothing to do in a bundled app, Mac or iPhone: it is launched by the system
        // as a foreground application, so there is no accessory state to escape. Both
        // calls are harmless there, so this stays unconditional on macOS rather than
        // growing a bundled/unbundled test.
        #if os(macOS)
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
        Self.setUnbundledDockIcon()
        #endif
    }

    #if os(macOS)
    /// Puts the portrait on the Dock tile of an unbundled build.
    ///
    /// The `.app` needs none of this. `App/Assets.xcassets` compiles to an `Assets.car`
    /// and a `CFBundleIconName`, and the Dock, Finder, Spotlight and Cmd-Tab all read
    /// that straight out of the bundle without the process being consulted. A bare
    /// SwiftPM executable has no bundle to read, so it takes the generic
    /// unbundled-executable tile however complete the asset catalog is, and
    /// `applicationIconImage` is the only way to change it — which also means the icon
    /// appears a moment after launch rather than the instant the tile does.
    ///
    /// Guarded on the nil bundle identifier rather than on `#if SWIFT_PACKAGE` alone,
    /// because those are not the same question. The `#if` is needed for `Bundle.module`
    /// to exist at all — Xcode does not synthesize it for the app target, and does not
    /// define `SWIFT_PACKAGE` there — but a SwiftPM-built binary can still be run from
    /// inside a `.app` someone assembled around it, where `Bundle.main` resolves to the
    /// enclosing bundle and the catalog's icon is the better one. Missing identifier is
    /// what actually means "no bundle behind me", and is the same test the README's note
    /// about split preferences turns on.
    private static func setUnbundledDockIcon() {
        #if SWIFT_PACKAGE
        guard Bundle.main.bundleIdentifier == nil,
            let url = Bundle.module.url(forResource: "AppIcon", withExtension: "png"),
            let icon = NSImage(contentsOf: url)
        else { return }
        NSApplication.shared.applicationIconImage = icon
        #endif
    }
    #endif

    // `SwiftUI.Scene` in full: this app's corpus has its own `Scene` type, and an
    // unqualified `some Scene` resolves to that one.
    var body: some SwiftUI.Scene {
        WindowGroup("Shakespeare Reader") {
            ContentView(options: Self.options)
                // The three panes need 210 + 420 + 380 = 1010 with all of them open, so
                // the floor has to clear that or the reader gets pushed under its own
                // minimum. AppKit clamps an autosaved frame up to a raised minimum on the
                // next launch, so a window left narrower than this widens once and holds.
                //
                // macOS only: on a phone this would force the content wider than the
                // screen.
                #if os(macOS)
            .frame(minWidth: 1080, minHeight: 640)
                #else
            // The 4B model's working set is a large fraction of what iOS will
            // let one app hold, and MLX's buffer-reuse pool is the part of it
            // that is safe to give back: the weights are still needed, cached
            // scratch buffers are not. Dropping them on a warning is what makes
            // the difference between the system reclaiming memory and jetsam
            // killing the app mid-annotation.
            .onReceive(
                NotificationCenter.default.publisher(
                    for: UIApplication.didReceiveMemoryWarningNotification)
            ) { _ in
                guard hasMLXDevice else { return }
                Memory.clearCache()
            }
                #endif
        }
    }
}
