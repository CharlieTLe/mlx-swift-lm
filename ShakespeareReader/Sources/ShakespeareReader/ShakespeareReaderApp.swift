import AppKit
import Dispatch
import Foundation
import MLX
import SwiftUI

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
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    // `SwiftUI.Scene` in full: this app's corpus has its own `Scene` type, and an
    // unqualified `some Scene` resolves to that one.
    var body: some SwiftUI.Scene {
        WindowGroup("Shakespeare Reader") {
            ContentView(options: Self.options)
                .frame(minWidth: 1000, minHeight: 640)
        }
    }
}
