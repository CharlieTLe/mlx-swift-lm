import MLXLLM
import MLXLMCommon
import SwiftUI

@MainActor
struct ContentView: View {
    let options: AppOptions

    @State private var service: AnnotationService
    @State private var corpus: Corpus?
    @State private var corpusError: String?
    @State private var casts: [String: Cast] = [:]

    @State private var sceneKey: SceneKey?
    @State private var selection: LineSelection?
    @State private var context: PassageContext?

    /// Which side panes are on screen. Hiding them is how the reader gets the play on
    /// its own, so how they left them is how they come back. Persisted the same way
    /// `readerFont` is, below. The commentary comes back on the next pointer selection,
    /// since clicking a passage is a request to have it glossed.
    @AppStorage("showsNavigator") private var showsNavigator = true
    @AppStorage("showsCommentary") private var showsCommentary = true

    /// Lives here, not in `NavigatorView`, so collapsed acts survive hiding the pane.
    /// Seeded from the last session in `init(options:)`.
    @State private var collapsedActs: Set<String> = []

    /// The face the *play text* is set in; everything else stays on the system face.
    ///
    /// This persists even though the app is an unbundled SwiftPM executable with no
    /// bundle id, because CFPreferences falls back to the process name:
    /// `~/Library/Preferences/ShakespeareReader.plist` is already where SwiftUI keeps
    /// this window's frame.
    @AppStorage("readerFont") private var readerFont: ReaderFont = .system

    /// Off by default: the model name, the load check and the latency numbers are for
    /// working on the app, not for reading a play. Persisted the same way `readerFont`
    /// is, and forced on for a launch by `--diagnostics`.
    @AppStorage("showsDiagnostics") private var diagnosticsPreference = false

    private var showsDiagnostics: Bool { diagnosticsPreference || options.diagnostics }

    /// Which families are installed, and the on-demand download for Garamond.
    @State private var fonts = ReaderFontLibrary()

    @State private var commentary = ""
    @State private var followUps: [String] = []
    @State private var transcript: [AnnotationPaneView.Exchange] = []
    @State private var phase: Phase = .idle
    @State private var promptTokens: Int?
    @State private var prefill: (processed: Int, total: Int)?
    @State private var timeToFirstToken: TimeInterval?
    @State private var stats: GenerationStats?
    @State private var errorMessage: String?

    init(options: AppOptions) {
        self.options = options
        _service = State(
            initialValue: AnnotationService(
                modelID: options.modelID ?? LLMRegistry.qwen3_4b_4bit.name,
                greedy: options.greedy))
        _collapsedActs = State(initialValue: ProgressStore.collapsedActs())
    }

    var body: some View {
        layout
            .task {
                loadCorpus()
                await service.load()
                if let corpus, let sceneKey, let scene = corpus.scene(sceneKey) {
                    service.rememberSceneForPrewarm(key: sceneKey, scene: scene)
                    if showsCommentary {
                        service.prewarmSynopsis(key: sceneKey, scene: scene)
                    }
                }
            }
            // Recorded here rather than inside `openScene(_:in:)`, so every path that
            // moves the reader is caught, including an arrow-key move, which comes back
            // through `SceneReaderView`'s `$selection` binding and never calls it. No
            // debounce: CFPreferences coalesces writes, so a held arrow key is not a
            // per-keypress disk hit.
            .onChange(of: sceneKey) { recordProgress() }
            .onChange(of: selection) { recordProgress() }
            .onChange(of: collapsedActs) {
                ProgressStore.save(collapsedActs: collapsedActs)
            }
    }

    // MARK: - Layout

    /// The one place the two platforms genuinely diverge. Both branches build their
    /// panes from the same three `@ViewBuilder` helpers below, so what differs here is
    /// the *container* and nothing else: a Mac shows all three at once, a phone shows
    /// one at a time and raises the commentary over the verse.
    @ViewBuilder
    private var layout: some View {
        #if os(macOS)
        VStack(spacing: 0) {
            header
            Divider()
            desktopPanes
        }
        #else
        phonePanes
        #endif
    }

    #if os(macOS)
    @ViewBuilder
    private var desktopPanes: some View {
        if let corpus, let sceneKey, let scene = corpus.scene(sceneKey),
            let play = corpus.play(sceneKey.playID)
        {
            // The side panes are fixed widths, and that is what keeps the navigator
            // still. `HSplitView` distributes width from each child's min / ideal /
            // max, and it will move *any* pane that has room between them: a `maxWidth`
            // caps how far a pane can grow but leaves it just as free to be shrunk. So
            // a navigator declared 180…320 gets squeezed whenever the split view is
            // over-subscribed, and it is over-subscribed exactly when the middle pane's
            // reported ideal grows — which happens on every scene change, since
            // `SceneReaderView` is a fresh identity per scene whose heading is as wide
            // as the setting string. Pinning the sides leaves the reader as the only
            // pane with any give, so it absorbs the whole difference and the dividers
            // never move. Giving either side a width *range* hands that pane back to
            // the split view and the drift returns.
            //
            // The cost is that the dividers are no longer draggable. 210 + 380 here
            // plus the reader's 420 floor is the 1010 that sets the window minimum in
            // `ShakespeareReaderApp`.
            HSplitView {
                if showsNavigator {
                    navigatorPane(corpus: corpus, key: sceneKey)
                        .frame(width: 210)
                }

                readerPane(corpus: corpus, play: play, key: sceneKey, scene: scene)
                    // The only pane with any flexibility, so every width change lands
                    // here. The explicit `idealWidth` keeps the scene heading from
                    // proposing the window's preferred width: without one this frame
                    // propagates the child's own ideal, which is the full single-line
                    // width of whichever setting string happens to be on screen.
                    .frame(minWidth: 420, idealWidth: 640, maxWidth: .infinity)

                if showsCommentary {
                    commentaryPane().frame(width: 380)
                }
            }
        } else if let corpusError {
            failure(corpusError)
        } else {
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
    #else
    /// Navigator and reader as the two columns of a `NavigationSplitView`, which an
    /// iPhone always renders collapsed, so it is a push from the scene list to the
    /// reader, with the system's own back button standing in for ⌘1.
    ///
    /// The commentary is an `.inspector` specifically because that container
    /// auto-presents as a **sheet** at this size class: the verse stays on screen
    /// above the gloss, which is the whole point of the three-pane desktop layout
    /// and the one part of it worth keeping on a phone.
    @ViewBuilder
    private var phonePanes: some View {
        NavigationSplitView(columnVisibility: navigatorVisibility) {
            Group {
                if let corpus, let sceneKey {
                    navigatorPane(corpus: corpus, key: sceneKey)
                } else if let corpusError {
                    failure(corpusError)
                } else {
                    ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .navigationTitle("Plays")
        } detail: {
            Group {
                if let corpus, let sceneKey, let scene = corpus.scene(sceneKey),
                    let play = corpus.play(sceneKey.playID)
                {
                    readerPane(
                        corpus: corpus, play: play, key: sceneKey, scene: scene
                    )
                    .navigationTitle(play.title)
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar { readerToolbar }
                } else {
                    ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        // Presented by *having a gloss*, not by the `showsCommentary` preference.
        //
        // On a Mac that preference is a layout question (is the third pane on screen)
        // and the reader answers it once. On a phone the pane is a sheet over the verse,
        // so "is it up" is not a preference at all but a moment: it rises when a passage
        // is glossed and it is done when the reader swipes it away. Binding it to the
        // stored preference instead put the placeholder sheet over the play on first
        // launch, before anything had been selected.
        //
        // Dismissing calls `cancel()`, which is Esc's behaviour minus clearing the
        // selection, so a gloss nobody will read stops generating. It deliberately does
        // *not* write `showsCommentary`: that flag still gates `commit(_:)` and the
        // scene-summary prewarm, and turning it off here would quietly stop both for
        // the rest of the session.
        .inspector(
            isPresented: Binding(
                get: { context != nil }, set: { if !$0 { cancel() } })
        ) {
            commentaryPane()
        }
    }

    /// The persisted navigator preference, in the shape `NavigationSplitView` wants.
    /// Reused rather than duplicated: collapsed, the split view writes `.detailOnly`
    /// on a push and `.all` on a pop, so "was I reading or browsing" survives a
    /// relaunch off the same key the Mac uses.
    private var navigatorVisibility: Binding<NavigationSplitViewVisibility> {
        Binding(
            get: { showsNavigator ? .all : .detailOnly },
            set: { showsNavigator = $0 != .detailOnly })
    }
    #endif

    // MARK: - Panes

    @ViewBuilder
    private func navigatorPane(corpus: Corpus, key: SceneKey) -> some View {
        NavigatorView(
            corpus: corpus,
            key: Binding(get: { key }, set: { openScene($0, in: corpus) }),
            collapsed: $collapsedActs)
    }

    @ViewBuilder
    private func readerPane(corpus: Corpus, play: Play, key: SceneKey, scene: Scene)
        -> some View
    {
        SceneReaderView(
            play: play, key: key, scene: scene,
            cast: cast(for: play),
            selection: $selection,
            onCommit: { selection, origin in
                commit(
                    selection, play: play, key: key, scene: scene,
                    revealingCommentary: origin == .pointer)
            },
            onCancel: { cancel() },
            onRegenerate: { regenerate() },
            onStepScene: { step in stepScene(step, in: corpus) }
        )
        .environment(\.readerTypeface, fonts.typeface(for: readerFont))
    }

    @ViewBuilder
    private func commentaryPane() -> some View {
        VStack(spacing: 0) {
            AnnotationPaneView(
                citation: context?.citation,
                selectedLines: context?.selected ?? [],
                commentary: commentary,
                followUps: followUps,
                transcript: transcript,
                isBusy: isBusy,
                synopsisIsPartial: context?.synopsisIsPartial ?? false,
                onAsk: ask)
            // Both go together: a divider with nothing under it would
            // leave an empty band at the foot of the pane.
            if showsDiagnostics || errorMessage != nil {
                Divider()
                statusStrip
            }
        }
        // Half height by default, which is the entire reason the commentary is an
        // `.inspector` rather than a plain `.sheet`: at `.medium` the verse is still on
        // screen above the gloss, which is what the third pane does on a Mac. Without
        // these the inspector presents at full height on a phone and the passage being
        // annotated disappears behind its own annotation.
        //
        // `presentationBackgroundInteraction` is the other half: at `.medium` the reader
        // can tap the next line without dismissing the sheet first, so moving through a
        // scene stays one tap per passage the way it is on a Mac.
        #if !os(macOS)
        .presentationDetents([.medium, .large])
        .presentationBackgroundInteraction(.enabled(upThrough: .medium))
        #endif
    }

    // MARK: - Header

    #if os(macOS)
    @ViewBuilder
    private var header: some View {
        HStack(spacing: 10) {
            paneToggle(
                "the scene list", systemImage: "sidebar.leading", shortcut: "1",
                isVisible: showsNavigator
            ) {
                showsNavigator.toggle()
            }

            typefaceMenu

            if showsDiagnostics {
                Text("\(shortModelName) · on-device")
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: Capsule())
            }

            Spacer()

            loadStateIndicator

            diagnosticsMenu

            paneToggle(
                "the commentary", systemImage: "sidebar.trailing", shortcut: "2",
                isVisible: showsCommentary
            ) {
                setCommentary(!showsCommentary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
    #else
    /// The header's controls, in a navigation bar instead.
    ///
    /// The header's controls, in a navigation bar instead.
    ///
    /// Both `paneToggle`s are gone. The split view's own back button is ⌘1, and there is
    /// nothing for ⌘2 to toggle: the gloss sheet rises when a passage is selected and
    /// closes when it is swiped away, so a button that claimed to show or hide it would
    /// be lying about a preference the phone does not keep. Regenerate, Copy and Clear
    /// move into the overflow menu because ⌘R, ⌘C and Esc are not keys a phone has, and
    /// they are the reason this file, not `SceneReaderView`, owns that menu: the
    /// selection they act on lives here.
    @ToolbarContentBuilder
    private var readerToolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .topBarTrailing) {
            loadStateIndicator
            typefaceMenu
            overflowMenu
        }
    }

    @ViewBuilder
    private var overflowMenu: some View {
        Menu {
            Button("Regenerate", systemImage: "arrow.clockwise") { regenerate() }
                .disabled(selection == nil || isBusy)
            Button("Copy passage", systemImage: "doc.on.doc") { copySelection() }
                .disabled(selection == nil)
            Button("Clear selection", systemImage: "xmark") { clearSelection() }
                .disabled(selection == nil)
            Divider()
            Toggle("Show diagnostics", isOn: $diagnosticsPreference)
        } label: {
            Image(systemName: "ellipsis.circle")
        }
        .borderlessMenu()
        .accessibilityLabel("More")
    }

    /// ⌘C's counterpart. `SceneReaderView` cannot own this the way it owns
    /// `copyItem()`: the menu is in the navigation bar, which is this view's.
    private func copySelection() {
        guard let corpus, let sceneKey, let selection,
            let play = corpus.play(sceneKey.playID),
            let scene = corpus.scene(sceneKey),
            let range = selection.clamped(to: scene.lines)?.range
        else { return }
        copyToPasteboard(
            Citation.quotation(play: play, key: sceneKey, scene: scene, range: range))
    }

    /// Esc's counterpart. `SceneReaderView` cancels its own pending commit off the
    /// `selection` change, so clearing it from out here is enough.
    private func clearSelection() {
        selection = nil
        cancel()
    }
    #endif

    /// Where the model is, on both platforms. The first launch downloads gigabytes and
    /// an app that says nothing while it does looks broken, so the loading and failed
    /// states are never hidden by the diagnostics preference; idle and ready are noise
    /// and are.
    @ViewBuilder
    private var loadStateIndicator: some View {
        switch service.loadState {
        case .idle:
            if showsDiagnostics {
                Text("Idle").foregroundStyle(.secondary).font(.caption)
            }
        case .loading(let progress):
            HStack(spacing: 6) {
                if let progress, progress.totalUnitCount > 0 {
                    // The determinate bar is macOS only. A navigation bar already
                    // holds three controls at this point and has no 120 points to
                    // spare; the percentage alone carries the same information.
                    #if os(macOS)
                    ProgressView(value: progress.fractionCompleted)
                        .frame(width: 120)
                    #endif
                    Text("\(Int(progress.fractionCompleted * 100))%")
                        .font(.caption.monospacedDigit())
                } else {
                    ProgressView().controlSize(.small)
                    Text("Loading…").font(.caption)
                }
            }
            .foregroundStyle(.secondary)
        case .ready:
            if showsDiagnostics {
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
        case .failed(let message):
            // Also never hidden: without this the reader gets a play that silently
            // refuses to annotate anything.
            Label("Load failed", systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.red)
                .help(message)
        }
    }

    /// The face the play is set in. Sits with the reader's own controls rather than
    /// with the model capsule, because that is what it changes.
    ///
    /// A `Menu` and **not** a `Picker`: picker rows are selection tags with nowhere to
    /// hang a download or a retry affordance, and a per-row `.font()` would not
    /// preview anything anyway — inside a menu it goes through `NSMenuItem`, which
    /// takes its title font from the menu.
    ///
    /// `Toggle` rather than `Button`, because an `NSMenuItem` has exactly **one** image
    /// slot and SwiftUI puts a `Button` label's leading `Image` in it. A hand-drawn
    /// checkmark would therefore *displace* the status glyph on the selected row —
    /// which is precisely the row whose download state matters, since `choose(_:)`
    /// selects and downloads together. A `Toggle` puts the selection in the menu's own
    /// state column and leaves the image slot for the status.
    ///
    /// Nothing is ever disabled. A disabled `NSMenuItem` shows no tooltip on macOS, so
    /// a greyed row with `.help()` attached would communicate nothing at all, and a row
    /// that failed to download stays tappable so choosing it again retries.
    @ViewBuilder
    private var typefaceMenu: some View {
        Menu {
            // `offered`, not `allCases`: the two iOS-only families would be dead rows
            // on a Mac and Big Caslon and Garamond are unresolvable on a phone.
            ForEach(ReaderFont.offered, id: \.self) { font in
                Toggle(
                    isOn: Binding(get: { font == readerFont }, set: { _ in choose(font) })
                ) {
                    if let glyph = statusGlyph(font) {
                        Label(font.displayName, systemImage: glyph)
                    } else {
                        Text(font.displayName)
                    }
                }
                .help(fonts.failed[font] ?? "")
            }
        } label: {
            Image(systemName: "textformat")
        }
        // `Menu` ignores `.buttonStyle(.borderless)`, hence `.menuStyle`; and
        // `.fixedSize()` stops a borderless-button menu claiming more width than its
        // label needs.
        .borderlessMenu()
        .fixedSize()
        .help("The face the play is set in")
        .accessibilityLabel("Reader typeface")
    }

    /// Brings the model capsule, the load check and the status strip back.
    ///
    /// `Toggle` rather than a `Button` drawing its own checkmark, for the same reason
    /// `typefaceMenu` uses one: an `NSMenuItem` has a single image slot, and the menu's
    /// own state column is where a check belongs.
    ///
    /// No `keyboardShortcut`: the app has no menu bar of its own, and a shortcut on an
    /// item inside a borderless-button `Menu` is not reliably registered.
    /// `--diagnostics` covers launching noisy.
    ///
    /// macOS only. On a phone this toggle is one row of `overflowMenu`, which also
    /// carries the three commands that were keyboard shortcuts here.
    #if os(macOS)
    @ViewBuilder
    private var diagnosticsMenu: some View {
        Menu {
            Toggle("Show diagnostics", isOn: $diagnosticsPreference)
        } label: {
            Image(systemName: "ellipsis.circle")
        }
        .borderlessMenu()
        .fixedSize()
        .help("Show the model, the load check and the latency numbers")
        .accessibilityLabel("Diagnostics")
    }
    #endif

    /// `nil` for a family that is installed and idle, which is the ordinary case and
    /// wants no glyph at all.
    ///
    /// A filled arrow for in flight rather than a `ProgressView`: a menu row *is* an
    /// `NSMenuItem`, which takes an image and not a view, so a spinner has nowhere to
    /// render and silently disappears.
    private func statusGlyph(_ font: ReaderFont) -> String? {
        if fonts.failed[font] != nil { return "exclamationmark.triangle" }
        if fonts.downloading.contains(font) { return "arrow.down.circle.fill" }
        return fonts.isAvailable(font) ? nil : "arrow.down.circle"
    }

    /// Picking a family that is not installed starts its download as well as
    /// selecting it. Until the family lands the reader sees the system face, because
    /// `ReaderTypeface.familyName` stays nil until then.
    private func choose(_ font: ReaderFont) {
        readerFont = font
        if !fonts.isAvailable(font) { fonts.download(font) }
    }

    /// The two pane toggles. `.command` shortcuts rather than menu items, because the
    /// app has no menu of its own to add them to.
    @ViewBuilder
    private func paneToggle(
        _ name: String, systemImage: String, shortcut: KeyEquivalent, isVisible: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .foregroundStyle(isVisible ? AnyShapeStyle(.primary) : AnyShapeStyle(.tertiary))
        }
        .buttonStyle(.borderless)
        .keyboardShortcut(shortcut, modifiers: .command)
        .help("\(isVisible ? "Hide" : "Show") \(name) (⌘\(String(shortcut.character)))")
        .accessibilityLabel("\(isVisible ? "Hide" : "Show") \(name)")
    }

    private var shortModelName: String {
        service.modelID.split(separator: "/").last.map(String.init) ?? service.modelID
    }

    @ViewBuilder
    private func failure(_ message: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.callout)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Status strip

    /// With diagnostics on, every wait gets a name and the numbers that decide whether
    /// the prompt needs shedding are on screen rather than in a log. With them off the
    /// strip is nothing but a place for a failure to be reported.
    @ViewBuilder
    private var statusStrip: some View {
        if showsDiagnostics {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    switch displayedPhase {
                    case .idle:
                        Image(systemName: "circle.dashed").foregroundStyle(.tertiary)
                        Text(selection == nil ? "Idle" : "Done")
                    case .cached:
                        Image(systemName: "bolt.fill").foregroundStyle(.green)
                        Text("Cached")
                    case .summarizingScene:
                        ProgressView().controlSize(.small)
                        Text("Summarizing the scene…")
                    case .prefilling:
                        ProgressView().controlSize(.small)
                        if let prefill, prefill.total > 0 {
                            Text("Prefilling \(prefill.processed) / \(prefill.total)")
                        } else {
                            Text("Prefilling…")
                        }
                    case .reasoning:
                        ProgressView().controlSize(.small)
                        Text("Reasoning (not streamed by the framework)…")
                    case .streaming:
                        Image(systemName: "text.cursor")
                        Text("Annotating")
                    case .listingFollowUps:
                        ProgressView().controlSize(.small)
                        Text("Finding what to ask next…")
                    case .answering:
                        Image(systemName: "text.cursor")
                        Text("Answering")
                    }

                    Spacer()

                    if let timeToFirstToken {
                        Text(String(format: "first token %.2fs", timeToFirstToken))
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption)

                if phase == .prefilling, let prefill, prefill.total > 0 {
                    ProgressView(value: Double(prefill.processed), total: Double(prefill.total))
                }

                if let line = statsLine {
                    Text(line)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                        .textSelection(.enabled)
                }

                if let errorMessage {
                    Text(errorMessage)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        } else if let errorMessage {
            // A generation that failed still has to say so.
            Text(errorMessage)
                .font(.caption2)
                .foregroundStyle(.red)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
        }
    }

    /// The background scene summary is only worth naming when nothing the reader
    /// asked for is running; otherwise it would look like their own request
    /// stalling.
    private var displayedPhase: Phase {
        phase == .idle && service.isSummarizing ? .summarizingScene : phase
    }

    private var statsLine: String? {
        var parts: [String] = []
        if let promptTokens { parts.append("\(promptTokens) prompt tok") }
        if let stats {
            parts.append(String(format: "prefill %.0f tok/s", stats.promptTokensPerSecond))
            parts.append(String(format: "decode %.1f tok/s", stats.tokensPerSecond))
            parts.append(
                String(format: "%.2f GB peak", Double(stats.peakBytes) / 1_073_741_824))
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private var isBusy: Bool {
        switch phase {
        case .idle, .cached: false
        default: true
        }
    }

    // MARK: - Corpus

    private func loadCorpus() {
        guard corpus == nil else { return }
        do {
            let loaded = try CorpusLoader.load()
            corpus = loaded
            // Built once here rather than on demand: each `Cast` walks every line of
            // its play, which is not something to redo during a view update.
            casts = Dictionary(
                uniqueKeysWithValues: loaded.plays.map { ($0.id, Cast(play: $0)) })
            // Back where the reader left off, or the first scene on a cold start.
            //
            // A restored selection generates nothing on its own: `commit` is only
            // reached through `SceneReaderView.scheduleCommit`, which restore never
            // calls, so no model work runs unbidden on launch. The passage is there on
            // demand: clicking it, or ⌘R, glosses it.
            let opening = loaded.opening(from: ProgressStore.progress())
            sceneKey = opening.key
            selection = opening.selection
        } catch {
            corpusError = error.localizedDescription
        }
    }

    /// Records the position, for the next launch to open on. The stamp is the play's
    /// own source digest, so a rebuilt corpus can tell that the stored indices are no
    /// longer indices into the same lines.
    private func recordProgress() {
        guard let corpus, let sceneKey, let play = corpus.play(sceneKey.playID) else {
            return
        }
        ProgressStore.save(
            ReadingProgress(
                schemaVersion: ProgressStore.schemaVersion,
                key: sceneKey,
                selection: selection,
                corpusStamp: play.source.textSHA256))
    }

    private func cast(for play: Play) -> Cast {
        casts[play.id] ?? Cast(play: play)
    }

    private func openScene(_ key: SceneKey, in corpus: Corpus) {
        guard key != sceneKey else { return }
        sceneKey = key
        selection = nil
        clearAnnotation()
        if let scene = corpus.scene(key) {
            service.rememberSceneForPrewarm(key: key, scene: scene)
            if showsCommentary {
                service.prewarmSynopsis(key: key, scene: scene)
            }
        }
    }

    /// An arrow key that ran off the edge of a scene: `+1` opens the next scene on its
    /// first line, `-1` the previous one on its last.
    ///
    /// Scoped to the current play. `sceneKeys` is flat across the whole corpus, and
    /// running off the end of Hamlet into another play is a bigger jump than an arrow
    /// key should make, so both ends of the play just stop.
    ///
    /// Arriving this way generates nothing, exactly as opening the scene from the
    /// navigator does not: the annotation is cleared and the synopsis prewarms, and
    /// clicking the line or ⌘R glosses it. Committing on arrival would mean duplicating
    /// `SceneReaderView`'s 350 ms debounce here, and the reader has already left the line
    /// that scheduled it.
    private func stepScene(_ step: Int, in corpus: Corpus) {
        guard let sceneKey else { return }
        let keys = corpus.sceneKeys.filter { $0.playID == sceneKey.playID }
        guard let at = keys.firstIndex(of: sceneKey) else { return }
        let next = at + step
        guard keys.indices.contains(next), let scene = corpus.scene(keys[next]) else {
            return
        }

        openScene(keys[next], in: corpus)
        // After `openScene`, which nils it: the edge line being arrowed onto.
        guard !scene.lines.isEmpty else { return }
        selection = LineSelection(at: step > 0 ? 0 : scene.lines.count - 1)
    }

    // MARK: - Panes

    /// Hiding the commentary hides the status strip with it, so anything running
    /// would run unannounced — and a summary or an annotation nobody can read is
    /// model work spent on nothing. So the pane going away stops the current work,
    /// and coming back picks up where the reader left off.
    ///
    /// This is the *deliberate* way back, for a reader who wants the pane again
    /// without choosing a new passage; `commit(revealingCommentary:)` is the other,
    /// where pointing at a passage brings the pane along with it.
    private func setCommentary(_ shown: Bool) {
        guard shown != showsCommentary else { return }
        showsCommentary = shown

        guard shown else {
            cancel()
            return
        }
        guard let corpus, let sceneKey, let scene = corpus.scene(sceneKey),
            let play = corpus.play(sceneKey.playID)
        else { return }
        // A selection made while the pane was hidden is exactly what the reader
        // wants glossed now; with nothing selected, get the scene summary going so
        // it is ready for the first one.
        if let selection {
            commit(selection, play: play, key: sceneKey, scene: scene)
        } else {
            service.prewarmSynopsis(key: sceneKey, scene: scene)
        }
    }

    // MARK: - Annotation

    private func clearAnnotation() {
        context = nil
        commentary = ""
        followUps = []
        transcript = []
        phase = .idle
        promptTokens = nil
        prefill = nil
        timeToFirstToken = nil
        stats = nil
        errorMessage = nil
    }

    private func commit(
        _ selection: LineSelection, play: Play, key: SceneKey, scene: Scene,
        ignoringCache: Bool = false, revealingCommentary: Bool = false
    ) {
        // A pointer selection is a request to have that passage glossed, so it brings
        // the commentary back if the reader had it hidden. Selecting from the keyboard
        // does not: arrow keys are how you move through a scene with the play on its
        // own, and ⌘C on a hidden pane is how you copy a quote without generating
        // anything. ⌘2 still picks the pending selection up by hand.
        guard showsCommentary || revealingCommentary, service.isReady else { return }
        let synopsis = service.synopsis(for: key)
        guard
            let built = PassageContext.build(
                play: play, key: key, scene: scene, selection: selection,
                cast: cast(for: play),
                synopsis: synopsis?.text,
                synopsisIsPartial: synopsis?.isPartial ?? false)
        else { return }

        if options.showPrompt {
            print(Prompts.annotationRequest(built))
        }

        // Only now: there is a passage to put in the pane.
        if revealingCommentary { showsCommentary = true }
        clearAnnotation()
        context = built
        start(built, ignoringCache: ignoringCache)
    }

    /// ⌘R. Resolves the scene from state at the moment it runs rather than closing over
    /// the values this body was built with, because the shortcut hangs off a hidden
    /// `Button` inside `SceneReaderView`'s `.background`, and that button keeps the
    /// action it was created with: reached through a captured `play`/`key`/`scene`, ⌘R
    /// regenerated whichever scene the app opened on for the rest of the session. Reading
    /// `@State` through a stale copy of this struct is safe — the storage is shared.
    private func regenerate() {
        guard let corpus, let sceneKey, let selection,
            let play = corpus.play(sceneKey.playID), let scene = corpus.scene(sceneKey)
        else { return }
        commit(selection, play: play, key: sceneKey, scene: scene, ignoringCache: true)
    }

    private func start(_ built: PassageContext, ignoringCache: Bool) {
        let started = Date()
        Task {
            for await event in await service.annotate(
                built, ignoringCache: ignoringCache)
            {
                switch event {
                case .cached(let entry):
                    commentary = entry.commentary
                    promptTokens = entry.promptTokenCount
                case .phase(let value):
                    phase = value
                case .promptTokens(let count):
                    promptTokens = count
                case .prefill(let processed, let total):
                    prefill = (processed, total)
                    // A terminal (total, total) means the prompt is fully consumed;
                    // whatever follows before the first chunk is reasoning we never
                    // see.
                    if processed >= total, total > 0 { phase = .reasoning }
                case .commentary(let chunk):
                    if timeToFirstToken == nil {
                        timeToFirstToken = Date().timeIntervalSince(started)
                    }
                    commentary += chunk
                case .answer:
                    break
                case .followUps(let questions):
                    followUps = questions
                case .stats(let value):
                    stats = value
                case .followUpStats:
                    // The follow-up turn's numbers are for `--benchmark`; showing
                    // them here would replace the reader's own request with a
                    // 40-token prefill.
                    break
                case .failed(let message):
                    errorMessage = message
                }
            }
            prefill = nil
        }
    }

    /// Guarded here and not only by the pane's `.disabled`, because a Return keypress
    /// can race a phase change, and `AnnotationService` overwrites its active task per
    /// call rather than serializing — a second ask in flight would strand the first.
    private func ask(_ question: String) {
        let question = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty, !isBusy, context != nil else { return }

        let index = transcript.count
        transcript.append(.init(question: question, answer: ""))
        followUps = []

        Task {
            for await event in service.answer(question) {
                switch event {
                case .phase(let value):
                    phase = value
                case .answer(let chunk):
                    transcript[index].answer += chunk
                case .followUps(let questions):
                    followUps = questions
                case .stats(let value):
                    stats = value
                case .failed(let message):
                    errorMessage = message
                default:
                    break
                }
            }
        }
    }

    private func cancel() {
        Task { await service.stopActiveWork() }
        clearAnnotation()
    }
}
