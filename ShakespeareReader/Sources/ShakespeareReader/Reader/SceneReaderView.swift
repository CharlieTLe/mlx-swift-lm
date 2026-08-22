import Foundation
import SwiftUI

/// One scene, as a list of selectable lines.
///
/// **One scene at a time** is a deliberate constraint. It bounds rows to ~600
/// (Hamlet II.ii is the worst case), makes every selection intrinsically
/// scene-scoped so there is no cross-scene range to validate, and keeps the cache
/// key trivial. The cost is no continuous scroll through the play; the navigator
/// is how you move.
@MainActor
struct SceneReaderView: View {
    let play: Play
    let key: SceneKey
    let scene: Scene
    let cast: Cast

    @Binding var selection: LineSelection?

    /// What produced a commit. Only a pointer selection is a fresh request to have a
    /// passage glossed, so only a pointer selection brings a hidden commentary pane
    /// back; arrow keys are how you move through a scene with the play on its own.
    enum CommitOrigin: Sendable { case pointer, keyboard }

    /// Called 350 ms after the selection settles, and never while dragging, so
    /// sweeping through 40 lines starts exactly one generation. The origin says
    /// whether the reader pointed at the passage or arrowed onto it.
    let onCommit: (LineSelection, CommitOrigin) -> Void
    let onCancel: () -> Void
    let onRegenerate: () -> Void

    /// Rolls into the neighbouring scene when an arrow key runs off the edge of this one:
    /// `+1` forward, `-1` back. Stops at the ends of the play, since crossing into another
    /// is the navigator's job.
    let onStepScene: (Int) -> Void

    private static let space = "reader"
    private static let commitDelay = Duration.milliseconds(350)

    @State private var rowFrames: [Int: CGRect] = [:]
    @State private var isDragging = false
    /// The row a long press armed a sweep on, and the anchor the sweep extends from.
    /// `nil` means no sweep is armed, which is every ordinary pan. iOS only; macOS
    /// takes the drag bare and carries its anchor in `selectionDrag(from:)`.
    @State private var sweepAnchor: Int?
    @State private var commitTask: Task<Void, Never>?
    @FocusState private var isFocused: Bool

    /// The line an arrow move wants on screen, written by `move(_:)` and read back
    /// inside the `ScrollViewReader`, which is the only place a `ScrollViewProxy`
    /// exists. Cleared as each scene appears, since this view now outlives them.
    @State private var scrollTarget: Int?

    /// The face the play is set in. Injected here by `ContentView` and read by this
    /// view and `LineRow`; nothing outside the reader pane sees it.
    @Environment(\.readerTypeface) private var typeface

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            heading

            ScrollViewReader { scroller in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(scene.lines.enumerated()), id: \.offset) {
                            index, line in
                            row(index: index, line: line)
                        }
                    }
                    .padding(.vertical, 10)
                    .padding(.trailing, 12)
                    #if !os(macOS)
                    // Inside the scroll content on purpose: `SweepRecognizer` finds the
                    // scroll view by walking up from here, and a background of the
                    // `ScrollView` would sit outside it.
                    .background {
                        SweepRecognizer(
                            onBegan: { point in
                                guard let line = rowFrames.line(at: point) else { return }
                                sweepAnchor = line
                                extendDrag(from: line, to: point)
                            },
                            onChanged: { point in
                                guard let sweepAnchor else { return }
                                extendDrag(from: sweepAnchor, to: point)
                            },
                            onEnded: {
                                sweepAnchor = nil
                                if isDragging { endDrag() }
                            }
                        )
                    }
                    #endif
                }
                .coordinateSpace(name: Self.space)
                #if !os(macOS)
                // The other half of `SweepRecognizer`: it allows simultaneous
                // recognition so it never inhibits the pan, which means once a sweep is
                // under way the scroll view would otherwise still be panning under the
                // finger doing it. macOS is left alone deliberately — there the wheel is
                // not a `DragGesture`, so scrolling mid-drag is a feature rather than a
                // collision.
                .scrollDisabled(isDragging)
                #endif
                .onPreferenceChange(RowFramesKey.self) { rowFrames = $0 }
                .onChange(of: scrollTarget) {
                    // Where an arrow move lands. Scrolling can only happen in here,
                    // with the proxy, while `.onMoveCommand` has to sit on the
                    // focusable view itself — see the note beside it below. So the
                    // move writes a line index and this puts it on screen.
                    if let scrollTarget {
                        scroller.scrollTo(scrollTarget, anchor: .center)
                    }
                }
                .onChange(of: typeface) {
                    // `.id(key)` does not change on a font switch, which is right —
                    // the reader keeps their place rather than being thrown back to
                    // the top of the scene. But the scroll offset is preserved in
                    // *points* while the content just got taller or shorter, so the
                    // line they were reading drifts. Put it back under them.
                    scroller.scrollTo(selection?.head ?? 0, anchor: .center)
                }
                .onAppear {
                    // This subtree was just rebuilt at the top of the scene. A
                    // selection already set on first appearance was either restored
                    // from the last session or is the edge line an arrow rolled onto,
                    // so put it back on screen. `scrollTo` reaches a row that the
                    // `LazyVStack` has not materialized, because `ForEach` over the
                    // enumerated lines declares every id up front.
                    //
                    // `scrollTarget` outlives the scene now that the identity below is
                    // the scroll view's rather than the whole pane's; clearing it keeps
                    // a move onto the same index as the last scene's from being read as
                    // "no change" and skipping its scroll.
                    scrollTarget = nil
                    if let head = selection?.head {
                        scroller.scrollTo(head, anchor: .center)
                    }
                }
                // A fresh scroll view per scene: resets the scroll position to the top
                // without needing macOS 15's `ScrollPosition`.
                //
                // Here, and **not** on the whole pane, which took the `.focusable()`
                // responder below down with it on every scene change: an arrow-key roll
                // then landed in the next scene with `isFocused` still reading true —
                // the band even stayed accent — and every key dead, arrows and Esc
                // alike, until the reader clicked a line.
                .id(key)
            }
        }
        // macOS only, and this is the whole reason the pane is focusable at all: the
        // three responder-chain commands below are offered to the focused view. iOS has
        // none of them, and asking for focus there had a visible cost. The reader view
        // became first responder with no input view of its own, so the software keyboard
        // rose over the bottom third of the play every time a `Menu` in the navigation
        // bar opened. ⌘R still works with a hardware keyboard because it hangs off the
        // `Button` below, which needs no focus.
        #if os(macOS)
        .focusable()
        .focusEffectDisabled()
        .focused($isFocused)
        .onAppear { isFocused = true }
        #endif
        .onChange(of: key) {
            // This view now outlives the scene, so a commit scheduled for the line the
            // reader is leaving would land in a pane `ContentView` has just cleared for
            // the new one. The roll cancels it in `move(_:)` for the same reason; this
            // covers the navigator.
            commitTask?.cancel()
        }
        .onChange(of: selection) {
            // A selection cleared from *outside* this view, where the iPhone toolbar's
            // Clear has no reach into `commitTask`, must take a pending commit with it or
            // the debounce fires 350 ms later and glosses the passage that was just
            // dismissed. Esc cancels explicitly as well, since that path wants the
            // cancellation to be immediate rather than a change notification behind.
            if selection == nil { commitTask?.cancel() }
        }
        // Beside `.focusable()` and **not** on the `ScrollView` inside, where it used to
        // be and never fired: a command handler is offered to the focused view and then
        // to its ancestors, never to its descendants, so the arrows were dead even with
        // the pane focused while `.onExitCommand` here worked. Hence `scrollTarget`
        // rather than the proxy, which only exists inside the `ScrollViewReader`.
        //
        // macOS only, all three: these are responder-chain commands with no iOS
        // equivalent. Their touch replacements are in `ContentView`'s reader toolbar
        // (Clear, Copy) and in `selectionDrag(from:)` below.
        #if os(macOS)
        .onMoveCommand { move($0) }
        .onExitCommand { clearSelection() }
        .onCopyCommand { [copyItem()].compactMap { $0 } }
        #endif
        .background {
            // Keyboard shortcuts need a control to hang off. Zero-opacity rather
            // than `.hidden()`, which removes it from the hierarchy along with its
            // shortcut.
            //
            // Kept on iOS too: it costs nothing and works with a hardware keyboard.
            // The visible affordance there is the reader toolbar's Regenerate item.
            Button("Regenerate", action: onRegenerate)
                .keyboardShortcut("r", modifiers: .command)
                .opacity(0)
                .frame(width: 0, height: 0)
                .accessibilityHidden(true)
        }
    }

    // MARK: - Rows

    @ViewBuilder
    private func row(index: Int, line: Line) -> some View {
        LineRow(
            index: index,
            line: line,
            display: line.speaker.map(cast.display),
            isSelected: selection?.contains(index) ?? false,
            isFirstSelected: selection?.range.lowerBound == index,
            hasFocus: bandHasFocus
        )
        .reportRowFrame(index: index, space: Self.space)
        // The `count: 2` gesture must be attached *before* the `count: 1` gesture
        // or SwiftUI resolves every double-click as two single clicks.
        .onTapGesture(count: 2) {
            select(LineSelection.speech(at: index, in: scene))
        }
        .onTapGesture {
            // Read the modifiers synchronously from the event rather than through
            // `TapGesture().modifiers(.shift)`, which is unreliable here and would
            // need a separate gesture per modifier. Always false on iOS, where the
            // long press in `selectionDrag(from:)` is what extends a selection.
            if isShiftKeyDown, var extended = selection {
                extended.extend(to: index)
                select(extended)
            } else {
                select(LineSelection(at: index))
            }
        }
        #if os(macOS)
        .gesture(selectionDrag(from: index))
        #endif
        // iOS attaches nothing here. A `DragGesture` on these rows, in any shape, stops
        // the scene from scrolling; the touch equivalent of the sweep is the
        // `SweepRecognizer` on the scroll view instead. See its own note.
    }

    /// Whether the selection band should draw as though the pane holds the keyboard.
    ///
    /// `.focusable()` never becomes focused on a phone with no hardware keyboard, so
    /// the band would render in the unfocused grey for good. That grey means "the
    /// arrows are pointed somewhere else", and on a touch device there is nowhere else
    /// for them to point.
    private var bandHasFocus: Bool {
        #if os(macOS)
        isFocused
        #else
        true
        #endif
    }

    /// Sweeping a range of lines out with the pointer.
    ///
    /// Attached per row so the anchor is this row's own index; only the head has to be
    /// hit-tested through `rowFrames`. No edge auto-scroll: on macOS the scroll wheel
    /// and two-finger scroll are not `DragGesture` events, so the reader can scroll
    /// mid-drag without the drag noticing.
    ///
    /// That same fact is why macOS can take the drag bare and iOS cannot. A touch pan
    /// *is* a `DragGesture`, so a bare `minimumDistance: 4` on every row would win the
    /// vertical gesture against the enclosing `ScrollView` and the scene would not
    /// scroll at all. Sequencing it behind a long press is what makes
    /// press-and-hold-then-drag the touch affordance that replaces shift-click, but it
    /// is *not* what gives the pan back: see the note at the attachment site, which is
    /// simultaneous for that reason.
    ///
    /// Sweeping a range of lines out with the pointer. macOS only — the touch
    /// equivalent is the pair of simultaneous gestures at the attachment site, which a
    /// sequenced gesture cannot do without taking the scroll view's pan with it.
    ///
    /// Attached per row so the anchor is this row's own index; only the head has to be
    /// hit-tested through `rowFrames`. No edge auto-scroll: on macOS the scroll wheel
    /// and two-finger scroll are not `DragGesture` events, so the reader can scroll
    /// mid-drag without the drag noticing. That same fact is why macOS can take the
    /// drag bare and iOS cannot: a touch pan *is* a `DragGesture`.
    #if os(macOS)
    private func selectionDrag(from index: Int) -> some Gesture {
        DragGesture(minimumDistance: 4, coordinateSpace: .named(Self.space))
            .onChanged { extendDrag(from: index, to: $0.location) }
            .onEnded { _ in endDrag() }
    }
    #endif

    /// The head of a drag moved. `index` is the row the gesture started on, which is
    /// the anchor, and `location` is hit-tested against `rowFrames` for the head.
    private func extendDrag(from index: Int, to location: CGPoint) {
        if !isDragging {
            isDragging = true
            selection = LineSelection(at: index)
            // Same reason as `select(_:)`: the pane has to hold the keyboard for the
            // arrows to keep working after a drag. Inside the guard, because this runs
            // per event and focus is not worth re-requesting at frame rate.
            isFocused = true
        }
        selection?.extend(to: rowFrames.line(at: location) ?? index)
    }

    private func endDrag() {
        isDragging = false
        if let selection { scheduleCommit(selection, from: .pointer) }
    }

    @ViewBuilder
    private var heading: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text("Act \(RomanNumeral.string(key.act))")
                Text("·")
                Text(SceneLabel.string(key.scene))
            }
            .font(typeface.actSceneHeading)
            .tracking(typeface.actSceneTracking)
            Text(scene.setting)
                .font(typeface.sceneSetting)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 8)
    }

    // MARK: - Selection

    private func select(_ new: LineSelection) {
        selection = new
        // Clicking in the reader is what hands the keyboard back to it. A tap gesture
        // is not an `NSControl`, so it never moves first responder on its own: after a
        // click in the navigator's `List` the sidebar keeps it, and the arrows, Esc,
        // ⌘C and ⌘R here are all dead until something takes it back. `.onAppear` fires
        // once, outside `.id(key)`, so it cannot be that something.
        //
        // `isFocused` reads stale inside this closure — the write resolves on commit —
        // so nothing here may branch on it. Setting it beside `selection` is not a
        // hazard: both land in the one transaction.
        isFocused = true
        scheduleCommit(new, from: .pointer)
    }

    /// Debounces so a sweep through 40 lines starts one generation, not 40, and
    /// so a click that is really the start of a shift-click does not fire first.
    private func scheduleCommit(_ new: LineSelection, from origin: CommitOrigin) {
        commitTask?.cancel()
        commitTask = Task {
            try? await Task.sleep(for: Self.commitDelay)
            guard !Task.isCancelled, !isDragging else { return }
            onCommit(new, origin)
        }
    }

    /// Where an arrow key lands. macOS only: `MoveCommandDirection` is not available on
    /// iOS at all, and `.onMoveCommand`, its only caller, is not either. A hardware
    /// keyboard attached to a phone therefore does not move the selection. The touch
    /// equivalents are tap and press-and-hold-then-drag.
    #if os(macOS)
    private func move(_ direction: MoveCommandDirection) {
        let step: Int
        switch direction {
        case .up: step = -1
        case .down: step = 1
        default: return
        }

        // Read the modifiers from the event for the same reason the click path does at
        // `row(index:line:)`: SwiftUI does not report them on a move command.
        let extending = isShiftKeyDown
        guard
            let next = LineSelection.moved(
                from: selection, by: step, extending: extending, in: scene.lines)
        else {
            // Off the edge of the scene, with nothing to extend: roll into the next one.
            // Cancelling first matters — `.id(key)` replaces this subtree on a roll, and
            // a commit already scheduled for the line being left would land in a pane
            // `ContentView` has just cleared for the new scene.
            commitTask?.cancel()
            onStepScene(step)
            return
        }

        selection = next
        scrollTarget = next.head
        scheduleCommit(next, from: .keyboard)
    }
    #endif

    // MARK: - Copy

    /// Esc, and the reader toolbar's Clear on iOS.
    private func clearSelection() {
        commitTask?.cancel()
        selection = nil
        onCancel()
    }

    #if os(macOS)
    /// The selected lines with the citation appended, wrapped for the responder
    /// chain. iOS copies the same string through `UIPasteboard` from
    /// `ContentView`'s reader toolbar, since `.onCopyCommand` has no counterpart
    /// there.
    private func copyItem() -> NSItemProvider? {
        guard let selection, let range = selection.clamped(to: scene.lines)?.range
        else { return nil }
        let text = Citation.quotation(
            play: play, key: key, scene: scene, range: range)
        return NSItemProvider(object: text as NSString)
    }
    #endif
}
