import AppKit
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
                }
                .coordinateSpace(name: Self.space)
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
        .focusable()
        .focusEffectDisabled()
        .focused($isFocused)
        .onAppear { isFocused = true }
        .onChange(of: key) {
            // This view now outlives the scene, so a commit scheduled for the line the
            // reader is leaving would land in a pane `ContentView` has just cleared for
            // the new one. The roll cancels it in `move(_:)` for the same reason; this
            // covers the navigator.
            commitTask?.cancel()
        }
        // Beside `.focusable()` and **not** on the `ScrollView` inside, where it used to
        // be and never fired: a command handler is offered to the focused view and then
        // to its ancestors, never to its descendants, so the arrows were dead even with
        // the pane focused while `.onExitCommand` here worked. Hence `scrollTarget`
        // rather than the proxy, which only exists inside the `ScrollViewReader`.
        .onMoveCommand { move($0) }
        .onExitCommand {
            commitTask?.cancel()
            selection = nil
            onCancel()
        }
        .onCopyCommand { [copyItem()].compactMap { $0 } }
        .background {
            // Keyboard shortcuts need a control to hang off. Zero-opacity rather
            // than `.hidden()`, which removes it from the hierarchy along with its
            // shortcut.
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
            hasFocus: isFocused
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
            // need a separate gesture per modifier.
            if NSEvent.modifierFlags.contains(.shift), var extended = selection {
                extended.extend(to: index)
                select(extended)
            } else {
                select(LineSelection(at: index))
            }
        }
        // Attached per row so the anchor is this row's own index; only the head
        // has to be hit-tested through `rowFrames`. No edge auto-scroll: on macOS
        // the scroll wheel and two-finger scroll are not `DragGesture` events, so
        // the reader can scroll mid-drag without the drag noticing.
        .gesture(
            DragGesture(minimumDistance: 4, coordinateSpace: .named(Self.space))
                .onChanged { value in
                    if !isDragging {
                        isDragging = true
                        selection = LineSelection(at: index)
                        // Same reason as `select(_:)`: the pane has to hold the
                        // keyboard for the arrows to keep working after a drag.
                        // Inside the guard, because `onChanged` runs per event and
                        // focus is not worth re-requesting at frame rate.
                        isFocused = true
                    }
                    let head = rowFrames.line(at: value.location) ?? index
                    selection?.extend(to: head)
                }
                .onEnded { _ in
                    isDragging = false
                    if let selection { scheduleCommit(selection, from: .pointer) }
                }
        )
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

    private func move(_ direction: MoveCommandDirection) {
        let step: Int
        switch direction {
        case .up: step = -1
        case .down: step = 1
        default: return
        }

        // Read the modifiers from the event for the same reason the click path does at
        // `row(index:line:)`: SwiftUI does not report them on a move command.
        let extending = NSEvent.modifierFlags.contains(.shift)
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

    // MARK: - Copy

    /// The selected lines with the citation appended, which is what makes a quote
    /// pasted into notes traceable.
    private func copyItem() -> NSItemProvider? {
        guard let selection, let range = selection.clamped(to: scene.lines)?.range
        else { return nil }

        var text = scene.lines[range]
            .map { line -> String in
                if line.isDirection {
                    return "[\(line.plainText)]"
                }
                return line.startsSpeech && line.speaker != nil
                    ? "\(line.speaker!). \(line.text)" : line.text
            }
            .joined(separator: "\n")
        text += "\n\n" + Citation.string(play: play, key: key, scene: scene, range: range)
        return NSItemProvider(object: text as NSString)
    }
}
