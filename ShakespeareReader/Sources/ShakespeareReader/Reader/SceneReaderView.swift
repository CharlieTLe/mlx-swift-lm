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

    private static let space = "reader"
    private static let commitDelay = Duration.milliseconds(350)

    @State private var rowFrames: [Int: CGRect] = [:]
    @State private var isDragging = false
    @State private var commitTask: Task<Void, Never>?
    @FocusState private var isFocused: Bool

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
                .onMoveCommand { direction in
                    move(direction, scroller: scroller)
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
                    // `.id(key)` just rebuilt this view at the top of the scene. A
                    // selection already set on first appearance was restored from the
                    // last session, since the navigator clears it on every scene
                    // change, so put it back on screen. `scrollTo` reaches a row that
                    // the `LazyVStack` has not materialized, because `ForEach` over
                    // the enumerated lines declares every id up front.
                    if let head = selection?.head {
                        scroller.scrollTo(head, anchor: .center)
                    }
                }
            }
        }
        // A fresh view per scene: resets the scroll position to the top without
        // needing macOS 15's `ScrollPosition`.
        .id(key)
        .focusable()
        .focusEffectDisabled()
        .focused($isFocused)
        .onAppear { isFocused = true }
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
            isFirstSelected: selection?.range.lowerBound == index
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
                Text("Scene \(RomanNumeral.string(key.scene))")
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

    private func move(_ direction: MoveCommandDirection, scroller: ScrollViewProxy) {
        let step: Int
        switch direction {
        case .up: step = -1
        case .down: step = 1
        default: return
        }

        let extending = NSEvent.modifierFlags.contains(.shift)
        var next = selection ?? LineSelection(at: 0)
        let target = min(max(0, next.head + step), scene.lines.count - 1)
        if extending {
            next.extend(to: target)
        } else {
            next = LineSelection(at: target)
        }
        selection = next
        scroller.scrollTo(target, anchor: .center)
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
