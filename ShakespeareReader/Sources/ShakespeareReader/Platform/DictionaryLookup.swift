import SwiftUI

#if os(macOS)
import AppKit
import CoreServices
#endif

/// The system dictionary, which is the third platform surface this app has — joining the
/// shims in `PlatformCompat` and the two places where the *layout* differs
/// (`ContentView`'s container and `SceneReaderView`'s pointer gestures). It lives here
/// for that reason and not in the reader, which otherwise draws the same code on both
/// platforms.
///
/// macOS only for now. `showDefinition(for:at:)` is an `NSView` method with no UIKit
/// counterpart — the iOS equivalent is `UIReferenceLibraryViewController`, presented as a
/// sheet rather than popped over a point — and word lookup as a whole is not wired up on
/// iOS yet, because a long press there already arms `SweepRecognizer`.

/// The `NSView` the definition panel is popped over.
///
/// A panel needs a view and a point, and SwiftUI has neither to offer, so one view is
/// installed in the reader pane's background and reached back through here. It is held
/// `weak`: the view belongs to its window, and a reference from app state that outlives a
/// pane rebuild would be the only thing keeping a dead view alive.
@MainActor
final class DictionaryAnchor {

    /// The coordinate space the point passed to `showDefinition(_:at:)` is measured in.
    ///
    /// Named rather than `.local`, because the point is resolved inside a `LineRow` and
    /// spent against a view in the pane's background — two different `.local`s. Both ends
    /// name this, and `ContentView` registers it on the same view the anchor backs, so the
    /// space's origin and the anchor's origin are the same corner.
    static let space = "dictionaryAnchor"

    #if os(macOS)
    fileprivate weak var view: NSView?
    #endif

    /// Whether the dictionary has anything to say about `term`.
    ///
    /// The "Look Up" item is gated on this, because `showDefinition` over a word with no
    /// entry opens an empty panel, and an empty panel is worse than an item that was never
    /// offered. It is worth knowing which words those are: measured against the active
    /// dictionaries, `crowner`, `quietus`, `argal`, `to-morrow` and even `’tis` and `o’er`
    /// all have entries, while `undiscover’d`, `well-a-day` and `offendendo` do not. So
    /// this excludes elided *participles* and hyphenated coinages rather than elisions as a
    /// class — which is also why it is asked rather than guessed at.
    ///
    /// `static`, unlike `showDefinition(_:at:)`: asking the dictionary a question needs no
    /// view, so a row building its menu needs no anchor threaded down to it.
    static func hasDefinition(for term: String) -> Bool {
        #if os(macOS)
        guard !term.isEmpty else { return false }
        let range = CFRange(location: 0, length: term.utf16.count)
        return DCSCopyTextDefinition(nil, term as CFString, range) != nil
        #else
        false
        #endif
    }

    /// Pops the system definition panel over the word.
    ///
    /// `point` is the word's **baseline origin** in `space`, which is to say in SwiftUI's
    /// coordinates with y downwards. No conversion happens here because the anchor view
    /// declares itself flipped: agreeing with SwiftUI once, in `AnchorView`, is cheaper
    /// than converting at every call site.
    func showDefinition(_ term: String, at point: CGPoint) {
        #if os(macOS)
        guard let view, !term.isEmpty else { return }
        view.showDefinition(for: NSAttributedString(string: term), at: point)
        #endif
    }
}

/// Installs the anchor view, which fills the pane it backs so that its bounds *are*
/// `DictionaryAnchor.space`. Draws nothing and takes no clicks.
@MainActor
struct DictionaryAnchorView: View {
    let anchor: DictionaryAnchor

    var body: some View {
        #if os(macOS)
        Representable(anchor: anchor).accessibilityHidden(true)
        #else
        EmptyView()
        #endif
    }

    #if os(macOS)
    private struct Representable: NSViewRepresentable {
        let anchor: DictionaryAnchor

        func makeNSView(context: Context) -> AnchorView {
            let view = AnchorView()
            anchor.view = view
            return view
        }

        func updateNSView(_ view: AnchorView, context: Context) {
            // Reclaimed on every update, so a pane that was rebuilt — hiding the
            // navigator does it — hands back the view that is actually on screen.
            anchor.view = view
        }
    }

    final class AnchorView: NSView {
        /// So a point in SwiftUI's coordinates is a point in this view's, with no flip
        /// anywhere in the path from the hovered word to the panel.
        override var isFlipped: Bool { true }

        /// **Load-bearing.** A representable in a `.background` is a real `NSView` behind
        /// the pane, and AppKit hit-tests subviews before the content SwiftUI draws over
        /// them, so a plain view of this size would swallow every click in the reader —
        /// selection, shift-click, double-click and the sweep with it. This view is a
        /// coordinate system and a place to hang a panel, and nothing else.
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }
    #endif
}
